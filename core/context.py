"""
Context Builder Module

This is where we decide WHAT goes to the LLM. Key concerns:
1. Token/character limits - can't send everything
2. Relevance - most useful content first  
3. Source diversity - don't over-rely on one source
4. Metadata preservation - need to track what came from where for citations

Design decisions:
- Character-based limits (simpler than token counting, close enough)
- Score-based ranking combining relevance, recency, and diversity
- Keep metadata attached to each snippet for citation mapping
"""

from datetime import datetime
from typing import Optional
import re

# Reasonable limits
DEFAULT_MAX_CONTEXT_CHARS = 12000  # ~3000 tokens roughly
MAX_SNIPPET_CHARS = 2000  # Don't let any single snippet dominate


def build_context(
    fetched_pages: list,
    query: str,
    conversation_summary: str = "",
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS
) -> dict:
    """
    Build the context to send to the LLM
    
    Args:
        fetched_pages: List of page fetch results with content
        query: Current user query (for relevance scoring)
        conversation_summary: Summary of prior conversation if any
        max_chars: Maximum characters for web context
    
    Returns:
        dict with:
            - snippets: list of selected snippets with metadata
            - total_chars: total character count
            - sources_used: number of unique sources
            - truncated: whether we hit the limit
    """
    # Extract and score all content chunks
    all_chunks = []
    
    for page in fetched_pages:
        if not page.get("success") or not page.get("content"):
            continue
            
        # Split page content into chunks
        chunks = _split_into_chunks(page["content"], max_chunk_size=MAX_SNIPPET_CHARS)
        
        for idx, chunk in enumerate(chunks):
            # Score this chunk
            score = _score_chunk(
                chunk=chunk,
                query=query,
                position_in_page=idx,
                total_chunks=len(chunks)
            )
            
            all_chunks.append({
                "text": chunk,
                "score": score,
                "url": page["url"],
                "title": page.get("title", "Untitled"),
                "domain": page.get("domain", "unknown"),
                "retrieved_at": page.get("retrieved_at", ""),
                "chunk_index": idx
            })
    
    # Sort by score
    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    # Select chunks up to limit, ensuring source diversity
    selected = []
    used_domains = {}
    total_chars = 0
    
    # Reserve space for conversation summary if present
    context_budget = max_chars
    if conversation_summary:
        context_budget -= len(conversation_summary) + 100  # Buffer for formatting
    
    for chunk in all_chunks:
        chunk_len = len(chunk["text"])
        
        # Check if adding this would exceed limit
        if total_chars + chunk_len > context_budget:
            continue
        
        # Soft limit on chunks per domain (don't let one source dominate)
        domain = chunk["domain"]
        if used_domains.get(domain, 0) >= 3:
            # Skip if we already have 3 chunks from this domain
            # Unless it's scored much higher than average
            avg_score = sum(c["score"] for c in selected) / len(selected) if selected else 0
            if chunk["score"] < avg_score * 1.5:
                continue
        
        selected.append(chunk)
        used_domains[domain] = used_domains.get(domain, 0) + 1
        total_chars += chunk_len
    
    return {
        "snippets": selected,
        "total_chars": total_chars,
        "sources_used": len(used_domains),
        "domains": list(used_domains.keys()),
        "truncated": len(selected) < len(all_chunks),
        "chunks_evaluated": len(all_chunks),
        "chunks_selected": len(selected)
    }


def _split_into_chunks(content: str, max_chunk_size: int = 1500) -> list:
    """
    Split content into manageable chunks
    
    Tries to split at paragraph boundaries for coherent chunks
    """
    if not content:
        return []
    
    # Split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n', content)
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If this paragraph alone is too big, split it
        if len(para) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Split long paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if len(current_chunk) + len(sent) < max_chunk_size:
                    current_chunk += sent + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent + " "
        elif len(current_chunk) + len(para) < max_chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def _score_chunk(
    chunk: str,
    query: str,
    position_in_page: int,
    total_chunks: int
) -> float:
    """
    Score a chunk for relevance
    
    Factors:
    - Keyword overlap with query (primary signal)
    - Position in page (earlier content often more relevant)
    - Content quality signals (length, not mostly boilerplate)
    """
    score = 0.0
    
    # Keyword overlap (0-1)
    query_terms = set(_normalize_text(query).split())
    chunk_terms = set(_normalize_text(chunk).split())
    
    if query_terms:
        overlap = len(query_terms & chunk_terms)
        keyword_score = overlap / len(query_terms)
        score += keyword_score * 0.6  # Weight: 60%
    
    # Position score (earlier = better)
    if total_chunks > 0:
        position_score = 1.0 - (position_in_page / total_chunks)
        score += position_score * 0.2  # Weight: 20%
    
    # Length score (very short or very long chunks are less useful)
    chunk_len = len(chunk)
    if 200 < chunk_len < 1500:
        length_score = 1.0
    elif chunk_len < 200:
        length_score = chunk_len / 200
    else:
        length_score = max(0.5, 1.0 - (chunk_len - 1500) / 2000)
    score += length_score * 0.2  # Weight: 20%
    
    return round(score, 3)


def _normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def format_context_for_llm(
    context: dict,
    conversation_summary: str = ""
) -> str:
    """
    Format the built context into a string for the LLM
    
    Includes source markers for citation tracking
    """
    parts = []
    
    # Add conversation summary if present
    if conversation_summary:
        parts.append("=== Prior Conversation Summary ===")
        parts.append(conversation_summary)
        parts.append("")
    
    parts.append("=== Web Research Results ===")
    parts.append("")
    
    # Group snippets by source for readability
    by_source = {}
    for snippet in context["snippets"]:
        url = snippet["url"]
        if url not in by_source:
            by_source[url] = {
                "title": snippet["title"],
                "domain": snippet["domain"],
                "url": url,
                "chunks": []
            }
        by_source[url]["chunks"].append(snippet["text"])
    
    for idx, (url, source) in enumerate(by_source.items(), 1):
        parts.append(f"[Source {idx}] {source['title']} ({source['domain']})")
        parts.append(f"URL: {url}")
        parts.append("")
        for chunk in source["chunks"]:
            parts.append(chunk)
            parts.append("")
        parts.append("---")
        parts.append("")
    
    return "\n".join(parts)


def summarize_conversation(messages: list, max_length: int = 500) -> str:
    """
    Create a brief summary of conversation history
    
    Used when conversation gets too long to include in full
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        max_length: Maximum length of summary
    
    Returns:
        String summary of the conversation
    """
    if not messages:
        return ""
    
    # For now, just extract key points from user messages
    # A production system would use the LLM to summarize
    user_messages = [m["content"] for m in messages if m["role"] == "user"]
    
    summary_parts = ["Previous topics discussed:"]
    total_len = len(summary_parts[0])
    
    for msg in user_messages[-5:]:  # Last 5 user messages
        # Take first sentence or first 100 chars
        short = msg[:100].split('.')[0]
        if total_len + len(short) < max_length:
            summary_parts.append(f"- {short}")
            total_len += len(short) + 3
    
    return "\n".join(summary_parts)


# Test when running directly
if __name__ == "__main__":
    # Simulate some fetched pages
    test_pages = [
        {
            "success": True,
            "url": "https://example.com/article1",
            "title": "Understanding RAG Systems",
            "domain": "example.com",
            "content": "Retrieval Augmented Generation (RAG) is a technique that combines retrieval and generation. It works by first retrieving relevant documents from a knowledge base, then using those documents as context for a language model to generate responses. This approach helps reduce hallucinations and provides more accurate, grounded answers.",
            "retrieved_at": datetime.now().isoformat()
        },
        {
            "success": True,
            "url": "https://another.com/rag-guide",
            "title": "RAG Implementation Guide",
            "domain": "another.com",
            "content": "Implementing RAG requires several components: a document store, an embedding model, a retriever, and a generator. The retriever finds relevant documents based on the query. The generator then uses these documents to produce a response. Popular tools include FAISS for vector storage and various LLMs for generation.",
            "retrieved_at": datetime.now().isoformat()
        }
    ]
    
    context = build_context(test_pages, query="what is RAG")
    print(f"Built context with {context['chunks_selected']} snippets from {context['sources_used']} sources")
    print(f"Total chars: {context['total_chars']}")
    print("\nFormatted context:")
    print(format_context_for_llm(context))
