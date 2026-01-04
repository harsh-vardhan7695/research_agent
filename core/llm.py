"""
LLM Module - Google Gemini Integration

Handles all LLM interactions:
- Research planning
- Answer generation with citations
- Conversation summarization

Using Gemini because:
- Free tier is generous (15 RPM, 1M tokens/day on gemini-1.5-flash)
- Good at following instructions
- Supports structured output when needed
"""

import os
import google.generativeai as genai
from typing import Generator, Optional
import json
import re


def configure_gemini():
    """Configure the Gemini API with the API key"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment. "
            "Get one free at https://aistudio.google.com/apikey"
        )
    genai.configure(api_key=api_key)


def get_model(model_name: str = "models/gemini-2.5-flash"):
    """
    Get a Gemini model instance

    Using 2.5-flash by default - it's fast and free tier is generous.
    For complex research, could switch to models/gemini-2.5-pro
    """
    configure_gemini()
    return genai.GenerativeModel(model_name)


def generate_research_plan(query: str, conversation_context: str = "") -> str:
    """
    Generate a research plan for the query
    
    Returns a structured plan of what to search for and why
    """
    model = get_model()
    
    prompt = f"""You are a research assistant planning how to answer a question.

User Question: {query}

{f"Context from previous conversation: {conversation_context}" if conversation_context else ""}

Create a brief research plan with:
1. What the question is really asking (in one line)
2. 2-3 specific search queries to find relevant information
3. What types of sources would be most helpful

Keep it concise - just the plan, no extra commentary.
Format as:
UNDERSTANDING: <what the question is asking>
SEARCHES:
- <search query 1>
- <search query 2>
- <search query 3>
SOURCES NEEDED: <types of sources>
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Fallback to simple plan if LLM fails
        return f"""UNDERSTANDING: {query}
SEARCHES:
- {query}
- {query} explained
SOURCES NEEDED: General web sources"""


def extract_search_queries(plan: str) -> list:
    """Extract search queries from a research plan"""
    queries = []
    
    # Look for lines starting with - after SEARCHES:
    in_searches = False
    for line in plan.split('\n'):
        if 'SEARCHES:' in line.upper():
            in_searches = True
            continue
        if in_searches:
            if line.strip().startswith('-'):
                query = line.strip().lstrip('-').strip()
                if query:
                    queries.append(query)
            elif line.strip() and not line.strip().startswith('-'):
                # Hit a new section
                break
    
    # Fallback if parsing failed
    if not queries:
        return [plan.split('\n')[0]]  # Use first line as query
    
    return queries[:3]  # Max 3 queries


def generate_answer_stream(
    query: str,
    context: str,
    conversation_history: list = None
) -> Generator[str, None, None]:
    """
    Generate an answer with citations, streaming the response
    
    Args:
        query: User's question
        context: Formatted web research context
        conversation_history: List of prior messages
    
    Yields:
        Chunks of the response as they're generated
    """
    model = get_model()
    
    # Build the system instruction
    system_instruction = """You are a research assistant providing well-sourced answers.

IMPORTANT RULES:
1. Base your answer ONLY on the provided web research results
2. Cite sources for every major claim using format: [Title — domain](url)
3. If sources conflict, explicitly note the disagreement and cite both
4. If evidence is insufficient, say so clearly and suggest what additional research might help
5. Be direct and informative - no fluff
6. If the research doesn't contain relevant information, say "Based on the available research, I couldn't find information about X" rather than making things up"""

    # Build conversation context
    messages = []
    if conversation_history:
        for msg in conversation_history[-4:]:  # Last 4 messages for context
            messages.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [msg["content"]]
            })
    
    # Current query with context
    current_prompt = f"""Based on the following web research, answer the user's question.

{context}

User Question: {query}

Provide a comprehensive answer with citations in the format [Title — domain](url)."""

    messages.append({
        "role": "user",
        "parts": [current_prompt]
    })
    
    try:
        # Create chat with system instruction
        chat = model.start_chat(history=messages[:-1] if len(messages) > 1 else [])
        
        # Stream the response
        response = chat.send_message(
            messages[-1]["parts"][0],
            stream=True,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,  # Lower temp for factual responses
                max_output_tokens=2048
            )
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        yield f"\n\nError generating response: {str(e)}"


def generate_answer(
    query: str,
    context: str,
    conversation_history: list = None
) -> str:
    """
    Non-streaming version of answer generation
    
    Useful for evaluation where we need the full response at once
    """
    full_response = ""
    for chunk in generate_answer_stream(query, context, conversation_history):
        full_response += chunk
    return full_response


def summarize_for_context(
    conversation_history: list,
    max_length: int = 500
) -> str:
    """
    Use LLM to summarize conversation history
    
    Called when conversation gets too long to include in full
    """
    if not conversation_history or len(conversation_history) < 3:
        return ""
    
    model = get_model()
    
    # Format history for summarization
    history_text = ""
    for msg in conversation_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content'][:200]}...\n\n"
    
    prompt = f"""Summarize this conversation in 2-3 sentences, focusing on:
- Main topics discussed
- Key facts established
- Any user preferences or context mentioned

Conversation:
{history_text}

Summary:"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=200
            )
        )
        return response.text[:max_length]
    except Exception:
        # Fallback to simple extraction
        topics = [m["content"][:50] for m in conversation_history if m["role"] == "user"]
        return f"Previous topics: {', '.join(topics[-3:])}"


def check_for_conflicts(context: str) -> list:
    """
    Analyze context for conflicting information
    
    Returns list of potential conflicts found
    """
    model = get_model()
    
    prompt = f"""Analyze this research context for conflicting information.
List any claims where different sources disagree.

Context:
{context}

If there are conflicts, list them as:
- Conflict: <topic> | Source 1 says X | Source 2 says Y

If no conflicts found, respond with: NO_CONFLICTS_FOUND"""

    try:
        response = model.generate_content(prompt)
        text = response.text
        
        if "NO_CONFLICTS_FOUND" in text:
            return []
        
        conflicts = []
        for line in text.split('\n'):
            if line.strip().startswith('-') or line.strip().startswith('Conflict'):
                conflicts.append(line.strip())
        
        return conflicts
    except Exception:
        return []


# Quick test
if __name__ == "__main__":
    print("Testing Gemini integration...")
    
    try:
        configure_gemini()
        model = get_model()
        response = model.generate_content("Say 'Gemini is working!' if you can read this.")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure GOOGLE_API_KEY is set in your environment")
