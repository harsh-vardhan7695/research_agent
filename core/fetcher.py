"""
Page Fetcher Module

Responsible for:
1. Fetching full HTML from URLs
2. Extracting readable text content
3. Handling various edge cases (timeouts, blocked sites, etc.)

Design decisions:
- Use requests with reasonable timeouts (sites shouldn't take >10s)
- BeautifulSoup for HTML parsing (simple, battle-tested)
- Strip scripts, styles, navs - we want article content only
- Store metadata for citation tracking
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse
import re
import hashlib

# Simple cache - avoids re-fetching same page in a session
_page_cache = {}

# Some sites block automated requests or have paywalls
# We'll gracefully handle these rather than crash
BLOCKED_INDICATORS = [
    "access denied",
    "please enable javascript",
    "captcha",
    "robot",
    "blocked",
    "403 forbidden",
    "subscribe to read",
    "create an account"
]


def fetch_page(
    url: str,
    timeout: int = 10,
    use_cache: bool = True
) -> dict:
    """
    Fetch a web page and extract readable content
    
    Args:
        url: Full URL to fetch
        timeout: Request timeout in seconds
        use_cache: Whether to return cached result if available
    
    Returns:
        dict with:
            - url: the URL fetched
            - title: page title
            - content: extracted text content
            - domain: source domain
            - retrieved_at: timestamp
            - success: bool indicating if fetch worked
            - error: error message if failed
            - word_count: approximate word count of content
    """
    cache_key = hashlib.md5(url.encode()).hexdigest()
    
    if use_cache and cache_key in _page_cache:
        cached = _page_cache[cache_key].copy()
        cached["from_cache"] = True
        return cached
    
    result = {
        "url": url,
        "domain": _extract_domain(url),
        "retrieved_at": datetime.now().isoformat(),
        "from_cache": False
    }
    
    # Set headers to look like a browser
    # Some sites block requests without proper headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True
        )
        response.raise_for_status()
        
        # Check content type - we only handle HTML
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            result["success"] = False
            result["error"] = f"Not HTML content: {content_type}"
            result["title"] = ""
            result["content"] = ""
            result["word_count"] = 0
            return result
        
        html = response.text
        
    except requests.exceptions.Timeout:
        result["success"] = False
        result["error"] = "Request timed out"
        result["title"] = ""
        result["content"] = ""
        result["word_count"] = 0
        return result
    except requests.exceptions.RequestException as e:
        result["success"] = False
        result["error"] = f"Failed to fetch: {str(e)}"
        result["title"] = ""
        result["content"] = ""
        result["word_count"] = 0
        return result
    
    # Parse HTML and extract content
    try:
        soup = BeautifulSoup(html, "html.parser")
        
        # Get title
        title_tag = soup.find("title")
        result["title"] = title_tag.get_text().strip() if title_tag else "Untitled"
        
        # Remove elements that don't contain article content
        for tag in soup.find_all(["script", "style", "nav", "header", "footer", 
                                   "aside", "advertisement", "ads", "comments"]):
            tag.decompose()
        
        # Try to find main content area
        # Different sites use different containers
        content = None
        
        # Common article containers in order of preference
        content_selectors = [
            soup.find("article"),
            soup.find("main"),
            soup.find(class_=re.compile(r"article|content|post|entry", re.I)),
            soup.find(id=re.compile(r"article|content|post|entry", re.I)),
            soup.find("body")  # Fallback to body
        ]
        
        for selector in content_selectors:
            if selector:
                content = selector
                break
        
        if content:
            # Get text, preserving some structure
            text = content.get_text(separator="\n", strip=True)
            # Clean up excessive whitespace
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r' {2,}', ' ', text)
        else:
            text = ""
        
        # Check for blocked/paywall indicators
        text_lower = text.lower()[:500]  # Check first 500 chars
        for indicator in BLOCKED_INDICATORS:
            if indicator in text_lower:
                result["success"] = False
                result["error"] = f"Page appears blocked or requires authentication"
                result["title"] = result["title"]
                result["content"] = ""
                result["word_count"] = 0
                return result
        
        result["success"] = True
        result["content"] = text
        result["word_count"] = len(text.split())
        
        # Cache successful fetches
        if use_cache:
            _page_cache[cache_key] = result
        
        return result
        
    except Exception as e:
        result["success"] = False
        result["error"] = f"Failed to parse HTML: {str(e)}"
        result["title"] = ""
        result["content"] = ""
        result["word_count"] = 0
        return result


def _extract_domain(url: str) -> str:
    """Extract clean domain from URL"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return "unknown"


def fetch_multiple(
    urls: list,
    max_pages: int = 5,
    timeout_per_page: int = 8
) -> list:
    """
    Fetch multiple pages, with early stopping if we have enough content
    
    Args:
        urls: List of URLs to fetch
        max_pages: Maximum number of pages to actually fetch
        timeout_per_page: Timeout for each request
    
    Returns:
        List of fetch results (only successful ones)
    """
    results = []
    attempted = 0
    
    for url in urls:
        if len(results) >= max_pages:
            break
            
        attempted += 1
        result = fetch_page(url, timeout=timeout_per_page)
        
        # Only keep successful fetches with actual content
        if result["success"] and result["word_count"] > 100:
            results.append(result)
    
    return results


def extract_relevant_chunks(
    content: str,
    query: str,
    chunk_size: int = 500,
    max_chunks: int = 3
) -> list:
    """
    Extract the most relevant chunks from page content
    
    Simple approach: split into chunks, score by keyword overlap with query
    
    A more sophisticated system would use embeddings, but keyword matching
    works surprisingly well for most queries and requires no additional API calls.
    """
    if not content:
        return []
    
    # Split into paragraphs first, then combine into chunks
    paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Score each chunk by query term overlap
    query_terms = set(query.lower().split())
    scored_chunks = []
    
    for chunk in chunks:
        chunk_terms = set(chunk.lower().split())
        overlap = len(query_terms & chunk_terms)
        # Normalize by query length
        score = overlap / len(query_terms) if query_terms else 0
        scored_chunks.append((chunk, score))
    
    # Sort by score descending
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Return top chunks
    return [chunk for chunk, score in scored_chunks[:max_chunks]]


# Test when running directly
if __name__ == "__main__":
    test_url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    print(f"Testing fetch for: {test_url}\n")
    
    result = fetch_page(test_url)
    
    if result["success"]:
        print(f"Title: {result['title']}")
        print(f"Domain: {result['domain']}")
        print(f"Word count: {result['word_count']}")
        print(f"\nFirst 500 chars of content:")
        print(result['content'][:500])
    else:
        print(f"Failed: {result['error']}")
