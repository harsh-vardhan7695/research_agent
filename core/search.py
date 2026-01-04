"""
Search module using Tavily API

Why Tavily:
- Optimized for AI/LLM applications
- Returns high-quality, relevant results
- Provides clean, structured data with content snippets
- Built specifically for research and retrieval tasks

The search strategy here is intentionally simple:
1. Take the user query
2. Optionally expand it into multiple search queries for better coverage
3. Return normalized results with relevance scoring
"""

import os
import requests
import hashlib
from datetime import datetime

# Simple in-memory cache to avoid hitting rate limits during development
# Not persisted - clears on restart, which is fine for this use case
_search_cache = {}

TAVILY_API_URL = "https://api.tavily.com/search"


def _get_cache_key(query: str) -> str:
    """Generate cache key from query"""
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


def search_web(
    query: str,
    num_results: int = 10,
    use_cache: bool = True
) -> dict:
    """
    Execute a web search using Tavily API

    Args:
        query: Search query string
        num_results: Number of results to fetch (Tavily returns up to 10 by default)
        use_cache: Whether to use cached results (helps during dev/testing)

    Returns:
        dict with keys:
            - results: list of search results
            - query: original query
            - timestamp: when search was executed
            - cached: whether result came from cache
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY not found in environment. "
            "Get one free at https://tavily.com"
        )
    
    cache_key = _get_cache_key(query)
    
    # Check cache first
    if use_cache and cache_key in _search_cache:
        cached = _search_cache[cache_key]
        cached["cached"] = True
        return cached
    
    # Make the API request
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": min(num_results, 10),  # Tavily caps at 10
        "search_depth": "basic",  # Use "advanced" for deeper search
        "include_answer": False,
        "include_raw_content": False
    }
    
    try:
        response = requests.post(
            TAVILY_API_URL,
            headers=headers,
            json=payload,
            timeout=15  # Don't hang forever
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        return {
            "results": [],
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "cached": False,
            "error": "Search timed out - try again"
        }
    except requests.exceptions.RequestException as e:
        return {
            "results": [],
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "cached": False,
            "error": f"Search failed: {str(e)}"
        }

    # Normalize the results into our standard format
    # Tavily returns 'results' array
    raw_results = data.get("results", [])
    
    normalized = []
    for idx, item in enumerate(raw_results):
        normalized.append({
            "title": item.get("title", "Untitled"),
            "url": item.get("url", ""),
            "snippet": item.get("content", ""),
            "position": idx + 1,  # 1-indexed position in results
            # Use Tavily's score if available, otherwise position-based
            "relevance_score": item.get("score", round(1.0 - (idx * 0.08), 2)),
            "domain": _extract_domain(item.get("url", ""))
        })
    
    result = {
        "results": normalized,
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "cached": False
    }
    
    # Cache it
    if use_cache:
        _search_cache[cache_key] = result
    
    return result


def _extract_domain(url: str) -> str:
    """Extract domain from URL for citation purposes"""
    if not url:
        return "unknown"
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc
        # Remove www. prefix for cleaner display
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return "unknown"


def expand_query(original_query: str, num_variations: int = 2) -> list:
    """
    Generate query variations for broader coverage
    
    Why do this?
    - Single query might miss relevant results
    - Different phrasings surface different sources
    - Helps with multi-faceted questions
    
    This is a simple rule-based expansion. In a production system,
    you'd use the LLM to generate smarter variations.
    """
    variations = [original_query]  # Always include original
    
    # Add "what is" variant for definitional queries
    if not original_query.lower().startswith(("what", "who", "how", "why", "when")):
        variations.append(f"what is {original_query}")
    
    # Add "latest" variant for potentially time-sensitive queries
    time_indicators = ["recent", "latest", "new", "current", "2024", "2025"]
    if not any(ind in original_query.lower() for ind in time_indicators):
        variations.append(f"{original_query} latest")
    
    return variations[:num_variations + 1]  # Limit to requested count + original


def search_multiple(
    queries: list,
    results_per_query: int = 5
) -> dict:
    """
    Execute multiple searches and merge results
    
    Deduplicates by URL and combines relevance scores
    """
    all_results = {}
    all_queries = []
    
    for query in queries:
        search_result = search_web(query, num_results=results_per_query)
        all_queries.append({
            "query": query,
            "num_results": len(search_result.get("results", [])),
            "cached": search_result.get("cached", False)
        })
        
        for result in search_result.get("results", []):
            url = result["url"]
            if url in all_results:
                # Boost score if same URL appears in multiple searches
                all_results[url]["relevance_score"] = min(
                    1.0,
                    all_results[url]["relevance_score"] + 0.1
                )
                all_results[url]["found_in_queries"] += 1
            else:
                result["found_in_queries"] = 1
                all_results[url] = result
    
    # Sort by relevance score descending
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x["relevance_score"],
        reverse=True
    )
    
    return {
        "results": sorted_results,
        "queries_executed": all_queries,
        "timestamp": datetime.now().isoformat(),
        "total_unique_results": len(sorted_results)
    }


# Quick test when running directly
if __name__ == "__main__":
    # Test requires TAVILY_API_KEY to be set
    test_query = "what is retrieval augmented generation"
    print(f"Testing search for: {test_query}\n")

    result = search_web(test_query, num_results=5)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Found {len(result['results'])} results:\n")
        for r in result["results"]:
            print(f"  [{r['position']}] {r['title']}")
            print(f"      {r['domain']} | score: {r['relevance_score']}")
            print(f"      {r['snippet'][:100]}...")
            print()
