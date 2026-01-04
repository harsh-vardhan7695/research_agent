"""
Deep Research Agent - Main Orchestration

This is where everything comes together. The agent:
1. Receives a user query
2. Plans the research approach
3. Executes searches
4. Fetches and processes pages
5. Builds context
6. Generates a cited answer
7. Records everything for history

NO frameworks - just clean Python orchestration.

The flow is intentionally linear and explicit. This makes it:
- Easy to debug (you can trace exactly what happened)
- Easy to test (each step is isolated)
- Easy to explain (no magic happening behind abstractions)
"""

from typing import Generator, Optional, Callable
from datetime import datetime

from .search import search_web, expand_query, search_multiple
from .fetcher import fetch_page, fetch_multiple, extract_relevant_chunks
from .context import build_context, format_context_for_llm, summarize_conversation
from .llm import (
    generate_research_plan,
    extract_search_queries,
    generate_answer_stream,
    generate_answer,
    summarize_for_context,
    check_for_conflicts
)
from .session import (
    get_or_create_session,
    add_message,
    add_turn,
    get_conversation_history,
    needs_summarization,
    save_session
)


class ResearchAgent:
    """
    The main research agent class
    
    Orchestrates the entire research flow from query to answer.
    Maintains session state and streams progress updates.
    """
    
    def __init__(
        self,
        session_id: str = None,
        progress_callback: Callable[[str, str], None] = None
    ):
        """
        Initialize the agent
        
        Args:
            session_id: Existing session ID to resume, or None for new session
            progress_callback: Function to call with (step_name, message) updates
        """
        self.session = get_or_create_session(session_id)
        self.progress_callback = progress_callback or self._default_progress
        
        # Track current turn data
        self._current_turn = {}
    
    def _default_progress(self, step: str, message: str):
        """Default progress handler - just prints"""
        print(f"[{step}] {message}")
    
    def _emit_progress(self, step: str, message: str):
        """Emit a progress update"""
        if self.progress_callback:
            self.progress_callback(step, message)
    
    @property
    def session_id(self) -> str:
        return self.session["session_id"]
    
    def research(
        self,
        query: str,
        max_sources: int = 5,
        stream: bool = True
    ) -> Generator[str, None, str]:
        """
        Execute research for a query
        
        This is the main entry point. It:
        1. Plans the research
        2. Searches the web
        3. Fetches sources
        4. Builds context
        5. Generates answer
        
        Args:
            query: User's question
            max_sources: Maximum number of sources to fetch
            stream: Whether to stream the answer
        
        Yields:
            Progress updates and answer chunks
        
        Returns:
            The complete answer
        """
        # Initialize turn tracking
        self._current_turn = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "search_queries": [],
            "urls_opened": [],
            "context_snippets": [],
            "plan": "",
            "conflicts_detected": [],
            "final_answer": ""
        }
        
        # Add user message to history
        add_message(self.session, "user", query, save=False)
        
        # Check if we need to summarize conversation history
        conversation_context = ""
        if needs_summarization(self.session):
            self._emit_progress("summarizing", "Summarizing conversation history...")
            history = get_conversation_history(self.session)
            conversation_context = summarize_for_context(history)
        
        # === STEP 1: PLAN ===
        self._emit_progress("planning", "Creating research plan...")
        yield "🔍 Planning research approach...\n\n"
        
        plan = generate_research_plan(query, conversation_context)
        self._current_turn["plan"] = plan
        
        # Extract search queries from plan
        search_queries = extract_search_queries(plan)
        if not search_queries:
            # Fallback: use query variations
            search_queries = expand_query(query, num_variations=2)
        
        self._current_turn["search_queries"] = search_queries
        
        yield f"📋 Research plan:\n{plan}\n\n"
        
        # === STEP 2: SEARCH ===
        self._emit_progress("searching", f"Searching with {len(search_queries)} queries...")
        yield f"🌐 Searching the web with {len(search_queries)} queries...\n\n"
        
        search_results = search_multiple(search_queries, results_per_query=5)
        
        if not search_results["results"]:
            yield "❌ No search results found. Please try rephrasing your question.\n"
            self._current_turn["final_answer"] = "No search results found."
            self._finalize_turn()
            return "No search results found."
        
        yield f"📊 Found {search_results['total_unique_results']} unique results\n\n"
        
        # === STEP 3: FETCH SOURCES ===
        self._emit_progress("fetching", f"Fetching top {max_sources} sources...")
        yield f"📥 Fetching content from top sources...\n\n"
        
        # Get URLs from search results
        urls_to_fetch = [r["url"] for r in search_results["results"][:max_sources * 2]]
        
        fetched_pages = fetch_multiple(urls_to_fetch, max_pages=max_sources)
        self._current_turn["urls_opened"] = [p["url"] for p in fetched_pages]
        
        if not fetched_pages:
            yield "❌ Couldn't fetch any pages. Sources may be blocked or unavailable.\n"
            self._current_turn["final_answer"] = "Could not fetch source content."
            self._finalize_turn()
            return "Could not fetch source content."
        
        yield f"✅ Successfully fetched {len(fetched_pages)} sources\n\n"
        
        # === STEP 4: BUILD CONTEXT ===
        self._emit_progress("context", "Selecting relevant content...")
        yield "📝 Analyzing and selecting relevant content...\n\n"
        
        context = build_context(
            fetched_pages,
            query,
            conversation_summary=conversation_context
        )
        
        self._current_turn["context_snippets"] = [
            {"text": s["text"][:200], "url": s["url"], "title": s["title"]}
            for s in context["snippets"]
        ]
        
        yield f"📚 Selected {context['chunks_selected']} snippets from {context['sources_used']} sources\n\n"
        
        # Check for conflicts
        formatted_context = format_context_for_llm(context, conversation_context)
        conflicts = check_for_conflicts(formatted_context)
        self._current_turn["conflicts_detected"] = conflicts
        
        if conflicts:
            yield "⚠️ Note: Found potentially conflicting information in sources\n\n"
        
        # === STEP 5: GENERATE ANSWER ===
        self._emit_progress("generating", "Generating answer with citations...")
        yield "💭 Generating answer with citations...\n\n"
        yield "---\n\n"
        
        # Stream or generate answer
        full_answer = ""
        
        history = get_conversation_history(self.session, max_messages=4)
        
        if stream:
            for chunk in generate_answer_stream(query, formatted_context, history):
                full_answer += chunk
                yield chunk
        else:
            full_answer = generate_answer(query, formatted_context, history)
            yield full_answer
        
        # Record the answer
        self._current_turn["final_answer"] = full_answer
        add_message(self.session, "assistant", full_answer, save=False)
        
        # Finalize turn
        self._finalize_turn()
        
        return full_answer
    
    def _finalize_turn(self):
        """Save the current turn to history"""
        add_turn(self.session, self._current_turn)
        self._current_turn = {}
    
    def research_sync(
        self,
        query: str,
        max_sources: int = 5
    ) -> dict:
        """
        Synchronous version of research - collects all output
        
        Returns:
            Dict with 'answer', 'sources', 'turn_data'
        """
        output_parts = []
        final_answer = ""
        
        for chunk in self.research(query, max_sources, stream=False):
            output_parts.append(chunk)
            if "---" in chunk:
                # Everything after --- is the answer
                idx = "".join(output_parts).find("---")
                final_answer = "".join(output_parts)[idx + 4:]
        
        # Get the last turn data
        turn_data = self.session["turn_history"][-1] if self.session["turn_history"] else {}
        
        return {
            "answer": final_answer.strip(),
            "full_output": "".join(output_parts),
            "sources": turn_data.get("urls_opened", []),
            "turn_data": turn_data
        }
    
    def get_history(self) -> list:
        """Get conversation history"""
        return get_conversation_history(self.session)
    
    def get_turn_history(self) -> list:
        """Get detailed turn history"""
        return self.session.get("turn_history", [])
    
    def get_session_summary(self) -> dict:
        """Get session summary/stats"""
        return {
            "session_id": self.session_id,
            "created_at": self.session.get("created_at"),
            "total_messages": len(self.session.get("conversation_history", [])),
            "total_turns": len(self.session.get("turn_history", [])),
            "metadata": self.session.get("metadata", {})
        }


def quick_research(query: str, session_id: str = None) -> str:
    """
    Quick one-shot research function
    
    For simple use cases where you just want an answer
    """
    agent = ResearchAgent(session_id)
    result = agent.research_sync(query)
    return result["answer"]


# Test when running directly
if __name__ == "__main__":
    import os
    
    # Check for required API keys
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY not set")
        exit(1)
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not set")
        exit(1)
    
    print("Testing Research Agent...")
    print("=" * 50)
    
    agent = ResearchAgent()
    print(f"Session ID: {agent.session_id}")
    
    query = "What are the latest developments in AI agents?"
    print(f"\nQuery: {query}\n")
    
    for chunk in agent.research(query, max_sources=3):
        print(chunk, end="", flush=True)
    
    print("\n" + "=" * 50)
    print("Session Summary:", agent.get_session_summary())
