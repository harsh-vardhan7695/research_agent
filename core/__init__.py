"""
Deep Research Agent - Core Module

This module contains all the core components:
- search: Web search via Serper
- fetcher: Page content extraction
- context: Context building for LLM
- llm: Gemini integration
- session: Session management
- agent: Main orchestration
"""

from .agent import ResearchAgent, quick_research
from .search import search_web, search_multiple, expand_query
from .fetcher import fetch_page, fetch_multiple
from .context import build_context, format_context_for_llm
from .llm import generate_answer, generate_research_plan
from .session import (
    get_or_create_session,
    load_session,
    list_sessions,
    export_session
)

__all__ = [
    "ResearchAgent",
    "quick_research",
    "search_web",
    "search_multiple",
    "expand_query",
    "fetch_page",
    "fetch_multiple",
    "build_context",
    "format_context_for_llm",
    "generate_answer",
    "generate_research_plan",
    "get_or_create_session",
    "load_session",
    "list_sessions",
    "export_session"
]
