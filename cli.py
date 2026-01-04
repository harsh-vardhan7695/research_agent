#!/usr/bin/env python3
"""
Command Line Interface for Deep Research Agent

A simple CLI for quick research queries without the Streamlit UI.

Usage:
    python cli.py "What is retrieval augmented generation?"
    python cli.py --session abc123 "Follow up question"
    python cli.py --list-sessions
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from core.agent import ResearchAgent
from core.session import list_sessions, export_session


def check_api_keys():
    """Check for required API keys"""
    missing = []
    if not os.getenv("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    if not os.getenv("GOOGLE_API_KEY"):
        missing.append("GOOGLE_API_KEY")

    if missing:
        print("ERROR: Missing required API keys:")
        for key in missing:
            print(f"  - {key}")
        print("\nSet them with:")
        print("  export TAVILY_API_KEY='your-key'")
        print("  export GOOGLE_API_KEY='your-key'")
        sys.exit(1)


def list_all_sessions():
    """Display all saved sessions"""
    sessions = list_sessions()
    
    if not sessions:
        print("No sessions found.")
        return
    
    print("\nSaved Sessions:")
    print("-" * 60)
    for s in sessions:
        print(f"  ID: {s['session_id']}")
        print(f"      Created: {s.get('created_at', 'unknown')[:19]}")
        print(f"      Turns: {s.get('num_turns', 0)}")
        print()


def run_research(query: str, session_id: str = None, max_sources: int = 5):
    """Run research and stream output"""
    agent = ResearchAgent(session_id=session_id)
    
    print(f"\nSession: {agent.session_id}")
    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60 + "\n")
    
    # Stream the research process
    for chunk in agent.research(query, max_sources=max_sources):
        print(chunk, end="", flush=True)
    
    print("\n" + "=" * 60)
    
    # Show session info
    summary = agent.get_session_summary()
    print(f"\nSession Stats:")
    print(f"  Total turns: {summary['total_turns']}")
    print(f"  Total searches: {summary['metadata'].get('total_searches', 0)}")
    print(f"\nTo continue this session:")
    print(f"  python cli.py --session {agent.session_id} \"your follow-up question\"")


def main():
    parser = argparse.ArgumentParser(
        description="Deep Research Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What is machine learning?"
  %(prog)s --session abc123 "Tell me more about neural networks"
  %(prog)s --list-sessions
  %(prog)s --export abc123
        """
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Research query"
    )
    parser.add_argument(
        "--session", "-s",
        help="Session ID to continue"
    )
    parser.add_argument(
        "--max-sources", "-m",
        type=int,
        default=5,
        help="Maximum sources to fetch (default: 5)"
    )
    parser.add_argument(
        "--list-sessions", "-l",
        action="store_true",
        help="List all saved sessions"
    )
    parser.add_argument(
        "--export", "-e",
        metavar="SESSION_ID",
        help="Export a session to text"
    )
    
    args = parser.parse_args()
    
    # Handle list sessions
    if args.list_sessions:
        list_all_sessions()
        return
    
    # Handle export
    if args.export:
        text = export_session(args.export)
        if text:
            print(text)
        else:
            print(f"Session {args.export} not found")
        return
    
    # Require query for research
    if not args.query:
        parser.print_help()
        print("\nERROR: Query is required for research")
        sys.exit(1)
    
    # Check API keys
    check_api_keys()
    
    # Run research
    run_research(
        query=args.query,
        session_id=args.session,
        max_sources=args.max_sources
    )


if __name__ == "__main__":
    main()
