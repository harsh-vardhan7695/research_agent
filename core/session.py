"""
Session Management Module

Handles:
1. Session creation and retrieval
2. Conversation history storage
3. Turn history (detailed per-query tracking)
4. Persistence to JSON (simple, no external DB needed)

Design decisions:
- JSON file storage (SQLite would be overkill for this scope)
- One file per session (easy to inspect, debug)
- Automatic cleanup of old sessions (prevent unbounded growth)
"""

import os
import json
import uuid
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

# Default storage location
DATA_DIR = Path(__file__).parent.parent / "data" / "sessions"


def ensure_data_dir():
    """Create data directory if it doesn't exist"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())[:8]  # Short enough to be readable


def get_session_path(session_id: str) -> Path:
    """Get the file path for a session"""
    ensure_data_dir()
    return DATA_DIR / f"session_{session_id}.json"


def create_session(session_id: str = None) -> dict:
    """
    Create a new session
    
    Returns:
        Session dict with id, created_at, and empty histories
    """
    if session_id is None:
        session_id = generate_session_id()
    
    session = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "conversation_history": [],  # All messages
        "turn_history": [],  # Detailed per-query tracking
        "metadata": {
            "total_queries": 0,
            "total_searches": 0,
            "sources_consulted": []
        }
    }
    
    save_session(session)
    return session


def load_session(session_id: str) -> Optional[dict]:
    """
    Load an existing session
    
    Returns:
        Session dict if found, None otherwise
    """
    path = get_session_path(session_id)
    
    if not path.exists():
        return None
    
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_session(session: dict):
    """Save session to disk"""
    session["updated_at"] = datetime.now().isoformat()
    path = get_session_path(session["session_id"])
    
    with open(path, 'w') as f:
        json.dump(session, f, indent=2, default=str)


def get_or_create_session(session_id: str = None) -> dict:
    """
    Get existing session or create new one
    
    This is the main entry point for session management
    """
    if session_id:
        session = load_session(session_id)
        if session:
            return session
    
    return create_session(session_id)


def add_message(
    session: dict,
    role: str,
    content: str,
    save: bool = True
) -> dict:
    """
    Add a message to conversation history
    
    Args:
        session: Session dict
        role: 'user' or 'assistant'
        content: Message content
        save: Whether to save to disk immediately
    
    Returns:
        Updated session
    """
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    
    session["conversation_history"].append(message)
    
    if save:
        save_session(session)
    
    return session


def add_turn(
    session: dict,
    turn_data: dict,
    save: bool = True
) -> dict:
    """
    Add a turn record with full research details
    
    Args:
        session: Session dict
        turn_data: Dict with query, searches, urls, snippets, answer, etc.
        save: Whether to save immediately
    
    Returns:
        Updated session
    """
    turn = {
        "turn_number": len(session["turn_history"]) + 1,
        "timestamp": datetime.now().isoformat(),
        **turn_data
    }
    
    session["turn_history"].append(turn)
    
    # Update metadata
    session["metadata"]["total_queries"] += 1
    if "search_queries" in turn_data:
        session["metadata"]["total_searches"] += len(turn_data["search_queries"])
    if "urls_opened" in turn_data:
        for url in turn_data["urls_opened"]:
            if url not in session["metadata"]["sources_consulted"]:
                session["metadata"]["sources_consulted"].append(url)
    
    if save:
        save_session(session)
    
    return session


def get_conversation_history(session: dict, max_messages: int = None) -> list:
    """
    Get conversation history, optionally limited
    
    Args:
        session: Session dict
        max_messages: Max number of messages to return (from end)
    
    Returns:
        List of messages
    """
    history = session.get("conversation_history", [])
    
    if max_messages and len(history) > max_messages:
        return history[-max_messages:]
    
    return history


def get_recent_turns(session: dict, n: int = 3) -> list:
    """
    Get the N most recent turns
    
    Useful for providing context to the agent about recent research
    """
    turns = session.get("turn_history", [])
    return turns[-n:] if len(turns) > n else turns


def needs_summarization(session: dict, threshold: int = 10) -> bool:
    """
    Check if conversation history needs summarization
    
    Returns True if history is getting long and should be summarized
    """
    return len(session.get("conversation_history", [])) > threshold


def clear_old_sessions(max_age_days: int = 7):
    """
    Remove sessions older than specified days
    
    Call this periodically to prevent unbounded growth
    """
    ensure_data_dir()
    cutoff = datetime.now() - timedelta(days=max_age_days)
    
    for path in DATA_DIR.glob("session_*.json"):
        try:
            with open(path, 'r') as f:
                session = json.load(f)
            
            updated = datetime.fromisoformat(session.get("updated_at", "2000-01-01"))
            if updated < cutoff:
                path.unlink()
        except (json.JSONDecodeError, IOError, KeyError):
            # Corrupted file, remove it
            path.unlink()


def list_sessions() -> list:
    """
    List all available sessions
    
    Returns:
        List of session summaries
    """
    ensure_data_dir()
    sessions = []
    
    for path in DATA_DIR.glob("session_*.json"):
        try:
            with open(path, 'r') as f:
                session = json.load(f)
            
            sessions.append({
                "session_id": session["session_id"],
                "created_at": session.get("created_at"),
                "updated_at": session.get("updated_at"),
                "num_messages": len(session.get("conversation_history", [])),
                "num_turns": len(session.get("turn_history", []))
            })
        except (json.JSONDecodeError, IOError):
            continue
    
    # Sort by updated_at descending
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return sessions


def export_session(session_id: str) -> Optional[str]:
    """
    Export session as formatted text
    
    Useful for debugging or sharing conversation
    """
    session = load_session(session_id)
    if not session:
        return None
    
    lines = [
        f"Session: {session_id}",
        f"Created: {session.get('created_at')}",
        f"Messages: {len(session.get('conversation_history', []))}",
        f"Turns: {len(session.get('turn_history', []))}",
        "",
        "=== Conversation ===",
        ""
    ]
    
    for msg in session.get("conversation_history", []):
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"[{msg.get('timestamp', '')}] {role}:")
        lines.append(msg["content"])
        lines.append("")
    
    return "\n".join(lines)


# Test when running directly
if __name__ == "__main__":
    print("Testing session management...")
    
    # Create a session
    session = create_session()
    print(f"Created session: {session['session_id']}")
    
    # Add some messages
    add_message(session, "user", "What is RAG?")
    add_message(session, "assistant", "RAG stands for Retrieval Augmented Generation...")
    
    # Add a turn record
    add_turn(session, {
        "query": "What is RAG?",
        "search_queries": ["what is RAG", "retrieval augmented generation explained"],
        "urls_opened": ["https://example.com/rag"],
        "context_snippets": ["RAG is a technique..."],
        "final_answer": "RAG stands for..."
    })
    
    # Reload and verify
    loaded = load_session(session["session_id"])
    print(f"Loaded session with {len(loaded['conversation_history'])} messages")
    print(f"Turn history: {len(loaded['turn_history'])} turns")
    
    # List sessions
    print(f"\nAll sessions: {list_sessions()}")
