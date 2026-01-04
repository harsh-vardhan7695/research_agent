"""
Deep Research Agent - Streamlit Interface

A clean, functional interface for the research agent.
Streams progress and results in real-time.

Design philosophy:
- Simple and focused (not over-designed)
- Clear feedback at each step
- Session persistence across browser refreshes
"""

import streamlit as st
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import ResearchAgent
from core.session import list_sessions, load_session, export_session


# Page config
st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for cleaner look
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .research-step {
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
        background: #f0f2f6;
    }
    .source-card {
        padding: 10px;
        margin: 5px 0;
        border-left: 3px solid #4CAF50;
        background: #f9f9f9;
    }
    .chat-message {
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
    }
    .user-message {
        background: #e3f2fd;
    }
    .assistant-message {
        background: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)


def check_api_keys():
    """Check if required API keys are set"""
    missing = []
    if not os.getenv("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    if not os.getenv("GOOGLE_API_KEY"):
        missing.append("GOOGLE_API_KEY")
    return missing


def init_session_state():
    """Initialize Streamlit session state"""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None


def create_new_session():
    """Create a new research session"""
    st.session_state.agent = ResearchAgent()
    st.session_state.current_session_id = st.session_state.agent.session_id
    st.session_state.messages = []


def load_existing_session(session_id: str):
    """Load an existing session"""
    st.session_state.agent = ResearchAgent(session_id=session_id)
    st.session_state.current_session_id = session_id
    # Load messages from session history
    history = st.session_state.agent.get_history()
    st.session_state.messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
    ]


def display_chat_history():
    """Display the chat history"""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])


def main():
    st.title("🔬 Deep Research Agent")
    st.caption("AI-powered web research with source citations")
    
    # Initialize
    init_session_state()
    
    # Check API keys
    missing_keys = check_api_keys()
    if missing_keys:
        st.error(f"Missing API keys: {', '.join(missing_keys)}")
        st.info("""
        **Setup Instructions:**
        1. Get a Tavily API key from https://tavily.com (free tier available)
        2. Get a Google API key from https://aistudio.google.com/apikey
        3. Set environment variables before running:
        ```bash
        export TAVILY_API_KEY="your-tavily-key"
        export GOOGLE_API_KEY="your-google-key"
        streamlit run app.py
        ```
        """)
        return
    
    # Sidebar for session management
    with st.sidebar:
        st.header("📁 Sessions")
        
        # New session button
        if st.button("➕ New Session", use_container_width=True):
            create_new_session()
            st.rerun()
        
        st.divider()
        
        # List existing sessions
        sessions = list_sessions()
        if sessions:
            st.subheader("Recent Sessions")
            for s in sessions[:10]:  # Show last 10
                session_label = f"{s['session_id']} ({s['num_turns']} turns)"
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(
                        session_label,
                        key=f"load_{s['session_id']}",
                        use_container_width=True
                    ):
                        load_existing_session(s["session_id"])
                        st.rerun()
        
        st.divider()
        
        # Current session info
        if st.session_state.agent:
            st.subheader("Current Session")
            summary = st.session_state.agent.get_session_summary()
            st.text(f"ID: {summary['session_id']}")
            st.text(f"Turns: {summary['total_turns']}")
            
            # Export option
            if st.button("📤 Export Session"):
                export_text = export_session(summary["session_id"])
                if export_text:
                    st.download_button(
                        "Download",
                        export_text,
                        file_name=f"session_{summary['session_id']}.txt",
                        mime="text/plain"
                    )
    
    # Main chat area
    if st.session_state.agent is None:
        st.info("👆 Click 'New Session' in the sidebar to start researching")
        
        # Quick start
        st.markdown("---")
        st.subheader("Quick Start")
        st.markdown("""
        This agent conducts deep web research to answer your questions with citations.
        
        **How it works:**
        1. 📋 Creates a research plan
        2. 🔍 Searches the web with multiple queries
        3. 📥 Fetches and analyzes source content
        4. 💭 Generates a cited answer
        
        **Example questions:**
        - "What are the latest developments in AI agents?"
        - "Compare React and Vue.js for building web apps"
        - "What is retrieval augmented generation and how does it work?"
        """)
        return
    
    # Display current session ID
    st.caption(f"Session: `{st.session_state.current_session_id}`")
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Ask a research question..."):
        # Add user message to display
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Research and display response
        with st.chat_message("assistant"):
            # Create containers for streaming output
            progress_container = st.empty()
            answer_container = st.empty()
            
            progress_text = ""
            answer_text = ""
            past_divider = False
            
            # Stream the research process
            for chunk in st.session_state.agent.research(prompt, max_sources=5):
                if "---" in chunk:
                    past_divider = True
                    progress_container.markdown(progress_text)
                    continue
                
                if past_divider:
                    answer_text += chunk
                    answer_container.markdown(answer_text)
                else:
                    progress_text += chunk
                    progress_container.markdown(progress_text)
            
            # Store the final answer
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text
            })
        
        # Show sources used in expander
        turn_history = st.session_state.agent.get_turn_history()
        if turn_history:
            last_turn = turn_history[-1]
            with st.expander("📚 Sources Used"):
                for url in last_turn.get("urls_opened", []):
                    st.markdown(f"- [{url}]({url})")
            
            # Show search queries used
            with st.expander("🔍 Search Queries"):
                for q in last_turn.get("search_queries", []):
                    st.markdown(f"- {q}")


if __name__ == "__main__":
    main()
