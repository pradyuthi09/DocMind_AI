import streamlit as st
import os
import shutil

def initialize_sessions():
    """Initialize the multi-session state dictionary and current session tracking."""
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None

def create_session(session_id, file_name, page_count=0, chunk_count=0):
    """Create a new chat session with its own unique metadata and database directory."""
    db_dir = f"data/chroma_db_{session_id}"
    st.session_state.sessions[session_id] = {
        "file_name": file_name,
        "messages": [],
        "page_count": page_count,
        "chunk_count": chunk_count,
        "db_dir": db_dir
    }
    st.session_state.current_session_id = session_id

def get_active_session():
    """Retrieve the current active session data dictionary."""
    curr_id = st.session_state.current_session_id
    if curr_id and curr_id in st.session_state.sessions:
        return st.session_state.sessions[curr_id]
    return None

def add_message_to_active(role, content):
    """Append a message to the active session's history."""
    session = get_active_session()
    if session:
        session["messages"].append({
            "role": role,
            "content": content
        })

def delete_session(session_id):
    """Delete a session's state and its persisted vector database directory from disk."""
    if session_id in st.session_state.sessions:
        session = st.session_state.sessions[session_id]
        # Clean up database files
        db_dir = session.get("db_dir")
        if db_dir and os.path.exists(db_dir):
            try:
                shutil.rmtree(db_dir)
            except Exception as e:
                print(f"Error deleting vector database folder {db_dir}: {e}")
        
        # Remove from state
        del st.session_state.sessions[session_id]
        
        # Reset current session if active one was deleted
        if st.session_state.current_session_id == session_id:
            st.session_state.current_session_id = (
                list(st.session_state.sessions.keys())[0] if st.session_state.sessions else None
            )