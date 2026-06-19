import streamlit as st
import os
import time
import json
from dotenv import load_dotenv

load_dotenv(override=True)

from src.pdf_loader import load_pdf
from src.splitter import split_docs
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_db
from src.rag_chain import build_chain
from src.chat_manager import (
    initialize_sessions,
    create_session,
    get_active_session,
    add_message_to_active,
    delete_session
)
from langchain_community.vectorstores import Chroma

# Page Config
st.set_page_config(
    page_title="DocuMind AI - Smart PDF RAG",
    page_icon="📄",
    layout="wide"
)

# Load CSS
with open("assets/style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# Initialize Session State
initialize_sessions()

# Ensure we have at least one session if none exists
if not st.session_state.sessions:
    new_id = str(int(time.time()))
    create_session(new_id, "Untitled Document")

# Get active session
session = get_active_session()
if not session:
    # Fallback to the first available session
    keys = list(st.session_state.sessions.keys())
    if keys:
        st.session_state.current_session_id = keys[0]
        session = st.session_state.sessions[keys[0]]

# Sidebar Panel (Left Side inputs and summaries)
with st.sidebar:
    st.markdown("<div class='sidebar-title'>📄 DocuMind AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-subtitle'>Document Intelligence Center</div>", unsafe_allow_html=True)
    st.divider()

    # New Chat Button
    if st.button("➕ New Document Chat", use_container_width=True):
        new_id = str(int(time.time()))
        create_session(new_id, "Untitled Document")
        st.rerun()

    st.divider()

    # Active Document Controls & Stats
    if session:
        st.markdown(f"### 📍 Active: **{session['file_name']}**")
        
        # Check if PDF is uploaded
        has_file = session["file_name"] != "Untitled Document"
        
        if not has_file:
            uploaded_file = st.file_uploader(
                "Upload a PDF to start",
                type=["pdf"],
                key=f"uploader_{st.session_state.current_session_id}"
            )
            
            if uploaded_file:
                # Save file
                os.makedirs("data/uploads", exist_ok=True)
                pdf_path = os.path.join("data/uploads", f"{st.session_state.current_session_id}_{uploaded_file.name}")
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner("Processing document..."):
                    # Load and split
                    docs = load_pdf(pdf_path)
                    chunks = split_docs(docs)
                    
                    # Create vector database
                    embeddings = get_embedding_model()
                    db_dir = session["db_dir"]
                    vectordb = create_vector_db(chunks, embeddings, persist_directory=db_dir)
                    
                    # Update session state
                    session["file_name"] = uploaded_file.name
                    session["page_count"] = len(docs)
                    session["chunk_count"] = len(chunks)
                    session["pdf_path"] = pdf_path
                
                st.success("Document processed!")
                st.rerun()
        else:
            # Display file stats card
            st.markdown(
                f"""
                <div class='stat-card'>
                    <div style='font-size: 14px; opacity: 0.8;'>📄 Active Document</div>
                    <div style='font-weight: bold; font-size: 16px; margin: 5px 0;'>{session['file_name']}</div>
                    <div style='display: flex; justify-content: space-around; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px;'>
                        <div>
                            <div style='font-size: 12px; opacity: 0.7;'>Pages</div>
                            <div style='font-size: 18px; font-weight: bold; color: #60a5fa;'>{session['page_count']}</div>
                        </div>
                        <div>
                            <div style='font-size: 12px; opacity: 0.7;'>Chunks</div>
                            <div style='font-size: 18px; font-weight: bold; color: #34d399;'>{session['chunk_count']}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑 Delete Chat & Doc", use_container_width=True):
                delete_session(st.session_state.current_session_id)
                st.rerun()

    st.divider()

    # Session Selector List
    st.markdown("### 💬 Your Chats")
    
    for s_id, s_data in list(st.session_state.sessions.items()):
        is_active = (s_id == st.session_state.current_session_id)
        btn_label = f"📄 {s_data['file_name'][:20]}" + ("..." if len(s_data['file_name']) > 20 else "")
        
        cols = st.columns([0.8, 0.2])
        with cols[0]:
            if st.button(btn_label, key=f"select_{s_id}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.current_session_id = s_id
                st.rerun()
        with cols[1]:
            if st.button("❌", key=f"del_{s_id}", use_container_width=True):
                delete_session(s_id)
                st.rerun()

    st.divider()
    
    # Utilities
    if session and session["messages"]:
        if st.button("🗑 Reset Chat History", use_container_width=True):
            session["messages"] = []
            st.rerun()
            
        chat_json = json.dumps(session["messages"], indent=4)
        st.download_button(
            "⬇ Download Chat",
            chat_json,
            file_name=f"chat_history_{session['file_name']}.json",
            use_container_width=True
        )

# Main Application Window (Right Side Outputs and Chat)
st.markdown("<div class='main-title'>📄 DocuMind AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Chat Intelligently With Your Documents</div>", unsafe_allow_html=True)
st.divider()

if session:
    has_file = session["file_name"] != "Untitled Document"
    
    if not has_file:
        # Welcoming Landing Page
        st.markdown(
            """
            <div class='welcome-container'>
                <h2>Welcome to DocuMind AI!</h2>
                <p>To start, upload a PDF file in the left control panel. The system will automatically parse and chunk the document, build a local vector database, and set up a context-aware chat environment.</p>
                <div style='margin-top: 30px; font-size: 14px; opacity: 0.7;'>
                    💡 <b>Tip:</b> You can create multiple document chats using the <b>"+ New Document Chat"</b> button.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Render Chat History
        for msg in session["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # User input
        prompt = st.chat_input("Ask anything about your PDF...")
        
        if prompt:
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Save user message to history
            session["messages"].append({"role": "user", "content": prompt})
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Load vector DB from session's persist directory
                        embeddings = get_embedding_model()
                        vectordb = Chroma(
                            persist_directory=session["db_dir"],
                            embedding_function=embeddings
                        )
                        
                        # Build chain and run query
                        chain = build_chain(vectordb)
                        start_time = time.time()
                        response = chain.run(prompt)
                        end_time = time.time()
                        
                        # Render response
                        st.markdown(response)
                        st.caption(f"⏱ Response Time: {round(end_time - start_time, 2)} sec")
                        
                        # Save assistant message to history
                        session["messages"].append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        st.error(f"Error executing query: {e}")
            
            # Force refresh to persist state properly in Streamlit UI
            st.rerun()
else:
    st.info("No active chat sessions found. Please click '+ New Document Chat' on the sidebar.")