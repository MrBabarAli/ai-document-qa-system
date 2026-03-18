# app/main.py
"""
Streamlit Web Interface for AI Document QA System
"""

import streamlit as st
import tempfile
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.text_extractor import TextExtractor
from app.qa_engine import QAEngine
import time

# Page configuration
st.set_page_config(
    page_title="AI Document QA System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'qa_engine' not in st.session_state:
    st.session_state.qa_engine = QAEngine(persist_directory="./production_db")
    st.session_state.processed_files = []
    st.session_state.messages = []

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .main-header {
        color: #1e3c72;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        color: #2a5298;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        margin-left: 20%;
    }
    .bot-message {
        background-color: #e9ecef;
        color: black;
        margin-right: 20%;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📚 AI Document Question Answering System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload documents and ask questions about their content</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📁 Document Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT)",
        type=['pdf', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                with st.spinner(f'Processing {uploaded_file.name}...'):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Extract text
                        extractor = TextExtractor()
                        file_type = uploaded_file.name.split('.')[-1].lower()
                        
                        if file_type == 'pdf':
                            text = extractor.extract_from_pdf(uploaded_file.getvalue())
                        else:
                            text = extractor.extract_from_txt(uploaded_file.getvalue())
                        
                        # Add to QA engine
                        if text and not text.startswith("Error"):
                            st.session_state.qa_engine.add_document(text, uploaded_file.name)
                            st.session_state.processed_files.append(uploaded_file.name)
                            st.success(f"✅ {uploaded_file.name} processed successfully!")
                        else:
                            st.error(f"❌ Could not extract text from {uploaded_file.name}")
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
    
    # Display processed files
    if st.session_state.processed_files:
        st.markdown("### 📄 Processed Documents:")
        for file in st.session_state.processed_files:
            st.markdown(f"- {file}")
    
    # System stats
    if st.button("📊 System Statistics"):
        stats = st.session_state.qa_engine.get_stats()
        st.markdown("### System Stats:")
        st.json(stats)
    
    # Clear button
    if st.button("🧹 Clear All Documents"):
        st.session_state.qa_engine.clear_knowledge_base()
        st.session_state.processed_files = []
        st.session_state.messages = []
        st.success("All documents cleared!")
        st.rerun()

# Main chat interface
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 💬 Ask Questions")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}** (Similarity: {source['similarity']:.2f})")
                        st.markdown(f"*From: {source['metadata']['source']}*")
                        st.markdown(f"```\n{source['text']}\n```")
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.qa_engine.ask(prompt, n_results=3)
                
                # Display answer
                st.markdown(response['answer'])
                
                # Show sources
                if response['sources']:
                    with st.expander("📚 Sources Used"):
                        for i, source in enumerate(response['sources'], 1):
                            st.markdown(f"**Source {i}** (Similarity: {source['similarity']:.2f})")
                            st.markdown(f"*From: {source['metadata']['source']}*")
                            st.markdown(f"```\n{source['text']}\n```")
                            st.markdown("---")
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response['answer'],
            "sources": response['sources']
        })

with col2:
    st.markdown("## ℹ️ How to Use")
    st.info(
        """
        1. **Upload documents** using the sidebar
        2. **Ask questions** about the content
        3. **View answers** with source references
        
        **Supported formats:** PDF, TXT
        
        **Tips:**
        - Ask specific questions
        - Upload multiple documents
        - Check sources for verification
        """
    )
    
    # Quick example
    st.markdown("### 📝 Example Questions")
    example_questions = [
        "What is machine learning?",
        "Summarize the main points",
        "What are the key findings?"
    ]
    
    for q in example_questions:
        if st.button(q):
            # This would trigger the question
            st.info(f"Try asking: '{q}'")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Built with Streamlit • Powered by Sentence Transformers & ChromaDB
    </div>
    """,
    unsafe_allow_html=True
)