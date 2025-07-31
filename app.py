import streamlit as st
import os
from chatbot import DocumentChatbot # Ensure chatbot.py is in the same directory or accessible
import tempfile
from typing import List, Dict, Any
import json
from dotenv import load_dotenv

# Load environment variables from .env file at the very start
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Document AI Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .topic-tag {
        background-color: #007bff;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Function to initialize the chatbot
def initialize_chatbot_instance():
    """
    Initializes the DocumentChatbot. This function is called once
    and stores the instance in session state.
    """
    try:
        if st.session_state.chatbot is None:
            st.session_state.chatbot = DocumentChatbot()
            st.success("Chatbot initialized successfully!")
        return st.session_state.chatbot
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}. Please check your API key and environment setup.")
        st.session_state.chatbot = None # Ensure chatbot is None if initialization fails
        return None

# Attempt to initialize chatbot immediately if API key is already available
# This handles cases where the app restarts but the key is in .env or was previously set.
if os.getenv("GOOGLE_API_KEY") and st.session_state.chatbot is None:
    initialize_chatbot_instance()


def main():
    # Header
    st.markdown('<h1 class="main-header">📚 Document AI Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("Upload PDF, DOCX, or TXT files and get AI-powered analysis, summaries, topics, and MCQs!")

    # Sidebar
    with st.sidebar:
        st.markdown("## 🔧 Settings")

        # Get current API key from environment, allow user to input/override
        current_api_key_env = os.getenv("GOOGLE_API_KEY", "")
        api_key_input = st.text_input(
            "Google Gemini API Key", 
            type="password", 
            value=current_api_key_env, # Pre-fill if already set
            help="Enter your Google Gemini API key. This will be stored in the session."
        )

        # Only update environment variable and re-initialize if key has changed or is newly entered
        if api_key_input and api_key_input != current_api_key_env:
            os.environ["GOOGLE_API_KEY"] = api_key_input
            with st.spinner("API Key updated. Initializing chatbot..."):
                initialize_chatbot_instance()
            # Rerun to ensure the chatbot is fully initialized and UI reflects it
            st.rerun()
        elif not api_key_input and st.session_state.chatbot is None:
            st.warning("Please enter your Google Gemini API Key to enable AI features.")


        st.markdown("---")
        st.markdown("## 📁 File Management")

        if st.button("🗑️ Clear All Data"):
            if st.session_state.chatbot:
                with st.spinner("Clearing all data..."):
                    st.session_state.chatbot.clear_all_data()
                    st.session_state.chat_history = []
                    st.session_state.uploaded_files = []
                    st.success("All data cleared!")
                    st.rerun() # Rerun to clear UI
            else:
                st.info("Chatbot not initialized. No data to clear.")

        if st.session_state.chatbot:
            stats = st.session_state.chatbot.get_stats()
            st.markdown("### 📊 Statistics")
            st.metric("Processed Files Count", stats.get("processed_files_count", 0))
            if "vector_store_stats" in stats:
                st.metric("Total Document Chunks", stats["vector_store_stats"].get("total_documents", 0))
            
            # Display list of uploaded files with a remove button
            if st.session_state.uploaded_files:
                st.markdown("### 📄 Uploaded Files")
                for uploaded_file_name in st.session_state.uploaded_files:
                    col_file, col_remove = st.columns([0.7, 0.3])
                    with col_file:
                        st.write(uploaded_file_name)
                    with col_remove:
                        if st.button(f"Remove {uploaded_file_name}", key=f"remove_{uploaded_file_name}"):
                            # Need to reconstruct the full temporary path if you want to remove by path
                            # For simplicity, we'll just remove by basename from vector store
                            # This assumes basename is unique enough for removal
                            if st.session_state.chatbot.remove_document(uploaded_file_name): # Pass basename
                                st.session_state.uploaded_files.remove(uploaded_file_name)
                                st.success(f"Removed {uploaded_file_name}")
                            else:
                                st.error(f"Failed to remove {uploaded_file_name}")
                            st.rerun()


    # Main columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload Documents</h2>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Choose PDF, DOCX, or TXT files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            key="file_uploader" # Add a key to prevent re-rendering issues
        )

        if uploaded_files and st.button("📥 Process Documents", key="process_docs_button"):
            if not st.session_state.chatbot:
                st.error("Please enter your API key in the sidebar to initialize the chatbot first!")
            else:
                with st.spinner("Processing documents... This may take a moment for large files."):
                    for uploaded_file in uploaded_files:
                        # Create a temporary file to save the uploaded content
                        ext = uploaded_file.name.split('.')[-1]
                        # Use tempfile.mkstemp to get a unique temp file path
                        fd, tmp_path = tempfile.mkstemp(suffix=f".{ext}")
                        try:
                            with os.fdopen(fd, 'wb') as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())

                            result = st.session_state.chatbot.upload_document(tmp_path)
                            if result["status"] == "success":
                                st.success(f"✅ '{uploaded_file.name}' processed! Chunks: {result['chunks_created']}")
                                if uploaded_file.name not in st.session_state.uploaded_files:
                                    st.session_state.uploaded_files.append(uploaded_file.name)
                            elif result["status"] == "already_processed":
                                st.info(f"ℹ️ '{uploaded_file.name}' was already processed. Skipping.")
                            
                        except Exception as e:
                            st.error(f"❌ Error processing '{uploaded_file.name}': {str(e)}")
                        finally:
                            # Ensure the temporary file is deleted
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                    st.rerun() # Rerun to update the list of uploaded files and analysis options

        # Document Analysis Section
        if st.session_state.chatbot and st.session_state.uploaded_files:
            st.markdown('<h2 class="sub-header">🔍 Document Analysis</h2>', unsafe_allow_html=True)
            # Add "All Documents" as an option to analyze combined content
            selected_file_for_analysis = st.selectbox(
                "Select file for analysis or 'All Documents'", 
                ["All Documents"] + st.session_state.uploaded_files,
                key="analysis_file_select"
            )

            if st.button("📊 Generate Analysis", key="generate_analysis_button"):
                if not st.session_state.chatbot:
                    st.error("Chatbot not initialized. Please enter API key.")
                else:
                    with st.spinner("Generating analysis (summary, topics, MCQs)..."):
                        # Pass None if "All Documents" is selected
                        file_to_analyze = None if selected_file_for_analysis == "All Documents" else selected_file_for_analysis
                        analysis = st.session_state.chatbot.get_document_analysis(file_to_analyze)

                        st.markdown("### 📝 Summary")
                        st.write(analysis.get("summary", "No summary generated."))

                        if analysis.get("topics"):
                            st.markdown("### 🏷️ Topics")
                            # Check if it's a list before iterating
                            if isinstance(analysis["topics"], list):
                                # Filter out potential error topics from AIAnalyzer
                                display_topics = [t for t in analysis["topics"] if not t.get("topic", "").startswith("⚠️ Quota exceeded")]
                                if display_topics:
                                    for topic in display_topics:
                                        confidence = topic.get("confidence", 0)
                                        st.markdown(f'<span class="topic-tag">{topic["topic"]} ({confidence:.2f})</span>', unsafe_allow_html=True)
                                else:
                                    st.info("No specific topics extracted or quota exceeded for topics.")
                            else:
                                st.warning(f"Topics format unexpected: {analysis['topics']}")
                        elif "topics" in analysis and analysis["topics"] == []:
                            st.info("No topics extracted from the document(s).")
                        else:
                            st.warning(f"Topics generation failed: {analysis.get('topics', 'Unknown error')}")


                        if analysis.get("mcqs"):
                            st.markdown("### ❓ Multiple Choice Questions")
                            # Check if it's a list before iterating
                            if isinstance(analysis["mcqs"], list):
                                # Filter out potential error MCQs from AIAnalyzer
                                display_mcqs = [m for m in analysis["mcqs"] if not m.get("question", "").startswith("⚠️ Quota exceeded")]
                                if display_mcqs:
                                    for i, mcq in enumerate(display_mcqs, 1):
                                        with st.expander(f"Question {i}"):
                                            st.markdown(f"**{mcq.get('question', 'N/A')}**")
                                            options = mcq.get('options', {})
                                            for opt, txt in options.items():
                                                st.write(f"{opt}. {txt}")
                                            st.success(f"**Correct Answer: {mcq.get('correct_answer', 'N/A')}**")
                                            st.info(f"**Explanation:** {mcq.get('explanation', 'No explanation provided.')}")
                                else:
                                    st.info("No specific MCQs generated or quota exceeded for MCQs.")
                            else:
                                st.warning(f"MCQs format unexpected: {analysis['mcqs']}")
                        elif "mcqs" in analysis and analysis["mcqs"] == []:
                            st.info("No MCQs generated from the document(s).")
                        else:
                            st.warning(f"MCQs generation failed: {analysis.get('mcqs', 'Unknown error')}")


    with col2:
        st.markdown('<h2 class="sub-header">💬 Chat Interface</h2>', unsafe_allow_html=True)

        if st.session_state.chatbot:
            # Display chat messages from history
            for msg in st.session_state.chat_history:
                role = msg["role"]
                css_class = "user-message" if role == "user" else "bot-message"
                speaker = "👤 You" if role == "user" else "🤖 Bot"
                st.markdown(f'<div class="chat-message {css_class}"><strong>{speaker}:</strong> {msg["content"]}</div>', unsafe_allow_html=True)

            user_question = st.text_input("Ask a question about your documents:", key="chat_input")
            
            # Add a selectbox for specific file context in chat
            chat_file_context = st.selectbox(
                "Limit chat to a specific file (optional)",
                ["All Documents"] + st.session_state.uploaded_files,
                key="chat_file_context_select"
            )
            file_context_for_chat = None if chat_file_context == "All Documents" else chat_file_context


            col_ask, col_summary, col_topics, col_mcqs = st.columns(4)

            with col_ask:
                if st.button("❓ Ask Question", key="ask_question_button") and user_question:
                    with st.spinner("Thinking..."):
                        answer = st.session_state.chatbot.ask_question(user_question, file_context_for_chat)
                        st.session_state.chat_history.extend([
                            {"role": "user", "content": user_question},
                            {"role": "assistant", "content": answer}
                        ])
                    st.rerun()

            with col_summary:
                if st.button("📝 Get Summary (Chat)", key="get_summary_chat_button"):
                    with st.spinner("Generating summary..."):
                        summary = st.session_state.chatbot.get_summary(file_context_for_chat)
                        st.session_state.chat_history.extend([
                            {"role": "user", "content": f"Generate a summary for {chat_file_context}"},
                            {"role": "assistant", "content": summary}
                        ])
                    st.rerun()

            with col_topics:
                if st.button("🏷️ Get Topics (Chat)", key="get_topics_chat_button"):
                    with st.spinner("Extracting topics..."):
                        topics = st.session_state.chatbot.get_topics(file_context_for_chat)
                        if topics:
                            topics_text = "**Topics found:**\n" + "\n".join(
                                [f"• {topic['topic']} (confidence: {topic.get('confidence', 0):.2f})" for topic in topics if not topic.get("topic", "").startswith("⚠️ Quota exceeded")]
                            )
                        else:
                            topics_text = "No topics extracted or quota exceeded."
                        st.session_state.chat_history.extend([
                            {"role": "user", "content": f"Extract topics for {chat_file_context}"},
                            {"role": "assistant", "content": topics_text}
                        ])
                    st.rerun()

            with col_mcqs:
                if st.button("❓ Generate MCQs (Chat)", key="generate_mcqs_chat_button"):
                    with st.spinner("Generating MCQs..."):
                        mcqs = st.session_state.chatbot.generate_mcqs(file_context_for_chat, num_questions=3)
                        if mcqs:
                            mcqs_text = "**Generated MCQs:**\n\n"
                            # Filter out potential error MCQs
                            display_mcqs_chat = [m for m in mcqs if not m.get("question", "").startswith("⚠️ Quota exceeded")]
                            if display_mcqs_chat:
                                for i, mcq in enumerate(display_mcqs_chat, 1):
                                    mcqs_text += f"{i}. {mcq.get('question', 'N/A')}\n"
                                    options = mcq.get('options', {})
                                    for opt, txt in options.items():
                                        mcqs_text += f"    {opt}. {txt}\n"
                                    mcqs_text += f"    **Answer: {mcq.get('correct_answer', 'N/A')}**\n"
                                    mcqs_text += f"    *{mcq.get('explanation', 'No explanation provided.')}*\n\n"
                            else:
                                mcqs_text = "No MCQs generated or quota exceeded."
                        else:
                            mcqs_text = "No MCQs generated or quota exceeded."

                        st.session_state.chat_history.extend([
                            {"role": "user", "content": f"Generate MCQs for {chat_file_context}"},
                            {"role": "assistant", "content": mcqs_text}
                        ])
                    st.rerun()
        else:
            st.info("Please enter your API key in the sidebar to initialize the chatbot and enable chat features.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ using LangChain, Google Gemini, and Streamlit</p>
        <p>Upload your documents and start exploring with AI!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()