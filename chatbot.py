import os
from typing import List, Dict, Any, Optional
# Assuming these are in the same project structure
from document_processor import DocumentProcessor
from vector_store import VectorStore
from ai_analyzer import AIAnalyzer
import logging
from dotenv import load_dotenv # Import load_dotenv once

# Configure basic logging for the entire application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # Get a logger for this module

# --- Corrected API Key Loading and Configuration ---
# Load environment variables from .env file FIRST.
# This should be at the very top of your main script or any script that needs
# to access environment variables before they are used.
load_dotenv() 

# Get API key and LLM model name from environment variables
api_key = os.getenv('GOOGLE_API_KEY')
llm_model_name = os.getenv('LLM_MODEL', 'gemini-1.5-pro') # Default if not set in .env
embedding_model_name = os.getenv('EMBEDDING_MODEL', 'models/embedding-001') # Default if not set in .env
db_path = os.getenv('DB_PATH', './vector_db') # Default if not set in .env

# Configure the Google Generative AI library only if the API key is found
if api_key:
    # Import google.generativeai here, after loading dotenv and getting the key
    import google.generativeai as genai 
    genai.configure(api_key=api_key)
    # Log only the first few characters of the API key for security
    logger.info(f"Loaded API Key (first 5 chars): {api_key[:5]}*****")
else:
    logger.error("Error: GOOGLE_API_KEY environment variable not found. Please ensure your .env file is correct.")
    # You might want to raise an exception or exit here if the API key is strictly required
    # For example: raise ValueError("GOOGLE_API_KEY not set. Cannot proceed.")

logger.info(f"Using LLM Model: {llm_model_name}")
logger.info(f"Using Embedding Model: {embedding_model_name}")
logger.info(f"Using Vector DB Path: {db_path}")

# --- End of Corrected API Key Loading ---


class DocumentChatbot:
    """Main chatbot class that integrates all components for document processing and analysis."""

    def __init__(self,
                 db_path: str = db_path, # Use the value read from env, or the default if not set
                 embedding_model: str = embedding_model_name, # Use the value read from env
                 llm_model: str = llm_model_name): # Use the value read from env

        # Initialize the sub-components
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore(db_path=db_path, embedding_model=embedding_model)
        self.ai_analyzer = AIAnalyzer(model_name=llm_model)
        
        # Keep track of processed files to avoid re-processing
        # Consider making this persistent if restarting the app should remember processed files
        self.processed_files = set() 

        logger.info("DocumentChatbot initialized successfully")

    def _normalize_filename(self, path: str) -> str:
        """Normalizes a file path to a lower-case basename for consistent comparison."""
        return os.path.basename(path).strip().lower()

    def upload_document(self, file_path: str) -> Dict[str, Any]:
        """
        Processes and uploads a document to the vector store.
        Args:
            file_path: The path to the document file.
        Returns:
            A dictionary indicating the status of the upload.
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If no content can be extracted or all chunks are empty.
            Exception: For other processing errors.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check if file has already been processed in the current session
            if file_path in self.processed_files:
                logger.info(f"File {file_path} already processed. Skipping.")
                return {"status": "already_processed", "file_path": file_path}

            # Get basic document information
            doc_info = self.document_processor.get_document_info(file_path)
            logger.info(f"Processing document: {doc_info.get('file_name', file_path)}")

            # Process the document into LangChain Document objects (chunks)
            documents = self.document_processor.process_document(file_path)
            if not documents:
                raise ValueError("No content extracted from document.")

            # Check if extracted content is meaningful
            total_content_length = sum(len(doc.page_content.strip()) for doc in documents)
            if total_content_length == 0:
                raise ValueError("All document chunks are empty after processing.")

            # Add the document chunks to the vector store for retrieval
            self.vector_store.add_documents(documents)
            self.processed_files.add(file_path) # Mark as processed

            logger.info(f"Successfully processed {len(documents)} chunks from {file_path}")
            return {
                "status": "success",
                "file_path": file_path,
                "document_info": doc_info,
                "chunks_created": len(documents)
            }

        except Exception as e:
            logger.error(f"Error uploading document {file_path}: {e}", exc_info=True)
            raise # Re-raise the exception after logging

    def _get_documents(self, file_path: Optional[str]) -> List[Any]:
        """
        Retrieves documents from the vector store, optionally filtered by file path.
        If file_path is None, retrieves all documents.
        """
        # Note: similarity_search("", k=N) usually retrieves the top N documents
        # that are "most similar" to an empty query, which often means
        # it just returns some documents, but not necessarily ALL.
        # If you truly want ALL documents, your VectorStore implementation
        # should have a method like `get_all_documents()`.
        # For now, k=1000 is a large number to approximate "all".
        all_docs = self.vector_store.similarity_search("", k=1000) 
        
        if file_path:
            target_name = self._normalize_filename(file_path)
            # Filter documents by their normalized source metadata
            filtered_docs = [
                doc for doc in all_docs
                if self._normalize_filename(doc.metadata.get("source", "")) == target_name
            ]
            logger.debug(f"Filtered {len(filtered_docs)} documents for {file_path}")
            return filtered_docs
        
        logger.debug(f"Returning all {len(all_docs)} documents.")
        return all_docs

    def get_summary(self, file_path: Optional[str] = None) -> str:
        """Generates a summary for the specified document(s)."""
        try:
            documents = self._get_documents(file_path)
            if not documents:
                return f"No documents found for '{file_path}'." if file_path else "No documents found to summarize."

            full_text = "\n\n".join([doc.page_content for doc in documents])
            if not full_text.strip():
                return "No text content found in documents to summarize."

            logger.info(f"Generating summary for {file_path if file_path else 'all documents'}")
            return self.ai_analyzer.summarize_text(full_text)
        except Exception as e:
            logger.error(f"Error generating summary: {e}", exc_info=True)
            # AIAnalyzer's methods already handle quota errors with custom messages
            # so we just return the string representation of the error here.
            return f"Error generating summary: {str(e)}"

    def get_topics(self, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extracts main topics from the specified document(s)."""
        try:
            documents = self._get_documents(file_path)
            if not documents:
                return [] # Return empty list if no documents

            full_text = "\n\n".join([doc.page_content for doc in documents])
            if not full_text.strip():
                return [] # Return empty list if no extractable text

            logger.info(f"Extracting topics for {file_path if file_path else 'all documents'}")
            return self.ai_analyzer.extract_topics(full_text)
        except Exception as e:
            logger.error(f"Error extracting topics: {e}", exc_info=True)
            return [] # Return empty list on error

    def generate_mcqs(self, file_path: Optional[str] = None, num_questions: int = 5) -> List[Dict[str, Any]]:
        """Generates multiple-choice questions for the specified document(s)."""
        try:
            documents = self._get_documents(file_path)
            if not documents:
                return []

            full_text = "\n\n".join([doc.page_content for doc in documents])
            if not full_text.strip():
                return []

            logger.info(f"Generating {num_questions} MCQs for {file_path if file_path else 'all documents'}")
            return self.ai_analyzer.generate_mcqs(full_text, num_questions)
        except Exception as e:
            logger.error(f"Error generating MCQs: {e}", exc_info=True)
            return []

    def ask_question(self, question: str, file_path: Optional[str] = None) -> str:
        """Answers a question based on relevant document chunks."""
        try:
            # Perform similarity search to find relevant document chunks
            relevant_docs = self.vector_store.similarity_search(question, k=5) 
            
            # If a specific file is requested, filter the relevant chunks
            if file_path:
                target_name = self._normalize_filename(file_path)
                relevant_docs = [
                    doc for doc in relevant_docs
                    if self._normalize_filename(doc.metadata.get("source", "")) == target_name
                ]

            if not relevant_docs:
                return f"I couldn't find any relevant information for '{question}' in '{file_path}'." if file_path else "I couldn't find any relevant information for your question."

            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            if not context.strip():
                return "No text content found in relevant documents to answer your question."

            logger.info(f"Answering question: '{question}' using context from {len(relevant_docs)} chunks.")
            return self.ai_analyzer.answer_question(context, question)
        except Exception as e:
            logger.error(f"Error answering question '{question}': {e}", exc_info=True)
            return f"Error: {str(e)}"

    def get_document_analysis(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Performs a comprehensive analysis (summary, topics, MCQs) on the specified document(s).
        """
        try:
            documents = self._get_documents(file_path)
            if not documents:
                msg = f"No documents found for '{file_path}' for analysis." if file_path else "No documents found for analysis."
                return {"summary": msg, "topics": [], "mcqs": [], "document_count": 0, "total_length": 0}

            # AIAnalyzer's analyze_document method will consolidate the text from documents
            # and perform all analyses, handling empty text internally.
            logger.info(f"Performing full analysis for {file_path if file_path else 'all documents'}")
            return self.ai_analyzer.analyze_document(documents)
        except Exception as e:
            logger.error(f"Error analyzing documents: {e}", exc_info=True)
            return {"summary": f"Error: {str(e)}", "topics": [], "mcqs": [], "document_count": 0, "total_length": 0}

    def get_stats(self) -> Dict[str, Any]:
        """Retrieves operational statistics of the chatbot and vector store."""
        try:
            stats = {
                "vector_store_stats": self.vector_store.get_collection_stats(),
                "processed_files": list(self.processed_files),
                "processed_files_count": len(self.processed_files)
            }
            logger.info("Retrieved chatbot statistics.")
            return stats
        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            return {"error": str(e)}

    def clear_all_data(self) -> None:
        """Clears all data from the vector store and resets processed files."""
        try:
            self.vector_store.clear_database()
            self.processed_files.clear()
            logger.info("Cleared all data successfully.")
        except Exception as e:
            logger.error(f"Error clearing data: {e}", exc_info=True)
            raise

    def remove_document(self, file_path: str) -> bool:
        """Removes a specific document and its chunks from the vector store."""
        try:
            self.vector_store.delete_documents_by_source(file_path)
            self.processed_files.discard(file_path) # Remove from in-memory set
            logger.info(f"Removed document: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error removing document {file_path}: {e}", exc_info=True)
            return False