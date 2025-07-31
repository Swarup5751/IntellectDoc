import os
import pickle
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Handles vector embeddings and similarity search using in-memory storage with persistence"""

    def __init__(self, db_path: str = "./vector_db", embedding_model: str = "models/embedding-001"):
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.embeddings = None
        self.documents: List[str] = []
        self.document_embeddings: List[List[float]] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []

        self._initialize_embeddings()
        self._load_or_create_db()

    def _initialize_embeddings(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment")

        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=self.embedding_model,
                google_api_key=api_key
            )
            logger.info(f"Initialized Google embeddings with model: {self.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise

    def _load_or_create_db(self):
        os.makedirs(self.db_path, exist_ok=True)
        db_file = Path(self.db_path) / "vector_store.pkl"
        if db_file.exists():
            try:
                with open(db_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get("documents", [])
                    self.document_embeddings = data.get("embeddings", [])
                    self.metadatas = data.get("metadatas", [])
                    self.ids = data.get("ids", [])
                logger.info(f"Loaded {len(self.documents)} documents from store")
            except Exception as e:
                logger.error(f"Error loading vector store: {str(e)}")
                raise
        else:
            logger.info("Creating new vector store")

    def _save_db(self):
        try:
            db_file = Path(self.db_path) / "vector_store.pkl"
            with open(db_file, 'wb') as f:
                pickle.dump({
                    "documents": self.documents,
                    "embeddings": self.document_embeddings,
                    "metadatas": self.metadatas,
                    "ids": self.ids
                }, f)
            logger.info(f"Saved vector store to {db_file}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            raise

    def _generate_id(self, text: str, i: int) -> str:
        return f"doc_{i}_{hashlib.md5(text.encode()).hexdigest()[:8]}"

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            logger.warning("No documents to add")
            return

        final_docs = [doc for doc in documents if doc.page_content and len(doc.page_content.strip()) >= 5]
        if not final_docs:
            raise ValueError("No valid documents with sufficient content")

        texts = [doc.page_content for doc in final_docs]
        metadatas = [doc.metadata for doc in final_docs]
        ids = [self._generate_id(text, len(self.documents) + i) for i, text in enumerate(texts)]

        try:
            embeddings = self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise

        self.documents.extend(texts)
        self.document_embeddings.extend(embeddings)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

        self._save_db()
        logger.info(f"Added {len(texts)} documents to vector store")

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        vec1, vec2 = np.array(vec1), np.array(vec2)
        norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return float(np.dot(vec1, vec2) / norm) if norm != 0 else 0.0

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if not self.documents:
            logger.warning("Vector store is empty")
            return []

        if not query.strip():
            return [
                Document(page_content=self.documents[i], metadata=self.metadatas[i])
                for i in range(min(k, len(self.documents)))
            ]

        try:
            query_vec = self.embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise

        similarities = [self._cosine_similarity(query_vec, doc_vec) for doc_vec in self.document_embeddings]
        top_indices = np.argsort(similarities)[-k:][::-1]

        return [
            Document(page_content=self.documents[i], metadata=self.metadatas[i])
            for i in top_indices
        ]

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        if not self.documents:
            logger.warning("Vector store is empty")
            return []

        if not query.strip():
            return [
                (Document(page_content=self.documents[i], metadata=self.metadatas[i]), 1.0)
                for i in range(min(k, len(self.documents)))
            ]

        try:
            query_vec = self.embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise

        similarities = [self._cosine_similarity(query_vec, doc_vec) for doc_vec in self.document_embeddings]
        top_indices = np.argsort(similarities)[-k:][::-1]

        return [
            (Document(page_content=self.documents[i], metadata=self.metadatas[i]), similarities[i])
            for i in top_indices
        ]

    def delete_documents_by_source(self, source: str) -> None:
        norm_source = source.strip().lower()
        indices_to_remove = [
            i for i, meta in enumerate(self.metadatas)
            if meta.get("source", "").strip().lower() == norm_source
        ]

        if not indices_to_remove:
            logger.info(f"No documents found for source: {source}")
            return

        for i in reversed(indices_to_remove):
            del self.documents[i]
            del self.document_embeddings[i]
            del self.metadatas[i]
            del self.ids[i]

        self._save_db()
        logger.info(f"Deleted {len(indices_to_remove)} documents from source: {source}")

    def clear_database(self) -> None:
        self.documents.clear()
        self.document_embeddings.clear()
        self.metadatas.clear()
        self.ids.clear()
        self._save_db()
        logger.info("Cleared vector store")

    def get_collection_stats(self) -> Dict[str, Any]:
        return {
            "total_documents": len(self.documents),
            "embedding_model": self.embedding_model,
            "db_path": self.db_path
        }
