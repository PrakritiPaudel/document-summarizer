"""
Vector store management for document embeddings and retrieval.
Uses ChromaDB for efficient similarity search.
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from ..config import config

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages vector store operations for document retrieval."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=config.OPENAI_API_KEY
        )
        self.vector_store = None
        self.retriever = None
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """
        Create a vector store from documents.
        
        Args:
            documents: List of Document objects to embed
            
        Returns:
            ChromaDB vector store
        """
        try:
            # Ensure the directory exists
            Path(config.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
            
            # Create vector store
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=config.VECTOR_DB_PATH,
                collection_name=config.COLLECTION_NAME
            )
            
            # Persist the vector store
            self.vector_store.persist()
            
            logger.info(f"Created vector store with {len(documents)} documents")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def load_vector_store(self) -> Optional[Chroma]:
        """
        Load existing vector store from disk.
        
        Returns:
            ChromaDB vector store if exists, None otherwise
        """
        try:
            if Path(config.VECTOR_DB_PATH).exists():
                self.vector_store = Chroma(
                    persist_directory=config.VECTOR_DB_PATH,
                    embedding_function=self.embeddings,
                    collection_name=config.COLLECTION_NAME
                )
                logger.info("Loaded existing vector store")
                return self.vector_store
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return None
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to existing vector store.
        
        Args:
            documents: List of Document objects to add
        """
        try:
            if self.vector_store is None:
                raise ValueError("Vector store not initialized")
            
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def create_retriever(self, k: int = 5, search_type: str = "similarity"):
        """
        Create a retriever from the vector store.
        
        Args:
            k: Number of documents to retrieve
            search_type: Type of search ("similarity" or "mmr")
            
        Returns:
            Chroma retriever object
        """
        try:
            if self.vector_store is None:
                raise ValueError("Vector store not initialized")
            
            self.retriever = self.vector_store.as_retriever(
                search_type=search_type,
                search_kwargs={"k": k}
            )
            
            logger.info(f"Created retriever with k={k}, search_type={search_type}")
            return self.retriever
            
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar Document objects
        """
        try:
            if self.vector_store is None:
                raise ValueError("Vector store not initialized")
            
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents for query")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
    
    def delete_collection(self) -> None:
        """Delete the current collection and reset vector store."""
        try:
            if self.vector_store is not None:
                self.vector_store.delete_collection()
                self.vector_store = None
                self.retriever = None
                
            logger.info("Deleted vector store collection")
            
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            if self.vector_store is None:
                return {"exists": False}
            
            # Get collection count
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                "exists": True,
                "document_count": count,
                "collection_name": config.COLLECTION_NAME,
                "path": config.VECTOR_DB_PATH
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"exists": False, "error": str(e)}