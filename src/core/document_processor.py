"""
Document processing module for handling various file formats.
Supports PDF, DOCX, TXT, and Markdown files.
"""

import logging
from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)

from ..config import config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading and processing for various file formats."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document from file path.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        try:
            file_path = Path(file_path)
            extension = file_path.suffix.lower()
            
            if extension == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif extension in [".txt", ".md"]:
                loader = TextLoader(str(file_path), encoding="utf-8")
            elif extension == ".docx":
                loader = UnstructuredWordDocumentLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file format: {extension}")
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {file_path}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents by splitting them into chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of processed Document chunks
        """
        try:
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "processed_at": str(Path(__file__).name),
                    "chunk_size": len(chunk.page_content)
                })
            
            logger.info(f"Split documents into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise
    
    def load_and_process(self, file_path: str) -> List[Document]:
        """
        Load and process a document in one step.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of processed Document chunks
        """
        documents = self.load_document(file_path)
        return self.process_documents(documents)
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate if a file can be processed.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                return False
            
            # Check file extension
            if file_path.suffix.lower() not in config.ALLOWED_EXTENSIONS:
                return False
            
            # Check file size
            if file_path.stat().st_size > config.MAX_FILE_SIZE:
                return False
            
            return True
            
        except Exception:
            return False