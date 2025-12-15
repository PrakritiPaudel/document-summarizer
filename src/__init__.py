"""
Main package initialization for Study Notes Summarizer.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "AI-powered research paper analysis system with RAG and PDO prompting"

from .config import config
from .core import RAGSummarizer, DocumentProcessor, VectorStoreManager
from .prompts import SummaryPromptTemplates
from .utils import FileManager, LangSmithMonitor

__all__ = [
    "config",
    "RAGSummarizer", 
    "DocumentProcessor",
    "VectorStoreManager",
    "SummaryPromptTemplates",
    "FileManager",
    "LangSmithMonitor"
]