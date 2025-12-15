"""
Configuration module for the Study Notes Summarizer.
Handles environment variables and application settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for application settings."""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
    
    # LangSmith Configuration
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "study-notes-summarizer")
    
    # Model Configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    
    # Document Processing
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Vector Store Configuration
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
    
    # File Upload Configuration
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10")) * 1024 * 1024  # 10MB in bytes
    ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"]
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "app.log")

# Create global config instance
config = Config()