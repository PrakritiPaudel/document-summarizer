"""
Test configuration and fixtures for the Study Notes Summarizer.
"""

import pytest
import tempfile
import os
from pathlib import Path

# Test configuration
@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    This is a sample research paper about sentiment analysis. 
    The paper introduces a novel approach to sentiment classification 
    using deep learning techniques. The methodology involves training 
    a neural network on a large dataset of movie reviews. The results 
    show significant improvement over baseline methods with an accuracy 
    of 92%. The paper concludes that deep learning approaches are 
    effective for sentiment analysis tasks.
    """

@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    # This would normally be actual PDF bytes, but for testing
    # we'll use a simple text representation
    return b"Sample PDF content about sentiment analysis research"

# Mock environment variables for testing
@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "./test_data/chroma_db")
    monkeypatch.setenv("UPLOAD_DIRECTORY", "./test_uploads")
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "10")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")