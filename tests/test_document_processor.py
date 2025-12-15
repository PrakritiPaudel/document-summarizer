"""
Tests for document processing functionality.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from src.core.document_processor import DocumentProcessor

class TestDocumentProcessor:
    """Test cases for DocumentProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor()
    
    def test_initialization(self):
        """Test DocumentProcessor initialization."""
        assert self.processor is not None
        assert self.processor.text_splitter is not None
    
    def test_get_supported_formats(self):
        """Test getting supported file formats."""
        formats = self.processor.get_supported_formats()
        expected_formats = ['.pdf', '.docx', '.txt', '.md']
        assert all(fmt in formats for fmt in expected_formats)
    
    def test_validate_file_nonexistent(self):
        """Test validation of non-existent file."""
        result = self.processor.validate_file("nonexistent.pdf")
        assert result is False
    
    def test_validate_file_unsupported_format(self, temp_dir):
        """Test validation of unsupported file format."""
        # Create a test file with unsupported extension
        test_file = temp_dir / "test.xyz"
        test_file.write_text("test content")
        
        result = self.processor.validate_file(test_file)
        assert result is False
    
    def test_validate_file_supported_format(self, temp_dir):
        """Test validation of supported file format."""
        # Create a test file with supported extension
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        result = self.processor.validate_file(test_file)
        assert result is True
    
    @patch('src.core.document_processor.TextLoader')
    def test_load_text_document(self, mock_loader, temp_dir):
        """Test loading text document."""
        # Create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test document content")
        
        # Mock the loader
        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance
        mock_loader_instance.load.return_value = [
            Mock(page_content="Test document content", metadata={})
        ]
        
        # Test loading
        documents = self.processor.load_document(test_file)
        
        assert len(documents) == 1
        assert documents[0].page_content == "Test document content"
        mock_loader.assert_called_once_with(str(test_file), encoding='utf-8')
    
    def test_process_documents(self, sample_text):
        """Test processing documents into chunks."""
        # Create mock document
        mock_doc = Mock()
        mock_doc.page_content = sample_text
        mock_doc.metadata = {}
        
        documents = [mock_doc]
        
        # Process documents
        chunks = self.processor.process_documents(documents)
        
        assert len(chunks) > 0
        # Check that metadata was added
        for chunk in chunks:
            assert 'chunk_id' in chunk.metadata
            assert 'chunk_size' in chunk.metadata
    
    def test_load_and_process_integration(self, temp_dir, sample_text):
        """Test the integrated load and process workflow."""
        # Create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text(sample_text)
        
        with patch('src.core.document_processor.TextLoader') as mock_loader:
            # Mock the loader
            mock_loader_instance = Mock()
            mock_loader.return_value = mock_loader_instance
            mock_loader_instance.load.return_value = [
                Mock(page_content=sample_text, metadata={})
            ]
            
            # Test the integrated workflow
            chunks = self.processor.load_and_process(test_file)
            
            assert len(chunks) > 0
            assert all('chunk_id' in chunk.metadata for chunk in chunks)