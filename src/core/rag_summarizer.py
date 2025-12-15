"""
RAG (Retrieval-Augmented Generation) system for document summarization.
Combines document retrieval with LLM generation for accurate summaries.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_community.callbacks.manager import get_openai_callback

from .document_processor import DocumentProcessor
from .vector_store import VectorStoreManager
from ..prompts.summary_prompts import SummaryPromptTemplates
from ..config import config

logger = logging.getLogger(__name__)

class RAGSummarizer:
    """RAG-based document summarization system."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.prompt_templates = SummaryPromptTemplates()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=config.TEMPERATURE,
            openai_api_key=config.OPENAI_API_KEY
        )
        
        # Initialize components
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        
    def initialize_system(self, documents: Optional[List[Document]] = None) -> None:
        """
        Initialize the RAG system with documents.
        
        Args:
            documents: Optional list of documents to initialize with
        """
        try:
            logger.info("Initializing RAG system...")
            
            # Initialize vector store
            self.vector_store = self.vector_store_manager.initialize_vector_store(documents)
            
            # Create retriever
            self.retriever = self.vector_store_manager.get_retriever(k=6)
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": self.prompt_templates.get_qa_prompt()
                }
            )
            
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            raise
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document and add it to the knowledge base.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Processing results dictionary
        """
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Validate file
            if not self.document_processor.validate_file(file_path):
                raise ValueError(f"Invalid or unsupported file: {file_path}")
            
            # Load and process document
            documents = self.document_processor.load_and_process(file_path)
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source_file': file_path,
                    'processed_at': datetime.now().isoformat(),
                    'document_type': 'research_paper'
                })
            
            # Add to vector store
            if not self.vector_store:
                self.initialize_system(documents)
            else:
                self.vector_store_manager.add_documents(documents)
            
            result = {
                'success': True,
                'file_path': file_path,
                'chunks_created': len(documents),
                'total_documents': self.vector_store_manager.get_document_count(),
                'processed_at': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully processed document: {len(documents)} chunks created")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return {
                'success': False,
                'file_path': file_path,
                'error': str(e),
                'processed_at': datetime.now().isoformat()
            }
    
    def generate_summary(
        self, 
        summary_type: str = "comprehensive",
        focus_area: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a summary of the processed documents.
        
        Args:
            summary_type: Type of summary ("comprehensive", "key_points", "methodology", "findings")
            focus_area: Optional focus area for targeted summary
            
        Returns:
            Summary results dictionary
        """
        try:
            logger.info(f"Generating {summary_type} summary...")
            
            if not self.qa_chain:
                raise ValueError("RAG system not initialized. Please process a document first.")
            
            # Get appropriate prompt based on summary type
            query = self.prompt_templates.get_summary_query(summary_type, focus_area)
            
            # Generate summary using RAG
            with get_openai_callback() as cb:
                result = self.qa_chain({"query": query})
            
            # Extract information
            summary = result['result']
            source_docs = result['source_documents']
            
            # Create comprehensive result
            summary_result = {
                'success': True,
                'summary_type': summary_type,
                'focus_area': focus_area,
                'summary': summary,
                'source_chunks': len(source_docs),
                'token_usage': {
                    'total_tokens': cb.total_tokens,
                    'prompt_tokens': cb.prompt_tokens,
                    'completion_tokens': cb.completion_tokens,
                    'total_cost': cb.total_cost
                },
                'generated_at': datetime.now().isoformat(),
                'sources': [
                    {
                        'chunk_id': doc.metadata.get('chunk_id', 'unknown'),
                        'source_file': doc.metadata.get('source_file', 'unknown'),
                        'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    }
                    for doc in source_docs
                ]
            }
            
            logger.info(f"Successfully generated {summary_type} summary")
            return summary_result
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {
                'success': False,
                'summary_type': summary_type,
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a specific question about the processed documents.
        
        Args:
            question: Question to ask
            
        Returns:
            Answer results dictionary
        """
        try:
            logger.info(f"Processing question: {question[:50]}...")
            
            if not self.qa_chain:
                raise ValueError("RAG system not initialized. Please process a document first.")
            
            # Process question through RAG
            with get_openai_callback() as cb:
                result = self.qa_chain({"query": question})
            
            answer_result = {
                'success': True,
                'question': question,
                'answer': result['result'],
                'source_chunks': len(result['source_documents']),
                'token_usage': {
                    'total_tokens': cb.total_tokens,
                    'prompt_tokens': cb.prompt_tokens,
                    'completion_tokens': cb.completion_tokens,
                    'total_cost': cb.total_cost
                },
                'answered_at': datetime.now().isoformat(),
                'sources': [
                    {
                        'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        'source_file': doc.metadata.get('source_file', 'unknown')
                    }
                    for doc in result['source_documents']
                ]
            }
            
            logger.info("Successfully processed question")
            return answer_result
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                'success': False,
                'question': question,
                'error': str(e),
                'answered_at': datetime.now().isoformat()
            }
    
    def get_document_insights(self) -> Dict[str, Any]:
        """
        Get insights about the processed documents.
        
        Returns:
            Document insights dictionary
        """
        try:
            if not self.vector_store:
                return {'error': 'No documents processed'}
            
            doc_count = self.vector_store_manager.get_document_count()
            
            # Generate multiple summary types
            insights = {
                'document_count': doc_count,
                'summaries': {}
            }
            
            if doc_count > 0:
                summary_types = ['key_points', 'methodology', 'findings']
                for summary_type in summary_types:
                    result = self.generate_summary(summary_type)
                    if result['success']:
                        insights['summaries'][summary_type] = result['summary']
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting document insights: {str(e)}")
            return {'error': str(e)}
    
    def clear_knowledge_base(self) -> bool:
        """
        Clear the entire knowledge base.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.vector_store_manager.clear_vector_store()
            self.vector_store = None
            self.retriever = None
            self.qa_chain = None
            
            logger.info("Knowledge base cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {str(e)}")
            return False