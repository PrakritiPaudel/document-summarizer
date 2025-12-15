"""
LangSmith integration for monitoring and evaluation.
Provides observability and performance tracking for the RAG system.
"""

import logging
from typing import Dict, Any, Optional, List
import os
from datetime import datetime

# LangSmith imports
try:
    from langsmith import Client
    from langchain.callbacks import LangChainTracer
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    logging.warning("LangSmith not available. Install with: pip install langsmith")

from ..config import config

logger = logging.getLogger(__name__)

class LangSmithMonitor:
    """LangSmith integration for monitoring RAG operations."""
    
    def __init__(self):
        self.client = None
        self.tracer = None
        self.project_name = config.LANGCHAIN_PROJECT
        self.enabled = LANGSMITH_AVAILABLE and config.LANGCHAIN_TRACING_V2
        
        if self.enabled:
            self._initialize_langsmith()
    
    def _initialize_langsmith(self):
        """Initialize LangSmith client and tracer."""
        try:
            # Set environment variables for LangChain tracing
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = config.LANGCHAIN_ENDPOINT
            os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.project_name
            
            # Initialize client
            self.client = Client(
                api_url=config.LANGCHAIN_ENDPOINT,
                api_key=config.LANGCHAIN_API_KEY
            )
            
            # Initialize tracer
            self.tracer = LangChainTracer(project_name=self.project_name)
            
            logger.info(f"LangSmith monitoring initialized for project: {self.project_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangSmith: {str(e)}")
            self.enabled = False
    
    def get_tracer(self):
        """Get the LangChain tracer for callback integration."""
        return self.tracer if self.enabled else None
    
    def log_document_processing(self, file_path: str, result: Dict[str, Any]):
        """
        Log document processing events.
        
        Args:
            file_path: Path to processed document
            result: Processing result dictionary
        """
        if not self.enabled:
            return
        
        try:
            # Create a custom run for document processing
            metadata = {
                "operation": "document_processing",
                "file_path": file_path,
                "success": result.get('success', False),
                "chunks_created": result.get('chunks_created', 0),
                "timestamp": datetime.now().isoformat()
            }
            
            if result.get('success'):
                logger.info(f"Logged successful document processing: {file_path}")
            else:
                metadata["error"] = result.get('error', 'Unknown error')
                logger.info(f"Logged failed document processing: {file_path}")
                
        except Exception as e:
            logger.error(f"Error logging document processing: {str(e)}")
    
    def log_summary_generation(
        self, 
        summary_type: str, 
        result: Dict[str, Any], 
        focus_area: Optional[str] = None
    ):
        """
        Log summary generation events.
        
        Args:
            summary_type: Type of summary generated
            result: Summary generation result
            focus_area: Optional focus area
        """
        if not self.enabled:
            return
        
        try:
            metadata = {
                "operation": "summary_generation",
                "summary_type": summary_type,
                "focus_area": focus_area,
                "success": result.get('success', False),
                "token_usage": result.get('token_usage', {}),
                "source_chunks": result.get('source_chunks', 0),
                "timestamp": datetime.now().isoformat()
            }
            
            if result.get('success'):
                logger.info(f"Logged successful summary generation: {summary_type}")
            else:
                metadata["error"] = result.get('error', 'Unknown error')
                logger.info(f"Logged failed summary generation: {summary_type}")
                
        except Exception as e:
            logger.error(f"Error logging summary generation: {str(e)}")
    
    def log_question_answering(self, question: str, result: Dict[str, Any]):
        """
        Log question answering events.
        
        Args:
            question: Question asked
            result: QA result dictionary
        """
        if not self.enabled:
            return
        
        try:
            metadata = {
                "operation": "question_answering",
                "question": question,
                "success": result.get('success', False),
                "token_usage": result.get('token_usage', {}),
                "source_chunks": result.get('source_chunks', 0),
                "timestamp": datetime.now().isoformat()
            }
            
            if result.get('success'):
                logger.info(f"Logged successful Q&A: {question[:50]}...")
            else:
                metadata["error"] = result.get('error', 'Unknown error')
                logger.info(f"Logged failed Q&A: {question[:50]}...")
                
        except Exception as e:
            logger.error(f"Error logging Q&A: {str(e)}")
    
    def get_project_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent runs from the project.
        
        Args:
            limit: Maximum number of runs to retrieve
            
        Returns:
            List of run dictionaries
        """
        if not self.enabled or not self.client:
            return []
        
        try:
            runs = list(self.client.list_runs(
                project_name=self.project_name,
                limit=limit
            ))
            
            return [
                {
                    "id": run.id,
                    "name": run.name,
                    "start_time": run.start_time,
                    "end_time": run.end_time,
                    "status": run.status,
                    "inputs": run.inputs,
                    "outputs": run.outputs,
                    "error": run.error,
                    "execution_order": run.execution_order
                }
                for run in runs
            ]
            
        except Exception as e:
            logger.error(f"Error getting project runs: {str(e)}")
            return []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the project.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.enabled:
            return {"error": "LangSmith not enabled"}
        
        try:
            runs = self.get_project_runs(limit=100)
            
            if not runs:
                return {"message": "No runs found"}
            
            # Calculate basic metrics
            total_runs = len(runs)
            successful_runs = len([r for r in runs if r['status'] == 'success'])
            failed_runs = total_runs - successful_runs
            
            # Calculate average execution time for completed runs
            completed_runs = [r for r in runs if r['end_time'] and r['start_time']]
            avg_execution_time = 0
            
            if completed_runs:
                execution_times = [
                    (r['end_time'] - r['start_time']).total_seconds() 
                    for r in completed_runs
                ]
                avg_execution_time = sum(execution_times) / len(execution_times)
            
            return {
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
                "average_execution_time_seconds": avg_execution_time,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {"error": str(e)}
    
    def create_feedback(
        self, 
        run_id: str, 
        score: float, 
        feedback_text: Optional[str] = None
    ) -> bool:
        """
        Create feedback for a specific run.
        
        Args:
            run_id: ID of the run to provide feedback for
            score: Numerical score (0.0 to 1.0)
            feedback_text: Optional text feedback
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.client:
            return False
        
        try:
            self.client.create_feedback(
                run_id=run_id,
                key="user_score",
                score=score,
                comment=feedback_text
            )
            
            logger.info(f"Created feedback for run {run_id}: score={score}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating feedback: {str(e)}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and configuration.
        
        Returns:
            System status dictionary
        """
        return {
            "langsmith_enabled": self.enabled,
            "langsmith_available": LANGSMITH_AVAILABLE,
            "project_name": self.project_name,
            "endpoint": config.LANGCHAIN_ENDPOINT,
            "tracing_enabled": config.LANGCHAIN_TRACING_V2,
            "client_initialized": self.client is not None,
            "tracer_initialized": self.tracer is not None
        }