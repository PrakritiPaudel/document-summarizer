"""
File management utilities for the Study Notes Summarizer.
Handles file operations, validation, and storage management.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

from ..config import config

logger = logging.getLogger(__name__)

class FileManager:
    """Manages file operations for the application."""
    
    def __init__(self):
        self.upload_dir = Path(config.UPLOAD_DIRECTORY)
        self.data_dir = Path("./data")
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure required directories exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Ensured directories: {self.upload_dir}, {self.data_dir}")
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> Path:
        """
        Save uploaded file content to disk.
        
        Args:
            file_content: Raw file content
            filename: Original filename
            
        Returns:
            Path to saved file
        """
        try:
            # Generate unique filename to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_hash = hashlib.md5(file_content).hexdigest()[:8]
            
            name, ext = os.path.splitext(filename)
            unique_filename = f"{name}_{timestamp}_{file_hash}{ext}"
            
            file_path = self.upload_dir / unique_filename
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"Saved file: {unique_filename}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving file {filename}: {str(e)}")
            raise
    
    def validate_file_type(self, filename: str) -> bool:
        """
        Validate file type based on extension.
        
        Args:
            filename: Name of the file to validate
            
        Returns:
            True if valid, False otherwise
        """
        valid_extensions = {'.pdf', '.docx', '.txt', '.md'}
        file_ext = Path(filename).suffix.lower()
        return file_ext in valid_extensions
    
    def validate_file_size(self, file_content: bytes) -> bool:
        """
        Validate file size.
        
        Args:
            file_content: Raw file content
            
        Returns:
            True if valid size, False otherwise
        """
        size_mb = len(file_content) / (1024 * 1024)
        return size_mb <= config.MAX_FILE_SIZE_MB
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        try:
            stat = file_path.stat()
            return {
                'name': file_path.name,
                'size': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'created': datetime.fromtimestamp(stat.st_ctime),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'extension': file_path.suffix.lower()
            }
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {str(e)}")
            return {}
    
    def list_uploaded_files(self) -> List[Dict[str, Any]]:
        """
        List all uploaded files with their information.
        
        Returns:
            List of file information dictionaries
        """
        try:
            files = []
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file():
                    info = self.get_file_info(file_path)
                    if info:
                        info['path'] = str(file_path)
                        files.append(info)
            
            # Sort by creation time (newest first)
            files.sort(key=lambda x: x.get('created', datetime.min), reverse=True)
            return files
            
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            return []
    
    def delete_file(self, file_path: Path) -> bool:
        """
        Delete a file.
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted file: {file_path.name}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {str(e)}")
            return False
    
    def clear_upload_directory(self) -> bool:
        """
        Clear all files from the upload directory.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
            
            logger.info("Cleared upload directory")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing upload directory: {str(e)}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage information
        """
        try:
            upload_size = sum(
                f.stat().st_size 
                for f in self.upload_dir.rglob('*') 
                if f.is_file()
            )
            
            data_size = sum(
                f.stat().st_size 
                for f in self.data_dir.rglob('*') 
                if f.is_file()
            )
            
            return {
                'upload_directory': {
                    'path': str(self.upload_dir),
                    'size_bytes': upload_size,
                    'size_mb': upload_size / (1024 * 1024),
                    'file_count': len([f for f in self.upload_dir.iterdir() if f.is_file()])
                },
                'data_directory': {
                    'path': str(self.data_dir),
                    'size_bytes': data_size,
                    'size_mb': data_size / (1024 * 1024)
                },
                'total_size_mb': (upload_size + data_size) / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {str(e)}")
            return {}
    
    def cleanup_old_files(self, days_old: int = 7) -> int:
        """
        Clean up files older than specified days.
        
        Args:
            days_old: Number of days to consider as old
            
        Returns:
            Number of files deleted
        """
        try:
            deleted_count = 0
            cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file():
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted old file: {file_path.name}")
            
            logger.info(f"Cleanup completed: {deleted_count} files deleted")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            return 0