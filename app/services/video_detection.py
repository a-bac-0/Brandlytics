<<<<<<< HEAD
=======
"""
Video Detection Service
Simplified lazy loading implementation for single HuggingFace ModelM.
"""

from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import requests
import logging
from uuid import uuid4

from app.config.model_config import settings
from app.utils.logger import get_logger
from app.utils.video_processor import OptimizedVideoProcessor

logger = get_logger(__name__)

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


class VideoDetectionService:
    """
    Service for video brand detection with simplified lazy loading.
    Optimized for single HuggingFace ModelM (coca cola, nike, starbucks).
    """
    
    _instance = None
    _processor = None
    
    def __new__(cls):
        """Singleton pattern to ensure single instance across the application."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the video detection service with lazy loading."""
        logger.info("VideoDetectionService initialized (ModelM will be loaded on first use)")
    
    def _get_processor(self) -> OptimizedVideoProcessor:
        """
        Get the processor instance, creating it if needed (lazy loading).
        Simplified for single model architecture.
        """
        if self._processor is None:
            logger.info("Loading HuggingFace ModelM and video processor for first time...")
            
            # Simple config for single model
            config = {
                'video_processing': {
                    'fps_sample': getattr(settings, 'VIDEO_FPS_SAMPLE', 1)
                }
            }
            
            try:
                self._processor = OptimizedVideoProcessor(config)
                logger.info("✓ ModelM loaded successfully - ready for 3-brand detection")
            except Exception as e:
                logger.error(f"✗ Failed to load ModelM: {e}")
                raise Exception(f"Could not initialize video processor with ModelM: {e}")
                
        return self._processor

    def get_model_info(self) -> Dict:
        """Get information about the model without triggering lazy loading."""
        if self._processor is None:
            return {
                'model_loaded': False,
                'model_source': 'CV-Brandlytics/ModelM',
                'supported_brands': ['coca cola', 'nike', 'starbucks'],
                'total_brands': 3,
                'lazy_loading': True,
                'status': 'Model will be loaded on first use'
            }
        else:
            return {
                'model_loaded': True,
                'model_source': 'CV-Brandlytics/ModelM',
                'supported_brands': list(self._processor.model.names.values()),
                'total_brands': len(self._processor.model.names),
                'lazy_loading': True,
                'status': 'Model loaded and ready'
            }


# Global singleton instance
video_detection_service = VideoDetectionService()
>>>>>>> 9ea8a448a33101826fda39f0d6f9c6a55d9d899f
