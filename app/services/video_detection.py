"""
Video Detection Service
Handles video processing for brand detection following the same patterns as ImageDetectionService.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

from app.utils.video_processor import OptimizedVideoProcessor
from app.database.operations import save_detections
from app.api.schemas.schemas_detection import DetectionCreate
from app.config.model_config import settings

logger = logging.getLogger(__name__)


class VideoDetectionService:
    """
    Service for video brand detection.
    Follows the same patterns as ImageDetectionService for consistency.
    """
    
    _instance = None
    _processor = None
    
    def __new__(cls):
        """Singleton pattern to ensure single instance across the application."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the video detection service."""
        if self._processor is None:
            # Initialize with default config
            config = {
                'video_processing': {
                    'fps_sample': getattr(settings, 'VIDEO_FPS_SAMPLE', 1)
                }
            }
            self._processor = OptimizedVideoProcessor(config)
            logger.info("VideoDetectionService initialized")
    
    async def process_video_file(
        self,
        video_path: Union[str, Path],
        video_name: Optional[str] = None,
        save_to_db: bool = True,
        max_frames: Optional[int] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Process a video file for brand detection.
        
        Args:
            video_path: Path to the video file
            video_name: Name for database entries (defaults to filename)
            save_to_db: Whether to save detections to database
            max_frames: Maximum frames to process (for testing)
            output_path: Path for annotated output video
        
        Returns:
            Dictionary with processing results and statistics
        """
        try:
            logger.info(f"Starting video processing for: {video_path}")
            
            # Process the video
            results = await self._processor.process_video(
                video_path=video_path,
                video_name=video_name,
                save_to_db=save_to_db,
                max_frames=max_frames,
                output_path=output_path,
                show_progress=True
            )
            
            # Add metadata
            results['metadata'] = {
                'processed_at': datetime.now().isoformat(),
                'video_name': video_name or Path(video_path).stem,
                'service_version': '1.0.0'
            }
            
            logger.info(f"Video processing completed successfully. Brands found: {results['summary']['brands_found']}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            raise
    
    async def process_video_batch(
        self,
        video_paths: List[Union[str, Path]],
        save_to_db: bool = True,
        max_frames: Optional[int] = None
    ) -> Dict[str, Dict]:
        """
        Process multiple video files in batch.
        
        Args:
            video_paths: List of video file paths
            save_to_db: Whether to save detections to database
            max_frames: Maximum frames to process per video
        
        Returns:
            Dictionary mapping video names to their processing results
        """
        results = {}
        
        for video_path in video_paths:
            try:
                video_name = Path(video_path).stem
                result = await self.process_video_file(
                    video_path=video_path,
                    video_name=video_name,
                    save_to_db=save_to_db,
                    max_frames=max_frames
                )
                results[video_name] = result
                
            except Exception as e:
                logger.error(f"Failed to process video {video_path}: {str(e)}")
                results[str(video_path)] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        return results
    
    def get_processor_stats(self) -> Dict:
        """Get current processor statistics."""
        return {
            'total_detection_time': self._processor.total_detection_time,
            'processed_frames': self._processor.processed_frames,
            'active_tracks': len(self._processor.active_tracks),
            'model_path': settings.MODEL_PATH
        }


# Singleton instance
video_detection_service = VideoDetectionService()
