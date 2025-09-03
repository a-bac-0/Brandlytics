"""
Video Detection Service
Merged and fixed lazy loading implementation for single HuggingFace ModelM.
"""

from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import requests
import logging
from uuid import uuid4
import time

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
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        logger.info("VideoDetectionService initialized (ModelM will be loaded on first use)")
    
    def _get_processor(self) -> OptimizedVideoProcessor:
        if self._processor is None:
            logger.info("Loading HuggingFace ModelM and video processor for first time...")
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

    def _download_video_from_url(self, url: str) -> tuple[bytes, str]:
        if "youtube.com" in url.lower() or "youtu.be" in url.lower():
            if not yt_dlp:
                raise RuntimeError("YouTube support requires 'yt-dlp': pip install yt-dlp")
            with tempfile.TemporaryDirectory() as tmpdir:
                ydl_opts = {
                    "format": "best[ext=mp4]/best",
                    "outtmpl": os.path.join(tmpdir, "%(title)s.%(ext)s"),
                    "quiet": True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    file_path = ydl.prepare_filename(info)
                with open(file_path, "rb") as f:
                    video_bytes = f.read()
                filename = os.path.basename(file_path)
                return video_bytes, filename
        else:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            filename = os.path.basename(url.split("?")[0]) or "video_from_url.mp4"
            return response.content, filename

    async def process_video_input(
        self, 
        video_file_bytes: Optional[bytes] = None, 
        filename: Optional[str] = None, 
        video_url: Optional[str] = None,
        save_to_db: bool = True,
        frame_step: int = 30,
        conf: float = 0.5,
        iou: float = 0.45,
        save_crops: bool = True
    ) -> Dict:
        try:
            if video_url:
                logger.info(f"Downloading video from URL: {video_url}")
                video_file_bytes, filename = self._download_video_from_url(video_url)
            if not video_file_bytes:
                raise ValueError("No video data provided. Supply either video_file_bytes or video_url.")
            if not filename:
                filename = f"video_{uuid4()}.mp4"

            logger.info(f"Processing video: {filename}")
            results = await self._process_video_bytes(
                video_bytes=video_file_bytes,
                filename=filename,
                save_to_db=save_to_db,
                frame_step=frame_step,
                conf=conf,
                iou=iou,
                save_crops=save_crops
            )

            results['metadata'] = {
                'processed_at': datetime.now().isoformat(),
                'video_name': filename,
                'model_source': 'CV-Brandlytics/ModelM',
                'brands_detected': ['coca cola', 'nike', 'starbucks'],
                'frame_step': frame_step,
                'conf_threshold': conf,
                'iou_threshold': iou,
                'service_version': '2.1.0',
                'processing_time_seconds': results.get('processing_time_seconds', 0)
            }
            detection_count = len(results.get('detections_to_save', []))
            logger.info(f"✓ Video processing completed. Detections: {detection_count}")
            return results
        except Exception as e:
            logger.error(f"✗ Video processing failed: {str(e)}")
            raise

    async def _process_video_bytes(
        self, 
        video_bytes: bytes, 
        filename: str,
        save_to_db: bool,
        frame_step: int,
        conf: float,
        iou: float,
        save_crops: bool
    ) -> Dict:
        temp_path = None
        start_time = time.time()
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_file.write(video_bytes)
                temp_path = temp_file.name
            processor = self._get_processor()
            results = await processor.process_video(
                video_path=temp_path,
                video_name=filename,
                save_to_db=save_to_db,
                max_frames=None,
                output_path=None,
                show_progress=True
            )
            results['processing_time_seconds'] = time.time() - start_time
            return results
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    logger.info("Cleaned up temporary video file")
                except Exception as e:
                    logger.warning(f"Could not delete temporary file {temp_path}: {e}")

    async def process_video_batch(
        self,
        video_paths: List[Path],
        save_to_db: bool = True,
        frame_step: int = 30,
        conf: float = 0.5,
        iou: float = 0.45,
    ) -> Dict[str, Dict]:
        results = {}
        for video_path in video_paths:
            try:
                video_name = Path(video_path).stem
                logger.info(f"Processing video in batch: {video_name}")
                result = await self.process_video_input(
                    video_file_bytes=open(video_path, 'rb').read(),
                    filename=Path(video_path).name,
                    save_to_db=save_to_db,
                    frame_step=frame_step,
                    conf=conf,
                    iou=iou
                )
                results[video_name] = result
            except Exception as e:
                logger.error(f"Batch processing failed for {video_path}: {e}")
                results[video_path] = {"error": str(e)}
        return results

    def get_model_info(self) -> Dict:
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


video_detection_service = VideoDetectionService()
