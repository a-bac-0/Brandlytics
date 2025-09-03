"""
Video Detection Service
Handles video processing for brand detection following the same patterns as ImageDetectionService.
"""

from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime
import cv2
import requests
import logging
from uuid import uuid4

from app.utils.image_processing import crop_and_upload_detection
from app.services.image_detection import ImageDetectionService
from app.api.schemas.schemas_detection import DetectionCreate
from app.config.model_config import settings
from app.database.operations import save_detections
from app.utils.logger import get_logger
import time


logger = get_logger(__name__)

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


class VideoDetectionService:
    """
    Service for video brand detection.
    Implements Singleton pattern and reuses ImageDetectionService for consistency.
    """
    
    _instance = None
    _image_service = None
    
    def __new__(cls):
        """Singleton pattern to ensure single instance across the application."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the video detection service."""
        if self._image_service is None:
            # Initialize with ImageDetectionService for consistency
            self._image_service = ImageDetectionService()
            self.model = self._image_service.model
            self.class_names = getattr(self.model, "names", {})
            
            # Config
            self.fps_sample = getattr(settings, 'VIDEO_FPS_SAMPLE', 1)
            logger.info("VideoDetectionService initialized")
    
    def _get_video_bytes_from_url(self, url: str) -> tuple[bytes, str]:
        """
        Download video from URL (including YouTube support)
        
        Args:
            url: URL to download from
            
        Returns:
            Tuple of (video_bytes, filename)
        """
        url_lower = url.lower()
        if "youtube.com" in url_lower or "youtu.be" in url_lower:
            if not yt_dlp:
                raise RuntimeError("Para URLs de YouTube, instala 'yt-dlp': pip install yt-dlp")
            
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
        video_file_bytes: Optional[bytes], 
        filename: Optional[str], 
        video_url: Optional[str],
        save_to_db: bool = True,
        frame_step: int = 30,  # Default to analyze every 30 frames (~1 frame/sec for 30fps video)
        conf: float = 0.25,
        iou: float = 0.6,
        save_crops: bool = True,
    ) -> Dict:
        """
        Process video input from either file bytes or URL
        
        Args:
            video_file_bytes: Raw video bytes
            filename: Video filename
            video_url: URL to video
            save_to_db: Whether to save detections to database
            frame_step: Number of frames to skip between processing
            conf: Confidence threshold for detections
            iou: IOU threshold for NMS
            save_crops: Whether to save crops of detections
            
        Returns:
            Dictionary with processing results and statistics
        """
        try:
            if video_url:
                logger.info(f"Downloading video from URL: {video_url}")
                video_file_bytes, filename = self._get_video_bytes_from_url(video_url)
            
            if not video_file_bytes:
                raise ValueError("No se proporcionaron datos de vídeo.")
            
            if not filename:
                filename = f"video_{uuid4()}.mp4"
            
            logger.info(f"Processing video: {filename}")
            
            # Process the frames
            results = await self._process_frames(
                video_bytes=video_file_bytes,
                filename=filename,
                frame_step=frame_step,
                conf=conf,
                iou=iou,
                save_crops=save_crops,
                save_to_db=save_to_db
            )
            
            # Add metadata
            results['metadata'] = {
                'processed_at': datetime.now().isoformat(),
                'video_name': filename,
                'service_version': '1.0.0',
                'frame_step': frame_step,
                'conf_threshold': conf,
                'iou_threshold': iou,
                'processing_time_seconds': results.get('processing_time_seconds', 0)
            }
            
            logger.info(f"Video processing completed successfully. Detections: {len(results.get('detections_to_save', []))}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise
    
    async def _process_frames(
        self, 
        video_bytes: bytes, 
        filename: str, 
        frame_step: int, 
        conf: float, 
        iou: float, 
        save_crops: bool,
        save_to_db: bool
    ) -> Dict:
        """
        Process video frames for brand detection
        
        Args:
            video_bytes: Raw video bytes
            filename: Video filename
            frame_step: Number of frames to skip between processing
            conf: Confidence threshold for detections
            iou: IOU threshold for NMS
            save_crops: Whether to save crops of detections
            save_to_db: Whether to save detections to database
            
        Returns:
            Dictionary with processing results
        """
        processing_start = time.time()
        suffix = Path(filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir el vídeo.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        detections_to_save: List[DetectionCreate] = []
        frame_idx = 0
        
        # Track statistics
        brand_counts = {}
        brand_timestamps = {}
        frames_with_brands = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step == 0:
                    # Using the image detection model for consistency
                    results = self._image_service.detect_brands(frame, conf=conf, iou=iou)
                    
                    # If brands were detected in this frame
                    if results:
                        frames_with_brands += 1
                    
                    for det in results:
                        brand_name = det["class_name"]
                        
                        # Update statistics
                        brand_counts[brand_name] = brand_counts.get(brand_name, 0) + 1
                        
                        # Track timestamp for each brand
                        if brand_name not in brand_timestamps:
                            brand_timestamps[brand_name] = []
                        brand_timestamps[brand_name].append(round(frame_idx / fps, 3))
                        
                        crop_url = None
                        if save_crops:
                            crop_url = crop_and_upload_detection(frame, det, f"videos/{Path(filename).stem}")

                        detections_to_save.append(DetectionCreate(
                            video_name=filename,
                            frame_number=frame_idx,
                            timestamp_seconds=round(frame_idx / fps, 3),
                            brand_name=brand_name,
                            confidence=det["confidence"],
                            bbox_x1=det["box"][0], bbox_y1=det["box"][1],
                            bbox_x2=det["box"][2], bbox_y2=det["box"][3],
                            image_crop_url=crop_url,
                            fps=fps,
                            detection_type="video"
                        ))
                frame_idx += 1
        finally:
            cap.release()
            os.unlink(tmp_path)
        
        # Save to database if requested
        if save_to_db and detections_to_save:
            logger.info(f"Saving {len(detections_to_save)} detections to database")
            try:
                save_detections(detections_to_save)
            except Exception as e:
                logger.error(f"Error saving detections to database: {e}")
        
        # Prepare analysis summary
        duration_seconds = total_frames / fps if fps > 0 else 0
        brand_screen_time = {
            brand: len(timestamps) * frame_step / fps 
            for brand, timestamps in brand_timestamps.items()
        }
        brand_confidence = {}
        for brand_name in brand_counts.keys():
            brand_detections = [d for d in detections_to_save if d.brand_name == brand_name]
            confidences = [d.confidence for d in brand_detections]
            brand_confidence[brand_name] = {
                "avg": sum(confidences) / len(confidences) if confidences else 0,
                "max": max(confidences) if confidences else 0
            }
        # Create summary
        summary = {
            "filename": filename,
            "duration_seconds": duration_seconds,
            "fps": fps,
            "total_frames": total_frames,
            "processed_frames": frame_idx // frame_step,
            "frames_with_brands": frames_with_brands,
            "brands_found": list(brand_counts.keys()),
            "brand_counts": brand_counts,
            "brand_screen_time": brand_screen_time,
            "brand_confidence": brand_confidence
        }
        processing_end = time.time()
        processing_time_seconds = processing_end - processing_start
        return {
            "summary": summary,
            "detections": [d.model_dump() for d in detections_to_save],
            "detections_to_save": detections_to_save,
            "processing_time_seconds": processing_time_seconds
        }
    
    async def process_video_batch(
        self,
        video_paths: List[Union[str, Path]],
        save_to_db: bool = True,
        frame_step: int = 30,
        conf: float = 0.25,
        iou: float = 0.6,
    ) -> Dict[str, Dict]:
        """
        Process multiple video files in batch.
        
        Args:
            video_paths: List of video file paths
            save_to_db: Whether to save detections to database
            frame_step: Number of frames to skip between processing
            conf: Confidence threshold for detections
            iou: IOU threshold for NMS
            
        Returns:
            Dictionary mapping video names to their processing results
        """
        results = {}
        
        for video_path in video_paths:
            try:
                video_name = Path(video_path).stem
                logger.info(f"Processing video in batch: {video_name}")
                
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                
                result = await self._process_frames(
                    video_bytes=video_bytes,
                    filename=str(Path(video_path).name),
                    frame_step=frame_step,
                    conf=conf,
                    iou=iou,
                    save_crops=True,
                    save_to_db=save_to_db
                )
                
                results[video_name] = result
                logger.info(f"Completed processing video: {video_name}")
                
            except Exception as e:
                logger.error(f"Error processing video {video_path}: {str(e)}")
                results[str(video_path)] = {"error": str(e)}
        
        return results


# Singleton instance
video_detection_service = VideoDetectionService()