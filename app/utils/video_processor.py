import os
import io
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
import time
from uuid import uuid4

import numpy as np
import cv2
# import av  # PyAV for FFmpeg integration - Commented out due to DLL issues
from PIL import Image
from ultralytics import YOLO
# from yolox.tracker.byte_tracker import BYTETracker  # Commented out due to dependencies
from tqdm import tqdm

from app.database.connection import get_supabase
from app.database.operations import save_detections, upload_public_bytes
from app.api.schemas.schemas_detection import DetectionCreate
from app.utils.image_processing import crop_and_upload_detection
from app.config.model_config import settings

logger = logging.getLogger(__name__)


class OptimizedVideoProcessor:
    """
    Unified video processor for brand detection with Supabase integration.
    Uses FFmpeg via PyAV for optimal video decoding performance.
    Follows the same patterns as the existing image processing pipeline.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.fps_sample = self.config.get('video_processing', {}).get('fps_sample', 1) or 1
        
        # Initialize Supabase (using existing connection pattern)
        self.supabase = get_supabase()
        
        # Initialize YOLO model (using existing settings)
        self.model = YOLO(settings.MODEL_PATH)
        
        # Initialize simple tracking (using ultralytics built-in tracking)
        self.use_tracking = True
        
        # Performance tracking
        self.total_detection_time = 0.0
        self.processed_frames = 0
        
        # Tracking state for detection events
        self.active_tracks: Dict[int, Dict] = {}  # track_id -> track_info

    def detect_brands(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect brands in a single frame using YOLO.
        Compatible with the original VideoProcessor interface.
        """
        start_time = time.perf_counter()
        
        results = self.model(frame)[0]
        detections = []
        
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box.tolist())
            detections.append({
                'brand_name': self.model.names[int(cls)],
                'confidence': float(conf),
                'bbox': (x1, y1, x2, y2),
                'class_id': int(cls)
            })
        
        self.total_detection_time += (time.perf_counter() - start_time)
        return detections

    def annotate_image(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Annotate frame with detection results.
        Compatible with the original VideoProcessor interface.
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            brand_name = det['brand_name']
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{brand_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated

    def _compute_sampling_step(self, fps: float) -> int:
        """Compute frame sampling step based on FPS and config."""
        if fps <= 0:
            return max(1, self.fps_sample)
        return max(1, int(fps / self.fps_sample))

    async def process_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
        show_progress: bool = True,
        save_to_db: bool = False,
        video_name: Optional[str] = None
    ) -> Dict:
        """
        Process video with PyAV for optimal performance and Supabase integration.
        
        Args:
            video_path: Path to input video
            output_path: Path for annotated output video (optional)
            max_frames: Maximum frames to process (for testing)
            show_progress: Show progress bar
            save_to_db: Save detections to Supabase database
            video_name: Name for database entries (defaults to filename)
        
        Returns:
            Dict with analysis results, performance metrics, and detection data
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        if video_name is None:
            video_name = video_path.stem
        
        # Open video with OpenCV
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        step = self._compute_sampling_step(fps)
        
        logger.info(
            f"Processing video: {video_path} | FPS: {fps:.2f} | "
            f"Frames: {total_frames} | Step: {step}"
        )
        
        # Initialize output writer if needed
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_fps = fps if fps > 0 else max(1, self.fps_sample)
            writer = cv2.VideoWriter(str(output_path), fourcc, float(out_fps), (width, height))
        
        # Analysis structures
        brand_acc: Dict[str, Dict] = {}
        frame_detections: List[Dict] = []
        all_detections_for_db: List[DetectionCreate] = []
        
        # Reset performance counters
        self.total_detection_time = 0.0
        self.processed_frames = 0
        
        start_total = time.perf_counter()
        pbar = tqdm(total=total_frames, desc="Processing frames", disable=not show_progress)
        
        try:
            frame_idx = 0
            while True:
                ret, frame_np = cap.read()
                if not ret:
                    break
                    
                if max_frames and frame_idx >= max_frames:
                    break
                
                # Calculate timestamp
                timestamp = frame_idx / fps if fps > 0 else frame_idx
                
                do_process = (frame_idx % step == 0)
                detections = []
                
                if do_process:
                    # Detect brands
                    detections = self.detect_brands(frame_np)
                    self.processed_frames += 1
                    
                    # Apply tracking
                    detections_tracked = self._track_detections(detections)
                    
                    # Process crops for database if needed
                    if save_to_db:
                        for det in detections_tracked:
                            # Extract crop coordinates
                            x1, y1, x2, y2 = det['bbox']
                            
                            # For now, skip crop upload to debug
                            crop_url = None  # TODO: Fix crop_and_upload_detection call
                            det['crop_url'] = crop_url
                            
                            # Prepare DetectionCreate object for database
                            detection_data = DetectionCreate(
                                video_name=video_name,
                                frame_number=frame_idx,
                                timestamp_seconds=timestamp,
                                brand_name=det['brand_name'],
                                confidence=det['confidence'],
                                bbox_x1=x1,
                                bbox_y1=y1,
                                bbox_x2=x2,
                                bbox_y2=y2,
                                image_crop_url=crop_url,
                                track_id=str(uuid4()) if det.get('track_id') else None,
                                fps=fps,
                                detection_type='video'
                            )
                            all_detections_for_db.append(detection_data)
                    
                    # Store frame-level detections
                    frame_info = {
                        'frame_idx': frame_idx,
                        'timestamp': timestamp,
                        'detections': detections_tracked,
                        'detection_count': len(detections_tracked)
                    }
                    frame_detections.append(frame_info)
                    
                    # Aggregate brand statistics
                    for det in detections_tracked:
                        brand = det['brand_name']
                        conf = det['confidence']
                        
                        if brand not in brand_acc:
                            brand_acc[brand] = {
                                'total_frames': 0,
                                'total_time': 0.0,
                                'detections': [],
                                'avg_confidence': 0.0,
                                'max_confidence': 0.0
                            }
                        
                        entry = brand_acc[brand]
                        entry['total_frames'] += 1
                        entry['total_time'] += (1.0 / self.fps_sample)
                        entry['detections'].append(det)
                        entry['max_confidence'] = max(entry['max_confidence'], conf)
                
                # Annotate frame for output video
                if writer:
                    if detections:
                        annotated_frame = self.annotate_image(frame_np, detections)
                    else:
                        annotated_frame = frame_np
                    writer.write(annotated_frame)
                
                frame_idx += 1
                pbar.update(1)
        
        finally:
            pbar.close()
            cap.release()
            if writer:
                writer.release()
        
        # Calculate final statistics
        total_elapsed = time.perf_counter() - start_total
        duration = total_frames / fps if fps > 0 else self.processed_frames / self.fps_sample
        
        # Finalize brand statistics
        for brand_name, entry in brand_acc.items():
            dets = entry.get('detections', [])
            if dets:
                confidences = [d['confidence'] for d in dets]
                entry['avg_confidence'] = float(np.mean(confidences))
            entry['appearance_percentage'] = (
                (entry['total_time'] / duration) * 100.0 if duration > 0 else 0.0
            )
        
        # Save to database if enabled
        if save_to_db and all_detections_for_db:
            try:
                save_detections(all_detections_for_db)
                logger.info(f"Saved {len(all_detections_for_db)} detections to database")
            except Exception as e:
                logger.error(f"Error saving detections to database: {e}")
        
        # Build results
        analysis_results = {
            'performance': {
                'total_elapsed_seconds': total_elapsed,
                'processed_frames': self.processed_frames,
                'fps_processed': self.processed_frames / total_elapsed if total_elapsed > 0 else 0.0,
                'avg_detection_time_per_frame': (
                    self.total_detection_time / self.processed_frames 
                    if self.processed_frames > 0 else 0.0
                )
            },
            'brand_detections': brand_acc,
            'frame_detections': frame_detections,
            'summary': {
                'total_brands_detected': len(brand_acc),
                'total_detections': sum(len(f['detections']) for f in frame_detections),
                'brands_found': list(brand_acc.keys())
            }
        }
        
        logger.info(
            f"Processing complete. Brands: {len(brand_acc)} | "
            f"Processed frames: {self.processed_frames} | "
            f"Elapsed: {total_elapsed:.2f}s"
        )
        
        return analysis_results

    def _track_detections(self, detections: List[Dict]) -> List[Dict]:
        """Apply simple tracking to detections using ultralytics."""
        if not detections:
            return detections
        
        # For now, assign simple sequential track IDs
        # In production, you could use ultralytics built-in tracking
        for i, det in enumerate(detections):
            det['track_id'] = f"track_{i}_{self.processed_frames}"
        
        return detections


# Usage example
async def main():
    """Example usage of OptimizedVideoProcessor."""
    # Initialize processor
    config = {
        'video_processing': {
            'fps_sample': 1  # Process 1 frame per second
        }
    }
    processor = OptimizedVideoProcessor(config)
    
    # Process video
    results = await processor.process_video(
        video_path="path/to/video.mp4",
        output_path="path/to/annotated_output.mp4",
        max_frames=100,  # Limit for testing
        save_to_db=True,
        video_name="test_video_001"
    )
    
    print(f"Detected brands: {results['summary']['brands_found']}")


if __name__ == "__main__":
    asyncio.run(main())
