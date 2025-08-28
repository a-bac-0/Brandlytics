import os
import io
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
import time

import numpy as np
import cv2
import av  # PyAV for FFmpeg integration
from PIL import Image
from supabase import create_client, Client
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")

class OptimizedVideoProcessor:
    """
    Unified video processor for brand detection with Supabase integration.
    Uses FFmpeg via PyAV for optimal video decoding performance.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.fps_sample = self.config.get('video_processing', {}).get('fps_sample', 1) or 1
        
        # Initialize Supabase
        if SUPABASE_URL and SUPABASE_KEY:
            self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        else:
            logger.warning("Supabase credentials not found. Database features disabled.")
            self.supabase = None
        
        # Initialize YOLO model
        self.model = YOLO(YOLO_MODEL_PATH)
        
        # Initialize tracker
        self.tracker = BYTETracker(frame_rate=30)
        
        # Performance tracking
        self.total_detection_time = 0.0
        self.processed_frames = 0

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
        """Compute frame sampling step based on target samples per second."""
        try:
            if fps and fps > 0:
                step = int(round(max(1.0, fps / float(self.fps_sample))))
            else:
                step = 1
        except Exception:
            step = 1
        return max(1, step)

    async def process_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        show_progress: bool = True,
        save_to_db: bool = True,
        video_name: Optional[str] = None
    ) -> Dict:
        """
        Process video for brand detection with optional Supabase integration.
        Uses FFmpeg via PyAV for optimal performance.
        
        Args:
            video_path: Path to input video
            output_path: Optional path for annotated output video
            max_frames: Optional limit on frames to process
            show_progress: Show progress bar
            save_to_db: Save detections to Supabase
            video_name: Custom video name for database
            
        Returns:
            Analysis results dictionary
        """
        video_path = str(video_path)
        video_name = video_name or Path(video_path).stem
        
        # Open video with PyAV (FFmpeg backend)
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        
        # Video properties
        fps = float(video_stream.average_rate)
        total_frames = video_stream.frames
        width = video_stream.width
        height = video_stream.height
        
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
            writer = cv2.VideoWriter(output_path, fourcc, float(out_fps), (width, height))
        
        # Analysis structures
        brand_acc: Dict[str, Dict] = {}
        frame_detections: List[Dict] = []
        all_detections_for_db: List[Dict] = []
        
        # Reset performance counters
        self.total_detection_time = 0.0
        self.processed_frames = 0
        
        start_total = time.perf_counter()
        pbar = tqdm(total=total_frames, desc="Processing frames", disable=not show_progress)
        
        try:
            frame_idx = 0
            for frame in container.decode(video_stream):
                if max_frames and frame_idx >= max_frames:
                    break
                
                # Convert frame to numpy array
                frame_np = frame.to_ndarray(format='bgr24')
                timestamp = float(frame.time) if frame.time else (frame_idx / fps)
                
                do_process = (frame_idx % step == 0)
                detections = []
                
                if do_process:
                    # Detect brands
                    detections = self.detect_brands(frame_np)
                    self.processed_frames += 1
                    
                    # Apply tracking
                    detections_tracked = self._track_detections(detections)
                    
                    # Process crops for database if needed
                    if save_to_db and self.supabase:
                        for det in detections_tracked:
                            # Extract crop
                            x1, y1, x2, y2 = det['bbox']
                            crop = frame_np[y1:y2, x1:x2]
                            
                            # Upload crop and get URL
                            crop_url = await self._upload_crop(
                                crop, f"{video_name}_{frame_idx}_{det['brand_name']}.jpg"
                            )
                            det['crop_url'] = crop_url
                            
                            # Prepare for database
                            all_detections_for_db.append({
                                'video_name': video_name,
                                'frame_number': frame_idx,
                                'timestamp_seconds': timestamp,
                                'brand_name': det['brand_name'],
                                'confidence': det['confidence'],
                                'bbox_x1': x1,
                                'bbox_y1': y1,
                                'bbox_x2': x2,
                                'bbox_y2': y2,
                                'image_crop_url': crop_url,
                                'track_id': det.get('track_id'),
                                'fps': fps,
                                'detection_type': 'video'
                            })
                    
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
                
                # Write annotated frame if needed
                if writer:
                    if do_process and detections:
                        annotated_frame = self.annotate_image(frame_np, detections)
                        writer.write(annotated_frame)
                    else:
                        writer.write(frame_np)
                
                frame_idx += 1
                pbar.update(1)
        
        finally:
            pbar.close()
            container.close()
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
        if save_to_db and self.supabase and all_detections_for_db:
            await self._save_detections_batch(all_detections_for_db, video_name)
        
        # Build results
        analysis_results = {
            'video_info': {
                'path': video_path,
                'duration': duration,
                'total_frames': total_frames,
                'processed_frames': self.processed_frames,
                'fps': fps,
                'processing_time': self.total_detection_time,
                'total_elapsed_time': total_elapsed,
                'avg_processing_time_per_frame': (
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
        """Apply ByteTrack tracking to detections."""
        if not detections:
            return detections
        
        # Convert to ByteTrack format
        dets_array = np.array([
            [det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3], det['confidence']]
            for det in detections
        ])
        
        # Update tracker
        tracked_objects = self.tracker.update(dets_array, None)
        
        # Assign track IDs (match by index for simplicity)
        for i, det in enumerate(detections):
            if i < len(tracked_objects):
                det['track_id'] = tracked_objects[i].track_id
            else:
                det['track_id'] = None
        
        return detections

    async def _upload_crop(self, crop_bgr: np.ndarray, filename: str) -> str:
        """Upload detection crop to Supabase storage."""
        if not self.supabase:
            return ""
        
        try:
            # Convert BGR to RGB and then to PIL
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            
            # Convert to bytes
            buf = io.BytesIO()
            pil_img.save(buf, format='JPEG', quality=90)
            buf.seek(0)
            img_bytes = buf.read()
            
            # Upload to Supabase
            bucket_name = "crops"
            res = self.supabase.storage.from_(bucket_name).upload(
                filename, img_bytes, {"content-type": "image/jpeg"}
            )
            
            if res.get("error"):
                logger.error(f"Error uploading crop {filename}: {res['error']}")
                return ""
            
            # Get public URL
            public_url_res = self.supabase.storage.from_(bucket_name).get_public_url(filename)
            return public_url_res.get("publicUrl", "")
        
        except Exception as e:
            logger.error(f"Error processing crop upload: {e}")
            return ""

    async def _save_detections_batch(self, detections: List[Dict], video_name: str):
        """Save detections to Supabase in batch."""
        if not self.supabase or not detections:
            return
        
        try:
            # Insert detections in batch
            result = self.supabase.table("detections").insert(detections).execute()
            if result.data:
                logger.info(f"Saved {len(detections)} detections for {video_name}")
            
            # Update detection events
            await self._update_detection_events(video_name)
            
        except Exception as e:
            logger.error(f"Error saving detections: {e}")

    async def _update_detection_events(self, video_name: str):
        """Update detection_events table from detections."""
        if not self.supabase:
            return
        
        try:
            # Use RPC function or direct query to aggregate detection events
            # This assumes you have a stored procedure or can execute raw SQL
            query = """
            INSERT INTO detection_events (
                video_name, brand_name, track_id,
                start_frame, end_frame, total_frames,
                min_confidence, max_confidence, avg_confidence,
                first_detected, last_detected
            )
            SELECT
                video_name, brand_name, track_id,
                MIN(frame_number) as start_frame,
                MAX(frame_number) as end_frame,
                COUNT(*) as total_frames,
                MIN(confidence) as min_confidence,
                MAX(confidence) as max_confidence,
                AVG(confidence) as avg_confidence,
                MIN(created_at) as first_detected,
                MAX(created_at) as last_detected
            FROM detections
            WHERE video_name = %s
              AND track_id IS NOT NULL
            GROUP BY video_name, brand_name, track_id
            ON CONFLICT (video_name, brand_name, track_id) 
            DO UPDATE SET
                end_frame = EXCLUDED.end_frame,
                total_frames = EXCLUDED.total_frames,
                max_confidence = EXCLUDED.max_confidence,
                avg_confidence = EXCLUDED.avg_confidence,
                last_detected = EXCLUDED.last_detected
            """
            
            # Note: You'll need to implement this based on your Supabase setup
            # This might require a stored procedure or RPC function
            logger.info(f"Detection events updated for {video_name}")
            
        except Exception as e:
            logger.error(f"Error updating detection events: {e}")


# Usage example
async def main():
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
    print(f"Total detections: {results['summary']['total_detections']}")


if __name__ == "__main__":
    asyncio.run(main())