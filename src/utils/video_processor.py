import cv2
import numpy as np
from typing import List, Dict, Generator, Tuple
from pathlib import Path
import logging
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Advanced video processing for brand detection
    """
    
    def __init__(self, detector, config: Dict):
        self.detector = detector
        self.config = config
        self.fps_sample = config.get('video_processing', {}).get('fps_sample', 1)
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Process entire video for brand detection
        
        Args:
            video_path: Path to input video
            output_path: Optional path for output video
            
        Returns:
            Analysis results dictionary
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        
        # Sample frames based on fps_sample
        frame_interval = max(1, int(fps // self.fps_sample))
        
        # Initialize video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Analysis variables
        brand_detections = {}
        frame_detections = []
        total_detection_time = 0
        processed_frames = 0
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Video properties - FPS: {fps}, Frames: {total_frames}, Duration: {duration:.2f}s")
        
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame based on sampling rate
                if frame_idx % frame_interval == 0:
                    start_time = time.time()
                    detections = self.detector.detect_brands(frame)
                    detection_time = time.time() - start_time
                    total_detection_time += detection_time
                    processed_frames += 1
                    
                    # Store frame detections
                    frame_info = {
                        'frame_idx': frame_idx,
                        'timestamp': frame_idx / fps,
                        'detections': detections,
                        'detection_count': len(detections)
                    }
                    frame_detections.append(frame_info)
                    
                    # Count brand occurrences
                    for detection in detections:
                        brand_name = detection['brand_name']
                        if brand_name not in brand_detections:
                            brand_detections[brand_name] = {
                                'total_frames': 0,
                                'total_time': 0,
                                'detections': [],
                                'avg_confidence': 0,
                                'max_confidence': 0
                            }
                        
                        brand_data = brand_detections[brand_name]
                        brand_data['total_frames'] += 1
                        brand_data['total_time'] += (1 / self.fps_sample)
                        brand_data['detections'].append(detection)
                        brand_data['max_confidence'] = max(
                            brand_data['max_confidence'], 
                            detection['confidence']
                        )
                    
                    # Annotate frame for output video
                    if writer:
                        annotated_frame = self.detector.annotate_image(frame, detections)
                        writer.write(annotated_frame)
                elif writer:
                    # Write original frame if not processing
                    writer.write(frame)
                
                frame_idx += 1
                pbar.update(1)
        
        # Clean up
        cap.release()
        if writer:
            writer.release()
        
        # Calculate final statistics
        for brand_name, brand_data in brand_detections.items():
            if brand_data['detections']:
                confidences = [d['confidence'] for d in brand_data['detections']]
                brand_data['avg_confidence'] = np.mean(confidences)
                brand_data['appearance_percentage'] = (brand_data['total_time'] / duration) * 100
        
        analysis_results = {
            'video_info': {
                'path': video_path,
                'duration': duration,
                'total_frames': total_frames,
                'processed_frames': processed_frames,
                'fps': fps,
                'processing_time': total_detection_time,
                'avg_processing_time_per_frame': total_detection_time / max(processed_frames, 1)
            },
            'brand_detections': brand_detections,
            'frame_detections': frame_detections,
            'summary': {
                'total_brands_detected': len(brand_detections),
                'total_detections': sum(len(frame['detections']) for frame in frame_detections),
                'brands_found': list(brand_detections.keys())
            }
        }
        
        logger.info(f"Video processing complete. Found {len(brand_detections)} brands in {processed_frames} frames")
        
        return analysis_results