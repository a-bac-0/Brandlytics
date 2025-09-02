import cv2
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from ultralytics import YOLO

from app.config.model_config import settings
from app.database.operations import save_detections
from app.api.schemas.schemas_detection import DetectionCreate
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from app.services.analytics_service import AnalyticsService
    ANALYTICS_SERVICE_AVAILABLE = True
    logger.info("✅ Team's AnalyticsService available for enhanced reporting")
except ImportError:
    ANALYTICS_SERVICE_AVAILABLE = False
    logger.warning("Team's AnalyticsService not available - using basic analytics only")


class VideoAnalysisService:
    """
    Professional video analysis service for brand detection.
    Handles any video format and provides comprehensive analytics.
    """
    
    def __init__(self, model_path: str = None):
        """Initialize the video analysis service."""
        self.model_path = model_path or settings.MODEL_PATH
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the YOLO model for brand detection."""
        try:
            logger.info(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("✅ YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def get_video_info(self, video_path: str) -> Dict:
        """Extract basic video information."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        file_size = Path(video_path).stat().st_size / (1024 * 1024)
        
        cap.release()
        
        return {
            'path': video_path,
            'fps': fps,
            'total_frames': total_frames,
            'duration_seconds': duration,
            'file_size_mb': file_size
        }
    
    def calculate_optimal_sampling(self, video_info: Dict, max_frames: Optional[int] = None) -> Tuple[int, int]:
        """
        Calculate optimal frame sampling strategy based on video characteristics.
        
        Returns:
            Tuple[sampling_interval, frames_to_process]
        """
        total_frames = video_info['total_frames']
        duration = video_info['duration_seconds']
        
        if max_frames:
            sampling_interval = max(1, total_frames // max_frames)
            frames_to_process = min(max_frames, total_frames)
        elif duration > 300:
            sampling_interval = 15
            frames_to_process = total_frames // sampling_interval
        elif duration > 60:
            sampling_interval = 10
            frames_to_process = total_frames // sampling_interval
        else:
            sampling_interval = 5
            frames_to_process = total_frames // sampling_interval
        
        logger.info(f"Sampling strategy: every {sampling_interval} frames, processing {frames_to_process} frames")
        return sampling_interval, frames_to_process
    
    def process_video(
        self,
        video_path: str,
        video_name: str,
        save_to_database: bool = True,
        max_frames: Optional[int] = None,
        confidence_threshold: float = 0.5
    ) -> Dict:
        """
        Process a video for brand detection with comprehensive analytics.
        
        Args:
            video_path: Path to the video file
            video_name: Unique identifier for the video
            save_to_database: Whether to save detections to Supabase
            max_frames: Maximum number of frames to process (None for auto)
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            Comprehensive analysis results dictionary
        """
        start_time = time.time()
        
        video_info = self.get_video_info(video_path)
        logger.info(f"🎬 Processing video: {video_name}")
        logger.info(f"📊 Duration: {video_info['duration_seconds']:.1f}s, "
                   f"Size: {video_info['file_size_mb']:.1f}MB, "
                   f"FPS: {video_info['fps']:.1f}")
        
        sampling_interval, frames_to_process = self.calculate_optimal_sampling(video_info, max_frames)
        
        detections_list = []
        detections_for_db = []
        processed_frames = 0
        total_detection_time = 0
        brands_detected = set()
        frame_stats = {'with_detections': 0, 'without_detections': 0}
        
        cap = cv2.VideoCapture(video_path)
        
        try:
            frame_idx = 0
            while processed_frames < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sampling_interval == 0:
                    timestamp = frame_idx / video_info['fps']
                    
                    detection_start = time.time()
                    results = self.model(frame, verbose=False)
                    detection_time = time.time() - detection_start
                    total_detection_time += detection_time
                    
                    frame_detections = self._process_frame_detections(
                        results, frame_idx, timestamp, video_name, confidence_threshold
                    )
                    
                    if frame_detections:
                        frame_stats['with_detections'] += 1
                        detections_list.extend(frame_detections)
                        
                        for det in frame_detections:
                            brands_detected.add(det['brand_name'])
                        
                        if save_to_database:
                            db_detections = self._prepare_detections_for_db(frame_detections)
                            detections_for_db.extend(db_detections)
                        
                        logger.debug(f"Frame {frame_idx} ({timestamp:.1f}s): {len(frame_detections)} detections")
                    else:
                        frame_stats['without_detections'] += 1
                    
                    processed_frames += 1
                
                frame_idx += 1
            
        finally:
            cap.release()
        
        detections_saved = 0
        if save_to_database and detections_for_db:
            detections_saved = self._save_detections_batch(detections_for_db)
        
        processing_time = time.time() - start_time
        results = self._compile_results(
            video_name, video_info, detections_list, brands_detected,
            processed_frames, total_detection_time, processing_time,
            detections_saved, frame_stats, sampling_interval
        )
        
        logger.info(f"✅ Video processing completed: {video_name}")
        logger.info(f"📊 {results['summary']['total_detections']} detections, "
                   f"{len(results['summary']['unique_brands'])} brands, "
                   f"{processing_time:.2f}s processing time")
        
        return results
    
    def _process_frame_detections(
        self, 
        results, 
        frame_idx: int, 
        timestamp: float, 
        video_name: str, 
        confidence_threshold: float
    ) -> List[Dict]:
        """Process YOLO detection results for a single frame."""
        frame_detections = []
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf >= confidence_threshold:
                        cls_id = int(box.cls[0])
                        brand_name = self.model.names.get(cls_id, f"class_{cls_id}")
                        
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        detection = {
                            'video_name': video_name,
                            'frame_number': frame_idx,
                            'timestamp_seconds': timestamp,
                            'brand_name': brand_name,
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'bbox_area': (int(x2) - int(x1)) * (int(y2) - int(y1))
                        }
                        
                        frame_detections.append(detection)
        
        return frame_detections
    
    def _prepare_detections_for_db(self, detections: List[Dict]) -> List[DetectionCreate]:
        """Convert detection dictionaries to database schema objects."""
        db_detections = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            db_detection = DetectionCreate(
                video_name=det['video_name'],
                frame_number=det['frame_number'],
                timestamp_seconds=det['timestamp_seconds'],
                brand_name=det['brand_name'],
                confidence=det['confidence'],
                bbox_x1=x1,
                bbox_y1=y1,
                bbox_x2=x2,
                bbox_y2=y2,
                crop_url=None
            )
            
            db_detections.append(db_detection)
        
        return db_detections
    
    def _save_detections_batch(self, detections: List[DetectionCreate], batch_size: int = 50) -> int:
        """Save detections to database in batches."""
        total_saved = 0
        
        for i in range(0, len(detections), batch_size):
            batch = detections[i:i + batch_size]
            try:
                save_detections(batch)
                total_saved += len(batch)
                logger.debug(f"💾 Saved batch of {len(batch)} detections to database")
            except Exception as e:
                logger.error(f"❌ Failed to save batch: {e}")
        
        return total_saved
    
    def _compile_results(
        self,
        video_name: str,
        video_info: Dict,
        detections: List[Dict],
        brands_detected: set,
        processed_frames: int,
        total_detection_time: float,
        processing_time: float,
        detections_saved: int,
        frame_stats: Dict,
        sampling_interval: int
    ) -> Dict:
        """Compile comprehensive analysis results."""
        
        # Brand analysis
        brand_analysis = {}
        for det in detections:
            brand = det['brand_name']
            if brand not in brand_analysis:
                brand_analysis[brand] = {
                    'count': 0,
                    'avg_confidence': 0,
                    'confidence_sum': 0,
                    'timestamps': [],
                    'avg_bbox_area': 0,
                    'bbox_areas': []
                }
            
            brand_analysis[brand]['count'] += 1
            brand_analysis[brand]['confidence_sum'] += det['confidence']
            brand_analysis[brand]['timestamps'].append(det['timestamp_seconds'])
            brand_analysis[brand]['bbox_areas'].append(det['bbox_area'])
        
        for brand, stats in brand_analysis.items():
            stats['avg_confidence'] = stats['confidence_sum'] / stats['count']
            stats['avg_bbox_area'] = sum(stats['bbox_areas']) / len(stats['bbox_areas'])
            if len(stats['timestamps']) > 1:
                stats['duration_seconds'] = max(stats['timestamps']) - min(stats['timestamps'])
                stats['first_appearance'] = min(stats['timestamps'])
                stats['last_appearance'] = max(stats['timestamps'])
            else:
                stats['duration_seconds'] = 0
                stats['first_appearance'] = stats['timestamps'][0] if stats['timestamps'] else 0
                stats['last_appearance'] = stats['timestamps'][0] if stats['timestamps'] else 0
            
            del stats['confidence_sum']
            del stats['bbox_areas']
        
        time_distribution = {}
        for det in detections:
            time_segment = int(det['timestamp_seconds'] // 10) * 10
            time_distribution[time_segment] = time_distribution.get(time_segment, 0) + 1
        
        enhanced_analytics = None
        if ANALYTICS_SERVICE_AVAILABLE and detections:
            try:
                analytics_service = AnalyticsService()
                enhanced_analytics = analytics_service.calculate_video_summary(detections, video_name)
                logger.info(f"✅ Enhanced analytics calculated: screen time analysis for {len(enhanced_analytics.get('analysis_summary', []))} brands")
            except Exception as e:
                logger.warning(f"Could not calculate enhanced analytics: {e}")
        
        base_results = {
            'video_info': video_info,
            'processing_info': {
                'video_name': video_name,
                'processed_frames': processed_frames,
                'total_frames_in_video': video_info['total_frames'],
                'sampling_interval': sampling_interval,
                'coverage_percentage': (processed_frames * sampling_interval / video_info['total_frames']) * 100,
                'total_detection_time': total_detection_time,
                'total_processing_time': processing_time,
                'avg_detection_time_per_frame': total_detection_time / processed_frames if processed_frames > 0 else 0,
                'detections_per_second': len(detections) / total_detection_time if total_detection_time > 0 else 0,
                'frames_with_detections': frame_stats['with_detections'],
                'frames_without_detections': frame_stats['without_detections']
            },
            'summary': {
                'total_detections': len(detections),
                'unique_brands': list(brands_detected),
                'brands_count': len(brands_detected),
                'detections_saved_to_db': detections_saved,
                'avg_confidence': sum(det['confidence'] for det in detections) / len(detections) if detections else 0
            },
            'brand_analysis': brand_analysis,
            'time_distribution': time_distribution,
            'sample_detections': detections[:10]
        }
        
        if enhanced_analytics:
            base_results['enhanced_analytics'] = enhanced_analytics
        
        return base_results
    
    def get_existing_video_analytics(self, video_name: str) -> Dict:
        """
        Get enhanced analytics for a previously processed video using team's database functions.
        Integrates with team's get_video_detections_byname and get_video_analytics functions.
        """
        try:
            from app.database.operations import get_video_detections_byname, get_video_analytics
            
            basic_analytics = get_video_analytics(video_name)
            
            detections = get_video_detections_byname(video_name)
            
            # Add enhanced analytics if available and we have detection data
            if ANALYTICS_SERVICE_AVAILABLE and detections:
                try:
                    analytics_service = AnalyticsService()
                    enhanced_analytics = analytics_service.calculate_video_summary(detections, video_name)
                    basic_analytics['enhanced_analytics'] = enhanced_analytics
                    logger.info(f"✅ Retrieved enhanced analytics for {video_name}: screen time data for {len(enhanced_analytics.get('analysis_summary', []))} brands")
                except Exception as e:
                    logger.warning(f"Could not calculate enhanced analytics for existing video: {e}")
            
            return basic_analytics
            
        except Exception as e:
            logger.error(f"❌ Failed to get existing video analytics: {e}")
            return {"error": f"Failed to retrieve analytics: {str(e)}"}


def analyze_video(
    video_path: str,
    video_name: str = None,
    save_to_database: bool = True,
    max_frames: Optional[int] = None
) -> Dict:
    """
    Convenience function to analyze any video.
    
    Args:
        video_path: Path to video file
        video_name: Unique name (auto-generated if None)
        save_to_database: Save results to Supabase
        max_frames: Limit frames processed
    
    Returns:
        Analysis results dictionary
    """
    if video_name is None:
        video_name = Path(video_path).stem + "_analysis"
    
    service = VideoAnalysisService()
    return service.process_video(video_path, video_name, save_to_database, max_frames)


def analyze_video_folder(
    folder_path: str,
    video_extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov', '.mkv'),
    save_to_database: bool = True
) -> Dict[str, Dict]:
    """
    Analyze all videos in a folder.
    
    Args:
        folder_path: Path to folder containing videos
        video_extensions: Supported video file extensions
        save_to_database: Save results to Supabase
    
    Returns:
        Dictionary with results for each video
    """
    folder = Path(folder_path)
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(folder.glob(f"*{ext}"))
    
    results = {}
    service = VideoAnalysisService()
    
    for video_file in video_files:
        video_name = video_file.stem + "_analysis"
        logger.info(f"🎬 Processing: {video_file.name}")
        
        try:
            results[video_file.name] = service.process_video(
                str(video_file), video_name, save_to_database
            )
        except Exception as e:
            logger.error(f"❌ Failed to process {video_file.name}: {e}")
            results[video_file.name] = {'error': str(e)}
    
    return results


def get_video_analytics_enhanced(video_name: str) -> Dict:
    """
    Convenience function to get enhanced analytics for a processed video.
    Combines basic database analytics with team's screen time analysis.
    
    Args:
        video_name: Name of the processed video
        
    Returns:
        Enhanced analytics dictionary including screen time data
    """
    service = VideoAnalysisService()
    return service.get_existing_video_analytics(video_name)
