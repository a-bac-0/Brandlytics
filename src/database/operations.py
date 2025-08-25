import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
import cv2
import numpy as np
from pathlib import Path

from .connection import DatabaseManager
from .models import Video, Brand, Detection, AnalysisReport

logger = logging.getLogger(__name__)

class DetectionOperations:
    """
    Database operations for brand detection data
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._initialize_brands()
    
    def _initialize_brands(self):
        """Initialize brand data in database"""
        try:
            with self.db_manager.get_session() as session:
                # Check if brands exist
                existing_brands = session.query(Brand).count()
                
                if existing_brands == 0:
                    # Add default brands from config
                    brands_config = self.db_manager.config.get('brands', [])
                    for brand_config in brands_config:
                        brand = Brand(
                            id=brand_config['id'],
                            name=brand_config['name'],
                            color_r=brand_config['color'][0],
                            color_g=brand_config['color'][1],
                            color_b=brand_config['color'][2]
                        )
                        session.add(brand)
                    
                    session.commit()
                    logger.info(f"Initialized {len(brands_config)} brands in database")
                
        except Exception as e:
            logger.error(f"Failed to initialize brands: {e}")
    
    async def save_video_analysis(self, analysis_results: Dict, video_path: str, filename: str) -> int:
        """
        Save complete video analysis to database
        
        Args:
            analysis_results: Analysis results from video processor
            video_path: Path to video file
            filename: Original filename
            
        Returns:
            Video ID
        """
        try:
            with self.db_manager.get_session() as session:
                # Create video record
                video_info = analysis_results['video_info']
                video_record = Video(
                    filename=filename,
                    filepath=video_path,
                    duration=video_info['duration'],
                    fps=video_info['fps'],
                    total_frames=video_info['total_frames'],
                    processing_time=video_info['processing_time'],
                    file_size=Path(video_path).stat().st_size if Path(video_path).exists() else None
                )
                
                session.add(video_record)
                session.flush()  # Get video ID
                
                video_id = video_record.id
                
                # Save detections
                detection_count = 0
                for frame_data in analysis_results['frame_detections']:
                    for detection in frame_data['detections']:
                        # Get or create brand
                        brand = session.query(Brand).filter_by(
                            name=detection['brand_name']
                        ).first()
                        
                        if not brand:
                            continue  # Skip if brand not in database
                        
                        # Create detection record
                        bbox = detection['bbox']
                        detection_record = Detection(
                            video_id=video_id,
                            brand_id=brand.id,
                            frame_idx=frame_data['frame_idx'],
                            timestamp=frame_data['timestamp'],
                            confidence=detection['confidence'],
                            bbox_x1=int(bbox[0]),
                            bbox_y1=int(bbox[1]),
                            bbox_x2=int(bbox[2]),
                            bbox_y2=int(bbox[3])
                        )
                        
                        # Optionally save cropped image
                        # cropped_image = self._extract_cropped_image(video_path, frame_data['frame_idx'], bbox)
                        # if cropped_image is not None:
                        #     detection_record.cropped_image = cropped_image
                        
                        session.add(detection_record)
                        detection_count += 1
                
                # Create analysis report
                report_data = {
                    'summary': analysis_results['summary'],
                    'brand_analysis': analysis_results['brand_detections'],
                    'processing_stats': {
                        'processed_frames': video_info['processed_frames'],
                        'avg_processing_time_per_frame': video_info['avg_processing_time_per_frame']
                    }
                }
                
                analysis_report = AnalysisReport(
                    video_id=video_id,
                    total_brands_detected=len(analysis_results['brand_detections']),
                    total_detections=detection_count,
                    report_data=json.dumps(report_data)
                )
                
                session.add(analysis_report)
                session.commit()
                
                logger.info(f"Saved video analysis: {filename} (ID: {video_id}) with {detection_count} detections")
                return video_id
                
        except Exception as e:
            logger.error(f"Failed to save video analysis: {e}")
            raise
    
    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict]:
        """Get analysis results by ID"""
        try:
            with self.db_manager.get_session() as session:
                # Try to find by video ID first
                try:
                    video_id = int(analysis_id)
                except ValueError:
                    return None
                
                video = session.query(Video).filter_by(id=video_id).first()
                if not video:
                    return None
                
                # Get analysis report
                report = session.query(AnalysisReport).filter_by(video_id=video_id).first()
                if not report:
                    return None
                
                # Get detections
                detections = session.query(Detection).filter_by(video_id=video_id).all()
                
                # Format response
                result = {
                    'video_info': {
                        'id': video.id,
                        'filename': video.filename,
                        'duration': video.duration,
                        'fps': video.fps,
                        'total_frames': video.total_frames,
                        'processed_at': video.processed_at.isoformat(),
                        'processing_time': video.processing_time
                    },
                    'analysis_summary': {
                        'total_brands_detected': report.total_brands_detected,
                        'total_detections': report.total_detections,
                        'generated_at': report.generated_at.isoformat()
                    },
                    'detections': []
                }
                
                # Add detection details
                for detection in detections:
                    result['detections'].append({
                        'brand_name': detection.brand.name,
                        'frame_idx': detection.frame_idx,
                        'timestamp': detection.timestamp,
                        'confidence': detection.confidence,
                        'bbox': [detection.bbox_x1, detection.bbox_y1, 
                                detection.bbox_x2, detection.bbox_y2]
                    })
                
                # Add detailed report data if available
                if report.report_data:
                    try:
                        detailed_data = json.loads(report.report_data)
                        result['detailed_analysis'] = detailed_data
                    except json.JSONDecodeError:
                        pass
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to get analysis by ID {analysis_id}: {e}")
            return None
    
    def get_video_statistics(self) -> Dict[str, Any]:
        """Get overall video processing statistics"""
        try:
            with self.db_manager.get_session() as session:
                # Basic counts
                total_videos = session.query(Video).count()
                total_detections = session.query(Detection).count()
                total_brands = session.query(Brand).count()
                
                # Brand popularity
                brand_stats = session.query(
                    Brand.name,
                    func.count(Detection.id).label('detection_count')
                ).join(Detection).group_by(Brand.name).all()
                
                # Recent videos
                recent_videos = session.query(Video).order_by(
                    desc(Video.processed_at)
                ).limit(10).all()
                
                # Average processing time
                avg_processing_time = session.query(
                    func.avg(Video.processing_time)
                ).scalar()
                
                return {
                    'overview': {
                        'total_videos': total_videos,
                        'total_detections': total_detections,
                        'total_brands': total_brands,
                        'avg_processing_time': float(avg_processing_time) if avg_processing_time else 0
                    },
                    'brand_popularity': [
                        {'brand': stat.name, 'detections': stat.detection_count}
                        for stat in brand_stats
                    ],
                    'recent_videos': [
                        {
                            'id': video.id,
                            'filename': video.filename,
                            'duration': video.duration,
                            'processed_at': video.processed_at.isoformat()
                        }
                        for video in recent_videos
                    ]
                }
                
        except Exception as e:
            logger.error(f"Failed to get video statistics: {e}")
            return {}
    
    def _extract_cropped_image(self, video_path: str, frame_idx: int, bbox: List[float]) -> Optional[bytes]:
        """Extract and encode cropped image from video frame"""
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None
            
            # Crop image
            x1, y1, x2, y2 = map(int, bbox)
            cropped = frame[y1:y2, x1:x2]
            
            # Encode as JPEG
            _, encoded = cv2.imencode('.jpg', cropped)
            return encoded.tobytes()
            
        except Exception as e:
            logger.error(f"Failed to extract cropped image: {e}")
            return None
