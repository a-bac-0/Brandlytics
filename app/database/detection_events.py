"""
Detection Events Operations
Database operations for detection events and tracking analytics.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
from uuid import UUID

from app.database.connection import get_supabase
from app.api.schemas.video_schemas import DetectionEventCreate, DetectionEvent

logger = logging.getLogger(__name__)


def save_detection_events(events_data: List[DetectionEventCreate]) -> List[Dict]:
    """
    Save detection events to the database.
    
    Args:
        events_data: List of DetectionEventCreate objects
    
    Returns:
        List of saved detection events
    """
    if not events_data:
        return []
    
    supabase = get_supabase()
    data_to_insert = [event.model_dump() for event in events_data]
    
    try:
        response = supabase.from_("detection_events").insert(data_to_insert).execute()
        
        if response.data is None and response.error is not None:
            raise Exception(f"Error saving detection events to Supabase: {response.error.message}")
        
        logger.info(f"Saved {len(response.data)} detection events to Supabase")
        return response.data
        
    except Exception as e:
        logger.error(f"Error in detection events database operation: {e}")
        raise e


def get_detection_events_by_video(video_name: str) -> List[Dict]:
    """
    Get all detection events for a specific video.
    
    Args:
        video_name: Name of the video
    
    Returns:
        List of detection events
    """
    supabase = get_supabase()
    
    try:
        response = supabase.from_("detection_events").select("*").eq("video_name", video_name).execute()
        
        if response.error:
            raise Exception(f"Error fetching detection events: {response.error.message}")
        
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error fetching detection events for video {video_name}: {e}")
        raise e


def get_detection_events_by_brand(brand_name: str, video_name: Optional[str] = None) -> List[Dict]:
    """
    Get detection events filtered by brand name.
    
    Args:
        brand_name: Name of the brand
        video_name: Optional video name filter
    
    Returns:
        List of detection events
    """
    supabase = get_supabase()
    
    try:
        query = supabase.from_("detection_events").select("*").eq("brand_name", brand_name)
        
        if video_name:
            query = query.eq("video_name", video_name)
        
        response = query.execute()
        
        if response.error:
            raise Exception(f"Error fetching detection events: {response.error.message}")
        
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error fetching detection events for brand {brand_name}: {e}")
        raise e


def aggregate_detection_events_from_detections(video_name: str) -> List[DetectionEventCreate]:
    """
    Generate detection events by aggregating individual detections with the same track_id.
    
    Args:
        video_name: Name of the video to aggregate
    
    Returns:
        List of DetectionEventCreate objects ready to be saved
    """
    supabase = get_supabase()
    
    try:
        # Get all detections for the video that have track_ids
        response = supabase.from_("detections").select("*").eq("video_name", video_name).not_.is_("track_id", "null").execute()
        
        if response.error:
            raise Exception(f"Error fetching detections: {response.error.message}")
        
        detections = response.data or []
        
        if not detections:
            logger.info(f"No detections with track_id found for video {video_name}")
            return []
        
        # Group detections by (brand_name, track_id)
        tracking_groups = {}
        
        for detection in detections:
            key = (detection['brand_name'], detection['track_id'])
            
            if key not in tracking_groups:
                tracking_groups[key] = []
            
            tracking_groups[key].append(detection)
        
        # Create detection events from grouped data
        detection_events = []
        
        for (brand_name, track_id), group in tracking_groups.items():
            # Sort by frame number
            group.sort(key=lambda x: x['frame_number'])
            
            # Calculate aggregated values
            confidences = [det['confidence'] for det in group]
            timestamps = [det['created_at'] for det in group]
            
            event = DetectionEventCreate(
                video_name=video_name,
                brand_name=brand_name,
                track_id=track_id,
                start_frame=group[0]['frame_number'],
                end_frame=group[-1]['frame_number'],
                total_frames=len(group),
                min_confidence=min(confidences),
                max_confidence=max(confidences),
                avg_confidence=sum(confidences) / len(confidences),
                first_detected=datetime.fromisoformat(timestamps[0].replace('Z', '+00:00')),
                last_detected=datetime.fromisoformat(timestamps[-1].replace('Z', '+00:00'))
            )
            
            detection_events.append(event)
        
        logger.info(f"Generated {len(detection_events)} detection events for video {video_name}")
        return detection_events
        
    except Exception as e:
        logger.error(f"Error aggregating detection events for video {video_name}: {e}")
        raise e


def generate_and_save_detection_events(video_name: str) -> List[Dict]:
    """
    Generate detection events from detections and save them to the database.
    
    Args:
        video_name: Name of the video
    
    Returns:
        List of saved detection events
    """
    try:
        # First, check if events already exist for this video
        existing_events = get_detection_events_by_video(video_name)
        
        if existing_events:
            logger.info(f"Detection events already exist for video {video_name}. Skipping generation.")
            return existing_events
        
        # Generate new events
        detection_events = aggregate_detection_events_from_detections(video_name)
        
        if not detection_events:
            logger.info(f"No detection events to generate for video {video_name}")
            return []
        
        # Save to database
        saved_events = save_detection_events(detection_events)
        
        logger.info(f"Successfully generated and saved {len(saved_events)} detection events for video {video_name}")
        return saved_events
        
    except Exception as e:
        logger.error(f"Error generating detection events for video {video_name}: {e}")
        raise e


def get_video_analytics(video_name: str) -> Dict:
    """
    Get comprehensive analytics for a video including detection events.
    
    Args:
        video_name: Name of the video
    
    Returns:
        Dictionary with video analytics
    """
    try:
        # Get detection events
        events = get_detection_events_by_video(video_name)
        
        if not events:
            # Try to generate events if none exist
            events = generate_and_save_detection_events(video_name)
        
        # Aggregate analytics
        unique_brands = list(set(event['brand_name'] for event in events))
        
        tracking_summary = {}
        for brand in unique_brands:
            brand_events = [e for e in events if e['brand_name'] == brand]
            tracking_summary[brand] = {
                'total_tracks': len(brand_events),
                'total_frames': sum(e['total_frames'] for e in brand_events),
                'avg_confidence': sum(e['avg_confidence'] for e in brand_events) / len(brand_events) if brand_events else 0,
                'max_confidence': max(e['max_confidence'] for e in brand_events) if brand_events else 0
            }
        
        return {
            'video_name': video_name,
            'total_detection_events': len(events),
            'unique_brands': unique_brands,
            'tracking_summary': tracking_summary,
            'detection_events': events,
            'analytics_generated_at': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error generating analytics for video {video_name}: {e}")
        raise e
