"""
Video Processing Schemas
Pydantic models for video processing API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
from datetime import datetime


class VideoProcessingRequest(BaseModel):
    """Request model for video processing."""
    video_name: str = Field(..., description="Name for the video (used in database)")
    save_to_db: bool = Field(default=True, description="Whether to save detections to database")
    max_frames: Optional[int] = Field(default=None, description="Maximum frames to process (for testing)")
    generate_output_video: bool = Field(default=False, description="Whether to generate annotated output video")
    fps_sample: Optional[int] = Field(default=1, description="Frames per second to sample")


class VideoUploadRequest(BaseModel):
    """Request model for video upload and processing."""
    video_name: str = Field(..., description="Name for the video")
    save_to_db: bool = Field(default=True, description="Whether to save detections to database")
    max_frames: Optional[int] = Field(default=None, description="Maximum frames to process")
    generate_output_video: bool = Field(default=False, description="Generate annotated video")


class VideoBatchRequest(BaseModel):
    """Request model for batch video processing."""
    video_names: List[str] = Field(..., description="List of video names to process")
    save_to_db: bool = Field(default=True, description="Whether to save detections to database")
    max_frames: Optional[int] = Field(default=None, description="Maximum frames to process per video")


class VideoProcessingResponse(BaseModel):
    """Response model for video processing results."""
    video_name: str
    status: str
    processing_time: float
    total_frames_processed: int
    brands_detected: List[str]
    total_detections: int
    performance_metrics: Dict
    brand_statistics: Dict
    metadata: Dict
    error: Optional[str] = None


class VideoBatchResponse(BaseModel):
    """Response model for batch video processing."""
    total_videos: int
    successful: int
    failed: int
    results: Dict[str, VideoProcessingResponse]
    summary: Dict[str, Union[int, List[str]]]


class VideoStatsResponse(BaseModel):
    """Response model for video processor statistics."""
    total_detection_time: float
    processed_frames: int
    active_tracks: int
    model_path: str
    uptime: Optional[str] = None


class DetectionEvent(BaseModel):
    """Model for detection events (tracking analytics)."""
    id: Optional[int] = None
    detection_id: Optional[str] = None
    video_name: str
    brand_name: str
    track_id: str
    start_frame: int
    end_frame: int
    total_frames: int
    min_confidence: float
    max_confidence: float
    avg_confidence: float
    first_detected: datetime
    last_detected: datetime


class DetectionEventCreate(BaseModel):
    """Model for creating detection events."""
    video_name: str
    brand_name: str
    track_id: str
    start_frame: int
    end_frame: int
    total_frames: int
    min_confidence: float = Field(..., ge=0.0, le=1.0)
    max_confidence: float = Field(..., ge=0.0, le=1.0)
    avg_confidence: float = Field(..., ge=0.0, le=1.0)
    first_detected: datetime
    last_detected: datetime


class VideoAnalyticsResponse(BaseModel):
    """Response model for video analytics and detection events."""
    video_name: str
    total_detection_events: int
    unique_brands: List[str]
    tracking_summary: Dict[str, Dict]
    detection_events: List[DetectionEvent]
    analytics_generated_at: datetime
