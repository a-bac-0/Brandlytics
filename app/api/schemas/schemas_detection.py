from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
from datetime import datetime

class DetectionBase(BaseModel):
    video_name: Optional[str] = None 
    frame_number: Optional[int] = None 
    timestamp_seconds: Optional[float] = None
    brand_name: str
    confidence: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    image_crop_url: Optional[str] = None
    track_id: Optional[UUID] = None
    fps: Optional[float] = None
    detection_type: str = 'image'

class DetectionCreate(DetectionBase):
    pass

class Detection(DetectionBase):
    id: UUID
    created_at: datetime

    class config:
        orm_mode = True

class DetectionEventBase(BaseModel):
    detection_id: Optional[UUID] = None
    video_name: Optional[str] = None
    brand_name: Optional[str] = None
    track_id: Optional[UUID] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    total_frames: Optional[int] = None
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    avg_confidence: Optional[float] = None
    first_detected: Optional[datetime] = None
    last_detected: Optional[datetime] = None

class DetectionEventCreate(DetectionEventBase):
    pass

class DetectionEvent(DetectionEventBase):
    id: int

    class config:
        orm_mode = True

