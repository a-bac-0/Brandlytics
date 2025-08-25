from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class DetectionResult(BaseModel):
    brand_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    class_id: int

class VideoInfo(BaseModel):
    path: str
    duration: float
    total_frames: int
    processed_frames: int
    fps: float
    processing_time: float
    avg_processing_time_per_frame: float

class BrandAnalysis(BaseModel):
    total_appearances: int
    total_time_seconds: float
    appearance_percentage: float
    average_confidence: float
    max_confidence: float

class ProcessingStats(BaseModel):
    total_frames_processed: int
    processing_time: float
    avg_time_per_frame: float

class VideoAnalysisResponse(BaseModel):
    video_info: VideoInfo
    summary: Dict[str, Any]
    brand_analysis: Dict[str, BrandAnalysis]
    processing_stats: ProcessingStats
    output_video_path: Optional[str] = None
    analysis_id: str

class BrandInfo(BaseModel):
    id: int
    name: str
    color: List[int]

class APIStatus(BaseModel):
    message: str
    status: str
    version: str
    model_loaded: bool