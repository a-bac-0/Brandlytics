"""
Video Processing API Routes
FastAPI routes for video brand detection and analytics.
"""

import logging
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse, FileResponse

from app.services.video_detection import video_detection_service
from app.api.schemas.video_schemas import (
    VideoProcessingRequest,
    VideoUploadRequest,
    VideoProcessingResponse,
    VideoBatchRequest,
    VideoBatchResponse,
    VideoStatsResponse,
    VideoAnalyticsResponse
)
from app.database.detection_events import (
    get_detection_events_by_video,
    get_detection_events_by_brand,
    generate_and_save_detection_events,
    get_video_analytics
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# Temporary storage for uploaded videos
UPLOAD_DIR = Path("data/raw/videos/input")
OUTPUT_DIR = Path("data/raw/videos/output")

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=VideoProcessingResponse)
async def upload_and_process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    video_name: Optional[str] = None,
    save_to_db: bool = True,
    max_frames: Optional[int] = None,
    generate_output_video: bool = False
):
    """
    Upload and process a video file for brand detection.
    """
    try:
        # Validate file type
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Generate video name
        if video_name is None:
            video_name = Path(file.filename).stem
        
        # Save uploaded file
        video_path = UPLOAD_DIR / f"{video_name}.mp4"
        
        with open(video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Video uploaded: {video_path}")
        
        # Determine output path if needed
        output_path = None
        if generate_output_video:
            output_path = OUTPUT_DIR / f"{video_name}_annotated.mp4"
        
        # Process video
        results = await video_detection_service.process_video_file(
            video_path=video_path,
            video_name=video_name,
            save_to_db=save_to_db,
            max_frames=max_frames,
            output_path=output_path
        )
        
        # Generate detection events in background if saved to DB
        if save_to_db:
            background_tasks.add_task(generate_and_save_detection_events, video_name)
        
        # Format response
        response = VideoProcessingResponse(
            video_name=video_name,
            status="completed",
            processing_time=results['performance']['total_elapsed_seconds'],
            total_frames_processed=results['performance']['processed_frames'],
            brands_detected=results['summary']['brands_found'],
            total_detections=results['summary']['total_detections'],
            performance_metrics=results['performance'],
            brand_statistics=results['brand_detections'],
            metadata=results.get('metadata', {})
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing uploaded video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")


@router.post("/process", response_model=VideoProcessingResponse)
async def process_existing_video(
    background_tasks: BackgroundTasks,
    request: VideoProcessingRequest
):
    """
    Process an existing video file by name.
    """
    try:
        # Find video file
        video_path = UPLOAD_DIR / f"{request.video_name}.mp4"
        
        if not video_path.exists():
            # Try alternative extensions
            for ext in ['.avi', '.mov', '.mkv']:
                alt_path = UPLOAD_DIR / f"{request.video_name}{ext}"
                if alt_path.exists():
                    video_path = alt_path
                    break
            else:
                raise HTTPException(status_code=404, detail=f"Video not found: {request.video_name}")
        
        # Determine output path if needed
        output_path = None
        if request.generate_output_video:
            output_path = OUTPUT_DIR / f"{request.video_name}_annotated.mp4"
        
        # Process video
        results = await video_detection_service.process_video_file(
            video_path=video_path,
            video_name=request.video_name,
            save_to_db=request.save_to_db,
            max_frames=request.max_frames,
            output_path=output_path
        )
        
        # Generate detection events in background if saved to DB
        if request.save_to_db:
            background_tasks.add_task(generate_and_save_detection_events, request.video_name)
        
        # Format response
        response = VideoProcessingResponse(
            video_name=request.video_name,
            status="completed",
            processing_time=results['performance']['total_elapsed_seconds'],
            total_frames_processed=results['performance']['processed_frames'],
            brands_detected=results['summary']['brands_found'],
            total_detections=results['summary']['total_detections'],
            performance_metrics=results['performance'],
            brand_statistics=results['brand_detections'],
            metadata=results.get('metadata', {})
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing video {request.video_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")


@router.post("/batch", response_model=VideoBatchResponse)
async def process_video_batch(
    background_tasks: BackgroundTasks,
    request: VideoBatchRequest
):
    """
    Process multiple videos in batch.
    """
    try:
        # Process videos
        results = await video_detection_service.process_video_batch(
            video_paths=[UPLOAD_DIR / f"{name}.mp4" for name in request.video_names],
            save_to_db=request.save_to_db,
            max_frames=request.max_frames
        )
        
        # Convert results to response format
        formatted_results = {}
        successful = 0
        failed = 0
        all_brands = set()
        
        for video_name, result in results.items():
            if 'error' in result:
                failed += 1
                formatted_results[video_name] = VideoProcessingResponse(
                    video_name=video_name,
                    status="failed",
                    processing_time=0,
                    total_frames_processed=0,
                    brands_detected=[],
                    total_detections=0,
                    performance_metrics={},
                    brand_statistics={},
                    metadata={},
                    error=result['error']
                )
            else:
                successful += 1
                brands_found = result['summary']['brands_found']
                all_brands.update(brands_found)
                
                formatted_results[video_name] = VideoProcessingResponse(
                    video_name=video_name,
                    status="completed",
                    processing_time=result['performance']['total_elapsed_seconds'],
                    total_frames_processed=result['performance']['processed_frames'],
                    brands_detected=brands_found,
                    total_detections=result['summary']['total_detections'],
                    performance_metrics=result['performance'],
                    brand_statistics=result['brand_detections'],
                    metadata=result.get('metadata', {})
                )
                
                # Generate detection events in background
                if request.save_to_db:
                    background_tasks.add_task(generate_and_save_detection_events, video_name)
        
        # Create summary
        summary = {
            'total_unique_brands': len(all_brands),
            'brands_found': list(all_brands),
            'total_detections': sum(r.total_detections for r in formatted_results.values() if r.status == "completed")
        }
        
        response = VideoBatchResponse(
            total_videos=len(request.video_names),
            successful=successful,
            failed=failed,
            results=formatted_results,
            summary=summary
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in batch video processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@router.get("/analytics/{video_name}", response_model=VideoAnalyticsResponse)
async def get_video_analytics_endpoint(video_name: str):
    """
    Get analytics and detection events for a specific video.
    """
    try:
        analytics = get_video_analytics(video_name)
        
        response = VideoAnalyticsResponse(**analytics)
        return response
        
    except Exception as e:
        logger.error(f"Error getting analytics for video {video_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analytics retrieval failed: {str(e)}")


@router.get("/events/{video_name}")
async def get_detection_events(video_name: str):
    """
    Get detection events for a specific video.
    """
    try:
        events = get_detection_events_by_video(video_name)
        return {"video_name": video_name, "detection_events": events}
        
    except Exception as e:
        logger.error(f"Error getting detection events for video {video_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection events retrieval failed: {str(e)}")


@router.get("/events/brand/{brand_name}")
async def get_brand_detection_events(
    brand_name: str,
    video_name: Optional[str] = Query(None, description="Filter by video name")
):
    """
    Get detection events filtered by brand name.
    """
    try:
        events = get_detection_events_by_brand(brand_name, video_name)
        return {
            "brand_name": brand_name,
            "video_name": video_name,
            "detection_events": events
        }
        
    except Exception as e:
        logger.error(f"Error getting detection events for brand {brand_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Brand detection events retrieval failed: {str(e)}")


@router.post("/events/generate/{video_name}")
async def generate_detection_events_endpoint(video_name: str):
    """
    Manually trigger generation of detection events for a video.
    """
    try:
        events = generate_and_save_detection_events(video_name)
        return {
            "video_name": video_name,
            "generated_events": len(events),
            "detection_events": events
        }
        
    except Exception as e:
        logger.error(f"Error generating detection events for video {video_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection events generation failed: {str(e)}")


@router.get("/stats", response_model=VideoStatsResponse)
async def get_processor_stats():
    """
    Get current video processor statistics.
    """
    try:
        stats = video_detection_service.get_processor_stats()
        
        response = VideoStatsResponse(**stats)
        return response
        
    except Exception as e:
        logger.error(f"Error getting processor stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


@router.get("/download/{video_name}")
async def download_annotated_video(video_name: str):
    """
    Download annotated output video.
    """
    try:
        output_path = OUTPUT_DIR / f"{video_name}_annotated.mp4"
        
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="Annotated video not found")
        
        return FileResponse(
            path=output_path,
            filename=f"{video_name}_annotated.mp4",
            media_type="video/mp4"
        )
        
    except Exception as e:
        logger.error(f"Error downloading video {video_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video download failed: {str(e)}")


@router.get("/list")
async def list_available_videos():
    """
    List all available videos for processing.
    """
    try:
        video_files = []
        
        for video_path in UPLOAD_DIR.glob("*"):
            if video_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                video_files.append({
                    "name": video_path.stem,
                    "filename": video_path.name,
                    "size": video_path.stat().st_size,
                    "modified": video_path.stat().st_mtime
                })
        
        return {"available_videos": video_files}
        
    except Exception as e:
        logger.error(f"Error listing videos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video listing failed: {str(e)}")
