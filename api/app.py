from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import yaml
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict
import logging
from datetime import datetime

from src.models.detector import BrandDetector
from src.utils.video_processor import VideoProcessor
from src.database.connection import DatabaseManager
from src.database.operations import DetectionOperations
from api.schemas.request_models import VideoAnalysisResponse, DetectionResult

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title="Brand Detection API",
    description="Computer Vision API for detecting brands in videos",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for serving results
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables (in production, use dependency injection)
detector = None
video_processor = None
db_manager = None
db_operations = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global detector, video_processor, db_manager, db_operations
    
    try:
        # Initialize model
        model_path = "models/trained/best.pt"
        if os.path.exists(model_path):
            detector = BrandDetector(model_path, config)
            video_processor = VideoProcessor(detector, config)
            logging.info("Model loaded successfully")
        else:
            logging.warning(f"Model not found at {model_path}")
        
        # Initialize database
        db_manager = DatabaseManager(config)
        db_operations = DetectionOperations(db_manager)
        
        logging.info("API startup completed")
        
    except Exception as e:
        logging.error(f"Startup error: {e}")

@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Brand Detection API",
        "status": "running",
        "version": "1.0.0",
        "model_loaded": detector is not None
    }

@app.get("/brands")
async def get_supported_brands():
    """Get list of supported brands"""
    if not detector:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    brands = []
    for brand_config in config.get('brands', []):
        brands.append({
            'id': brand_config['id'],
            'name': brand_config['name'],
            'color': brand_config['color']
        })
    
    return {"brands": brands}

@app.post("/detect/image", response_model=List[DetectionResult])
async def detect_brands_in_image(file: UploadFile = File(...)):
    """
    Detect brands in a single image
    """
    if not detector:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Load and process image
        import cv2
        image = cv2.imread(tmp_path)
        detections = detector.detect_brands(image)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Format response
        results = []
        for detection in detections:
            results.append({
                "brand_name": detection['brand_name'],
                "confidence": detection['confidence'],
                "bbox": detection['bbox'],
                "class_id": detection['class_id']
            })
        
        return results
        
    except Exception as e:
        logging.error(f"Image detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/video", response_model=VideoAnalysisResponse)
async def detect_brands_in_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    save_to_db: bool = True
):
    """
    Detect brands in a video file
    """
    if not detector or not video_processor:
        raise HTTPException(status_code=503, detail="Services not loaded")
    
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Check file size
    max_size = config.get('api', {}).get('max_video_size_mb', 100) * 1024 * 1024
    if file.size and file.size > max_size:
        raise HTTPException(status_code=413, detail="File too large")
    
    try:
        # Save uploaded video temporarily
        temp_dir = Path("tmp")
        temp_dir.mkdir(exist_ok=True)
        
        input_path = temp_dir / f"input_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        output_path = temp_dir / f"output_{input_path.stem}.mp4"
        
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process video
        logging.info(f"Processing video: {file.filename}")
        analysis_results = video_processor.process_video(
            str(input_path), 
            str(output_path)
        )
        
        # Save to database if requested
        if save_to_db and db_operations:
            background_tasks.add_task(
                save_analysis_to_db,
                analysis_results,
                str(input_path),
                file.filename
            )
        
        # Prepare response
        response = {
            "video_info": analysis_results['video_info'],
            "summary": analysis_results['summary'],
            "brand_analysis": {},
            "processing_stats": {
                "total_frames_processed": analysis_results['video_info']['processed_frames'],
                "processing_time": analysis_results['video_info']['processing_time'],
                "avg_time_per_frame": analysis_results['video_info']['avg_processing_time_per_frame']
            }
        }
        
        # Format brand analysis
        for brand_name, brand_data in analysis_results['brand_detections'].items():
            response["brand_analysis"][brand_name] = {
                "total_appearances": brand_data['total_frames'],
                "total_time_seconds": brand_data['total_time'],
                "appearance_percentage": brand_data.get('appearance_percentage', 0),
                "average_confidence": brand_data.get('avg_confidence', 0),
                "max_confidence": brand_data.get('max_confidence', 0)
            }
        
        # Add file paths for download
        response["output_video_path"] = str(output_path) if output_path.exists() else None
        response["analysis_id"] = str(hash(str(input_path)))
        
        return response
        
    except Exception as e:
        logging.error(f"Video processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def save_analysis_to_db(analysis_results: Dict, video_path: str, filename: str):
    """Background task to save analysis results to database"""
    try:
        if db_operations:
            await db_operations.save_video_analysis(analysis_results, video_path, filename)
            logging.info(f"Analysis saved to database for video: {filename}")
    except Exception as e:
        logging.error(f"Database save error: {e}")

@app.get("/analysis/{analysis_id}")
async def get_analysis_results(analysis_id: str):
    """Get detailed analysis results by ID"""
    if not db_operations:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        results = await db_operations.get_analysis_by_id(analysis_id)
        if not results:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return results
    except Exception as e:
        logging.error(f"Database query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{file_path:path}")
async def download_file(file_path: str):
    """Download processed video or analysis file"""
    full_path = Path(file_path)
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(full_path),
        filename=full_path.name,
        media_type='application/octet-stream'
    )

@app.delete("/cleanup")
async def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        temp_dir = Path("tmp")
        if temp_dir.exists():
            for file in temp_dir.iterdir():
                if file.is_file():
                    file.unlink()
        
        return {"message": "Temporary files cleaned up"}
    except Exception as e:
        logging.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the API
    api_config = config.get('api', {})
    uvicorn.run(
        app,
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000),
        reload=api_config.get('debug', False)
    )