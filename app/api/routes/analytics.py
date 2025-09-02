from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from typing import List, Optional
from app.database.operations import get_video_detections_byname
from app.services.analytics_service import analytics_service
from app.api.schemas.video_schemas import VideoAnalyticsResponse
from pydantic import BaseModel
from datetime import datetime


router = APIRouter()

@router.get("/analysis/{video_name}", summary="Obtener análisis detallado de vídeo")
async def get_analytics_for_video(
    video_name: str,
    export_format: Optional[str] = Query(None, description="Formato de exportación (json, csv)")
):
    try:
        analysis_summary = analytics_service.get_video_analysis_from_db(video_name)
        
        if not analysis_summary or "error" in analysis_summary:
            raise HTTPException(status_code=404, detail=f"No se encontraron detecciones para el video: {video_name}")
        
        if export_format:
            if export_format.lower() not in ["json", "csv"]:
                raise HTTPException(status_code=400, detail=f"Formato de exportación no soportado: {export_format}")
                
            return analytics_service.export_analysis(analysis_summary, export_format.lower())
        
        return analysis_summary
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar análisis: {str(e)}")


@router.post("/analyze-video", summary="Analizar video directamente")
async def analyze_video_direct(
    video_file: UploadFile = File(...),
    preset: str = Form("balanced"),
    save_to_db: bool = Form(True)
):
    try:
        video_bytes = await video_file.read()
        filename = video_file.filename
        
        if not filename:
            filename = f"uploaded_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        analysis_results = await analytics_service.analyze_video_file(
            video_bytes=video_bytes,
            filename=filename,
            preset=preset,
            save_to_db=save_to_db
        )
        
        return analysis_results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al analizar video: {str(e)}")