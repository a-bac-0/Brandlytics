from fastapi import APIRouter, HTTPException
from typing import List
from app.database.operations import get_video_detections_byname
from app.services.analytics_service import AnalyticsService


router = APIRouter()
analytics_service = AnalyticsService()

@router.get("/analysis/{video_name}", summary="Obtener análisis de vídeo")
async def get_analytics_for_video(video_name: str):
    detections = get_video_detections_byname(video_name)
    if not detections:
        raise HTTPException(status_code=404, detail=f"No se encontraron detecciones para el video : {video_name}")

    analysis_summary = analytics_service.calculate_video_summary(detections, video_name)

    return analysis_summary