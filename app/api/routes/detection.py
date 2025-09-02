from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
from pathlib import Path
from app.services.image_detection import ImageDetectionService
from app.services.video_detection import VideoDetectionService
from app.api.schemas.schemas_detection import DetectionCreate 
from app.database.operations import save_detections
from app.services.video_detection import video_detection_service
from app.utils.image_processing import crop_and_upload_detection
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

detection_service = ImageDetectionService()

@router.post("/process-image")
async def process_image(
    image_file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None)
):
    if not image_file and not image_url:
        raise HTTPException(status_code=400, detail="Debes proporcionar 'image_file' o 'image_url'.")
    
    try:
        image_source = await image_file.read() if image_file else image_url
        filename = image_file.filename if image_file else image_url.split("/")[-1]

        # Llamamos al método principal del servicio que hace todo el trabajo
        result = detection_service.process_image_input(image_source, filename)

        detections_to_save = result["detections_to_save"]
        db_status = "0 detecciones guardadas en Supabase."
        if detections_to_save:
            save_detections(detections_to_save)
            db_status = f"{len(detections_to_save)} detecciones guardadas en Supabase."
        return {
            "status": "success", 
            "filename": result["filename"],
            "detections": result["detections"],
            "database_status": db_status  
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/process-video")
async def process_video(
    video_file: Optional[UploadFile] = File(None),
    video_url: Optional[str] = Form(None),
    save_to_db: bool = Form(True),
    frame_step: int = Form(30),  # ~1 frame/sec para vídeos de 30fps
    conf: float = Form(0.25),
    iou: float = Form(0.6)
):
    """
    Detecta marcas en un vídeo subido o desde una URL
    """
    if not video_file and not video_url:
        raise HTTPException(status_code=400, detail="Debes enviar 'video_file' o 'video_url'.")
    if video_file and video_url:
        raise HTTPException(status_code=400, detail="Envía solo uno: 'video_file' o 'video_url'.")
    try:
            video_bytes = await video_file.read() if video_file else None
            filename = video_file.filename if video_file else None
            
            # Usar el servicio mejorado de detección de vídeo
            results = await video_detection_service.process_video_input(
                video_file_bytes=video_bytes,
                filename=filename,
                video_url=video_url,
                save_to_db=save_to_db,
                frame_step=frame_step,
                conf=conf,
                iou=iou
            )
            return {
            "status": "success",
            "summary": results["summary"],
            "detections_count": len(results.get("detections", [])),
            "metadata": results.get("metadata", {})
        }
    
    except Exception as e:
        logger.error(f"Error al procesar el vídeo: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))