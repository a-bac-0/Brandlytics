from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
from pathlib import Path
from app.services.image_detection import ImageDetectionService
from app.services.video_detection import VideoDetectionService
from app.api.schemas.schemas_detection import DetectionCreate 
from app.database.operations import save_detections
from app.utils.image_processing import crop_and_upload_detection

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
            "filename": filename, 
            "detections": result["detections"],
            "database_status": db_status  
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



## Anca :Ruta basándome en la de la Imagen , pero para vídeo, si tienes que cambiar cosas adelante.
@router.post("/process-video")
async def process_video(
    video_file: Optional[UploadFile] = File(None),
    video_url: Optional[str] = Form(None),
):
    if not video_file and not video_url:
        raise HTTPException(status_code=400, detail="Debes enviar 'video_file' o 'video_url'.")
    if video_file and video_url:
        raise HTTPException(status_code=400, detail="Envía solo uno: 'video_file' o 'video_url'.")
    
    try:
        video_bytes = await video_file.read() if video_file else None
        filename = video_file.filename if video_file else None

        video_service = VideoDetectionService(detection_service)
        result = video_service.process_video_input(
            video_file_bytes=video_bytes,
            filename=filename,
            video_url=video_url
        )

        detections_to_save = result["detections_to_save"]
        saved_count = 0
        if detections_to_save:
            save_detections(detections_to_save)
            saved_count = len(detections_to_save)

        return {
            "status": "success",
            "filename": filename or video_url,
            "fps": result["fps"],
            "total_frames": result["total_frames"],
            "detections_saved": saved_count,
            "database_status": f"{saved_count} detecciones guardadas en Supabase."
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))