from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
import cv2
import numpy as np
from uuid import uuid4                    
from pathlib import Path
from app.services.image_detection import ImageDetectionService
from app.api.schemas.schemas_detection import DetectionCreate 
from app.database.operations import save_detections, upload_public_bytes
from app.config.model_config import settings

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
        if image_file:
            image_source = await image_file.read()
            filename = image_file.filename
        else:
            image_source = image_url
            filename = image_url.split("/")[-1]

        detections = detection_service.detect_brands(image_source)

        img_array = detection_service.load_image(image_source)
        if img_array is None:
            raise HTTPException(status_code=400, detail="No se pudo cargar la imagen para recortar.")

        detections_to_save: List[DetectionCreate] = []
        stem = Path(filename).stem

        for det in detections:
            crop_url = None
            x1, y1, x2, y2 = det["box"]
            crop = img_array[y1:y2, x1:x2]
            if crop.size > 0:
                ok, buf = cv2.imencode(".jpg", crop)
                if ok:
                    storage_path = f"{stem}/{uuid4().hex}_{det['class_name']}.jpg"
                    crop_url = upload_public_bytes(
                        bucket=settings.BUCKET_CROPS,    
                        path=storage_path,
                        content=buf.tobytes(),
                        content_type="image/jpeg"
                    )

            new_detection = DetectionCreate(
                video_name=filename,
                frame_number=1,  
                brand_name=det["class_name"],
                confidence=det["confidence"],
                bbox_x1=det["box"][0],
                bbox_y1=det["box"][1],
                bbox_x2=det["box"][2],
                bbox_y2=det["box"][3],
                image_crop_url=crop_url,
                detection_type="image"
            )
            detections_to_save.append(new_detection)

        # Guardar en la base de datos
        db_status = "0 detecciones guardadas en Supabase."
        if detections_to_save:
            save_detections(detections_to_save)
            db_status = f"{len(detections_to_save)} detecciones guardadas en Supabase."

        return {
            "status": "success", 
            "filename": filename, 
            "detections": detections,
            "database_status": db_status  
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-video")
async def process_video(video: UploadFile = File(...)):
    return {
        "status": "received", 
        "filename": video.filename, 
        "message": "Video recibido. El procesamiento comenzará pronto."
    }

