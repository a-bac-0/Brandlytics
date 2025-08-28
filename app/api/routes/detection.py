from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from app.services.image_detection import ImageDetectionService

router = APIRouter()

#Rutas provisionales, ya que falta crear los servicios a los que llamaremos desde las rutas.
detection_service = ImageDetectionService()

@router.post("/process-image")
async def process_image(
    image_file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None)
):
    if not image_file and not image_url:
        raise HTTPException(status_code=400, details="Debes proporcionar 'image_file' o 'image_url'.")
    
    try:
        if image_file:
            image_source = await image_file.read()
            filename = image_file.filename
        else:
            image_source = image_url
            filename = image_url.split("/")[-1]

        detections = detection_service.detect_brands(image_source)

        return {
            "status": "success", 
            "filename": filename, 
            "detections": detections
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-video")
async def process_video(video: UploadFile = File(...)):
    # Cuando lo tengamos se modifica.
    return {
        "status": "received", 
        "filename": video.filename, 
        "message": "Video recibido. El procesamiento comenzará pronto."
    }

