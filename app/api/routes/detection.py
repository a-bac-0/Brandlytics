from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
from app.services.image_detection import ImageDetectionService
from app.api.schemas.schemas_detection import DetectionCreate 
from app.database.operations import save_detections

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
        detections_to_save: List[DetectionCreate] = []
        for det in detections:
            new_detection = DetectionCreate(
                video_name=filename,
                frame_number=1,  # Valor por defecto para imágenes estáticas
                brand_name=det["class_name"],
                confidence=det["confidence"],
                bbox_x1=det["box"][0],
                bbox_y1=det["box"][1],
                bbox_x2=det["box"][2],
                bbox_y2=det["box"][3],
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
            "database_status": db_status  # Añadimos información sobre el guardado
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

