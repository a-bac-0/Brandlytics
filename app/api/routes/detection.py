from fastapi import APIRouter, UploadFile, File

router = APIRouter()

#Rutas provisionales, ya que falta crear los servicios a los que llamaremos desde las rutas.


@router.post("/process-video")
async def process_video(video: UploadFile = File(...)):
    # Cuando lo tengamos se modifica.
    return {
        "status": "received", 
        "filename": video.filename, 
        "message": "Video recibido. El procesamiento comenzará pronto."
    }

@router.post("/process-image")
async def process_image(image: UploadFile = File(...)):
    #igualmente aquí.
    return {
        "status": "received", 
        "filename": image.filename, 
        "message": "Imagen recibida. El procesamiento comenzará pronto."
    }