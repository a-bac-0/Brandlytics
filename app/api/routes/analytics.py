from fastapi import APIRouter

router = APIRouter()

# Cuando sepamos qué analítica realizaremos lo modificaremos.

@router.get("/results/{video_name}")
async def get_analytics_for_video(video_name: str):
    return {
        "message": f"Obteniendo analíticas para el vídeo: {video_name}",
        "data": [] # Los datos vendrán de la base de datos en el futuro, cuando lo tengamos
    }