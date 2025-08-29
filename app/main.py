from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.api.routes.detection import router as detection_router
from app.api.routes.analytics import router as analytics_router
from app.api.routes.video import router as video_router

load_dotenv()

app = FastAPI(
    title="Brandlytics API",
    description="API para la detección de marcas en imágenes y vídeos.",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas
app.include_router(detection_router, prefix="/api/detection", tags=["detection"])
app.include_router(analytics_router, prefix="/api/analytics", tags=["analytics"])
app.include_router(video_router, prefix="/api/video", tags=["video"])


@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Brandlytics"}