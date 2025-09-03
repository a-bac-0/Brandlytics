from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.api.routes.detection import router as detection_router
from app.api.routes.analytics import router as analytics_router
from app.api.routes.huggingface import router as huggingface_router
from app.api.routes.video import router as video_router

load_dotenv()

app = FastAPI(
    title="Brandlytics API",
    description="API para la detección de marcas en imágenes y vídeos con integración de Hugging Face Hub.",
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
app.include_router(huggingface_router, prefix="/api", tags=["huggingface"])
app.include_router(video_router, prefix="/api/video", tags=["video"])


@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Brandlytics"}


@app.get("/brands")
def get_supported_brands():
    """Get list of brands that can be detected by the ModelM from HuggingFace."""
    # Return the 3 brands detected by CV-Brandlytics/ModelM
    try:
        brand_list = ["coca cola", "nike", "starbucks"]
        
        return {
            "brands": brand_list,
            "total_brands": len(brand_list),
            "models_count": 1,
            "model_source": "CV-Brandlytics/ModelM",
            "note": "Using single HuggingFace model for 3-brand detection"
        }
    except Exception as e:
        # Fallback response if there's any error
        return {
            "brands": ["coca cola", "nike", "starbucks"],
            "total_brands": 3,
            "models_count": 1,
            "model_source": "CV-Brandlytics/ModelM",
            "note": "Using fallback brand list"
        }