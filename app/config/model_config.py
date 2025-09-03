import os
from dotenv import load_dotenv


load_dotenv()

class Settings:
    # Configuración del Modelo
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/best2.pt")
    
    # Hugging Face Configuration
    HF_TOKEN: str = os.getenv("HF_TOKEN")
    HF_ORG: str = os.getenv("HF_ORG", "CV-Brandlytics")
    HF_MODEL_REPO: str = os.getenv("HF_MODEL_REPO", "CV-Brandlytics/ModelM")
    USE_HF_MODEL: bool = os.getenv("USE_HF_MODEL", "false").lower() == "true"
    INCLUDE_LOCAL_MODEL: bool = os.getenv("INCLUDE_LOCAL_MODEL", "true").lower() == "true"
    HF_CACHE_DIR: str = os.getenv("HF_CACHE_DIR", "models/hf_cache")

    # Configuración de Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")
    BUCKET_CROPS: str = os.getenv("BUCKET_CROPS", "crops")
    
    # Configuración de Video Processing
    VIDEO_FPS_SAMPLE: int = int(os.getenv("VIDEO_FPS_SAMPLE", "1"))
    VIDEO_MAX_FRAMES: int = int(os.getenv("VIDEO_MAX_FRAMES", "0"))  # 0 = no limit
    VIDEO_UPLOAD_DIR: str = os.getenv("VIDEO_UPLOAD_DIR", "data/raw/videos/input")
    VIDEO_OUTPUT_DIR: str = os.getenv("VIDEO_OUTPUT_DIR", "data/raw/videos/output")
    
    # ByteTracker Configuration
    TRACKER_FRAME_RATE: int = int(os.getenv("TRACKER_FRAME_RATE", "30"))
    TRACKER_TRACK_THRESH: float = float(os.getenv("TRACKER_TRACK_THRESH", "0.5"))
    TRACKER_TRACK_BUFFER: int = int(os.getenv("TRACKER_TRACK_BUFFER", "30"))
    TRACKER_MATCH_THRESH: float = float(os.getenv("TRACKER_MATCH_THRESH", "0.8"))

# Creamos una instancia única de la configuración para usarla en toda la app
settings = Settings()