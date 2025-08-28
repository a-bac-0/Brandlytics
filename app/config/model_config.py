import os
from dotenv import load_dotenv


load_dotenv()

class Settings:
    # Configuración del Modelo
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/best.pt")

    # Configuración de Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")
    BUCKET_CROPS: str = os.getenv("BUCKET_CROPS", "crops")

# Creamos una instancia única de la configuración para usarla en toda la app
settings = Settings()