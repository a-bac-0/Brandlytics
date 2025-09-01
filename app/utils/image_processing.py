from typing import Dict, Optional
import numpy as np
import cv2
from uuid import uuid4
from app.database.operations import upload_public_bytes
from app.config.model_config import settings
import re

# Función para limpiar una cadena de texto, para utilizarla de forma segura como nombre de archivo o ruta
def sanitize_for_path(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', text)


# Función para recortar la imagen de la marca detectada, y luego subir ese recorte al storage de supabase.
def crop_and_upload_detection(img: np.ndarray, det: Dict, prefix: str) -> Optional[str]:
    safe_prefix = sanitize_for_path(prefix)
    x1, y1, x2, y2 = map(int, det["bbox"])
    h, w = img.shape[:2]
    
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    
    if x2 <= x1 or y2 <= y1:
        return None
        
    crop = img[y1:y2, x1:x2]
    ok, buf = cv2.imencode(".jpg", crop)
    
    if not ok:
        return None

    storage_path = f"{safe_prefix}/{uuid4().hex}_{sanitize_for_path(det['brand_name'])}.jpg"

    return upload_public_bytes(
        bucket=settings.BUCKET_CROPS,
        path=storage_path,
        content=buf.tobytes(),
        content_type="image/jpeg"
    )

