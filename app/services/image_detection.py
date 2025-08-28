import os
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Any
import cv2
import numpy as np
import requests
from ultralytics import YOLO
from io import BytesIO
from app.config.model_config import settings

class ImageDetectionService:
    model: YOLO = None

    def __init__(self):
        model_path = settings.MODEL_PATH
        if self.model is None:
            print(f"Cargando modelo desde: {model_path}")
            try:
                self.__class__.model = YOLO(model_path)
            except Exception as e:
                raise RuntimeError(f"Error al cargar el modelo")


    #Cargamos la imagen desde una url, local o un archivo subido:        
    def load_image(self, image_source: Any) -> np.ndarray:
        if isinstance(image_source, str) and image_source.startswith('http'):
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            resp = requests.get(image_source, headers=headers)
            resp.raise_for_status()
            image_np = np.frombuffer(resp.content, np.uint8)
            return cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        elif isinstance(image_source, bytes):
            image_np = np.frombuffer(image_source, np.uint8)
            return cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        elif isinstance(image_source, str):
            return cv2.imread(image_source)
        else:
            raise ValueError("Tipo de fuente de imagen no soportado")
        
    def detect_brands(self, image_source: Any, conf: float = 0.5, iou: float = 0.6) -> List[Dict[str, Any]]:
        img = self.load_image(image_source)
        if img is None:
            raise ValueError("No se pudo cargar la imagen.")
        
        results = self.model.predict(source=img, conf=conf, iou=iou)[0]

        detections = []
        for det in results.boxes:
            x1, y1, x2, y2 = [int(v) for v in det.xyxy[0]]
            detection_data = {
                "class_id": int(det.cls[0]),
                "class_name": self.model.names[int(det.cls[0])],
                "confidence": float(det.conf[0]),
                "box": [x1, y1, x2, y2]
            }
            detections.append(detection_data)

        return detections


    def crop_and_upload(self, image_source: Any, detection: Dict[str, Any]) -> Optional[str]:
        try:
            if not isinstance(image_source, np.ndarray):
                img = self.load_image(image_source)
            else:
                img = image_source
                
            if img is None:
                return None
                
            # Recortar la región detectada
            x1, y1, x2, y2 = detection["box"]
            cropped_img = img[y1:y2, x1:x2]
            
            unique_id = uuid.uuid4().hex[:8]
            crop_filename = f"{detection['class_name']}_{unique_id}.jpg"
            
            # Convertir la imagen a bytes
            _, buffer = cv2.imencode('.jpg', cropped_img)
            crop_bytes = BytesIO(buffer).getvalue()
            
            # Subir a Supabase
            from app.database.connection import get_supabase
            supabase = get_supabase()
            
            bucket_name = "brand-crops"
            
            try:
                buckets = supabase.storage.list_buckets()
                bucket_exists = any(b["name"] == bucket_name for b in buckets)
                
                if not bucket_exists:
                    supabase.storage.create_bucket(bucket_name)
            except:
                try:
                    supabase.storage.create_bucket(bucket_name)
                except:
                    pass
            
            supabase.storage.from_(bucket_name).upload(
                path=crop_filename,
                file=crop_bytes,
                file_options={"content-type": "image/jpeg"}
            )
            
            image_url = supabase.storage.from_(bucket_name).get_public_url(crop_filename)
            return image_url
            
        except Exception as e:
            print(f"Error al recortar y subir imagen: {str(e)}")
            return None