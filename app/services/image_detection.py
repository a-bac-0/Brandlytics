from typing import List, Dict, Optional, Any
import cv2
from pathlib import Path
import numpy as np
import requests
from ultralytics import YOLO
from app.config.model_config import settings
from app.api.schemas.schemas_detection import DetectionCreate
from app.utils.image_processing import crop_and_upload_detection

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
        if isinstance(image_source, np.ndarray):
            img = image_source
        else:
            img = self.load_image(image_source)
        if img is None:
            raise ValueError("No se pudo cargar la imagen.")
        
        results = self.model.predict(source=img, conf=conf, iou=iou)[0]

        detections = []
        for det in results.boxes:
            x1, y1, x2, y2 = [int(v) for v in det.xyxy[0]]
            detection_data = {
                "class_id": int(det.cls[0]),
                "brand_name": self.model.names[int(det.cls[0])],
                "confidence": float(det.conf[0]),
                "bbox": [x1, y1, x2, y2]
            }
            detections.append(detection_data)

        return detections
    
    def process_image_input(self, image_source: Any, filename: str) -> Dict:
        img_np = self.load_image(image_source)
        if img_np is None:
            raise ValueError("No se pudo cargar la imagen.")

        detections = self.detect_brands(img_np)

        detections_to_save: List[DetectionCreate] = []
        stem = Path(filename).stem

        for det in detections:
            crop_url = crop_and_upload_detection(img_np, det, stem)
            
            detections_to_save.append(DetectionCreate(
                video_name=filename,
                frame_number=1,
                brand_name=det["brand_name"],
                confidence=det["confidence"],
                bbox_x1=det["bbox"][0], bbox_y1=det["bbox"][1],
                bbox_x2=det["bbox"][2], bbox_y2=det["bbox"][3],
                image_crop_url=crop_url,
                detection_type="image"
            ))
        
        return {
            "filename": filename,
            "detections": detections,
            "detections_to_save": detections_to_save
        }

