import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BrandDetector:
    """
    Advanced brand detection model with support for multiple architectures
    """
    
    def __init__(self, model_path: str, config: Dict):
        self.config = config
        self.model_path = model_path
        self.model = None
        self.brands = {brand['id']: brand['name'] for brand in config.get('brands', [])}
        self.brand_colors = {brand['id']: brand['color'] for brand in config.get('brands', [])}
        self.confidence_threshold = config.get('model', {}).get('confidence_threshold', 0.5)
        self.iou_threshold = config.get('model', {}).get('iou_threshold', 0.4)
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if self.config.get('model', {}).get('architecture') == 'yolov8':
                self.model = YOLO(self.model_path)
            else:
                raise NotImplementedError(f"Architecture not implemented")
            
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def detect_brands(self, image: np.ndarray) -> List[Dict]:
        """
        Detect brands in a single image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detections with bbox, confidence, and brand info
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Run inference
        results = self.model(image, conf=self.confidence_threshold, iou=self.iou_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract detection information
                    bbox = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if class_id in self.brands:
                        detection = {
                            'bbox': bbox.tolist(),
                            'confidence': confidence,
                            'class_id': class_id,
                            'brand_name': self.brands[class_id],
                            'color': self.brand_colors.get(class_id, [255, 255, 255])
                        }
                        detections.append(detection)
        
        return detections
    
    def annotate_image(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Input image
            detections: List of detections
            
        Returns:
            Annotated image
        """
        annotated_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            brand_name = detection['brand_name']
            confidence = detection['confidence']
            color = tuple(map(int, detection['color']))
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            label = f"{brand_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(annotated_image, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Text
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_image
