import pytest
import numpy as np
import cv2
from app.utils.visualization import draw_detections_on_image

class TestVisualization:
    
    def test_draw_detections_on_image(self, sample_image, sample_detection_result):
        """Test drawing detections on image"""
        detections = sample_detection_result["detections"]
        
        # Convertir al formato esperado
        formatted_detections = []
        for detection in detections:
            formatted_detections.append({
                "brand_name": detection["class_name"],
                "confidence": detection["confidence"],
                "bbox": detection["box"],
                "class_id": detection["class_id"]
            })
        
        result_image = draw_detections_on_image(sample_image, formatted_detections)
        
        assert isinstance(result_image, np.ndarray)
        assert result_image.shape == sample_image.shape
        # La imagen debería ser diferente después de dibujar
        assert not np.array_equal(result_image, sample_image)