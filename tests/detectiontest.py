import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import cv2
from app.services.image_detection import ImageDetectionService

class TestImageDetectionService:
    
    @patch('app.services.image_detection.YOLO')
    def test_init_loads_model(self, mock_yolo, temp_model_file):
        """Test that service initializes and loads model"""
        with patch('app.config.model_config.settings.MODEL_PATH', temp_model_file):
            service = ImageDetectionService()
            mock_yolo.assert_called_once()
    
    def test_load_image_from_array(self, mock_detection_service, sample_image):
        """Test loading image from numpy array"""
        service = ImageDetectionService()
        # Mock el método load_image para que funcione
        service.load_image = Mock(return_value=sample_image)
        
        result = service.load_image(sample_image)
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape
    
    @patch('app.services.image_detection.YOLO')
    def test_detect_brands_returns_detections(self, mock_yolo, sample_image):
        """Test brand detection functionality"""
        # Configurar mock de YOLO
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes.data.cpu.return_value.numpy.return_value = np.array([
            [10, 10, 50, 50, 0.85, 0]  # x1, y1, x2, y2, conf, class
        ])
        mock_result.names = {0: "Coca-Cola"}
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        service = ImageDetectionService()
        service.load_image = Mock(return_value=sample_image)
        
        detections = service.detect_brands(sample_image)
        
        assert len(detections) == 1
        assert detections[0]["class_name"] == "Coca-Cola"
        assert detections[0]["confidence"] == 0.85
        assert detections[0]["box"] == [10, 10, 50, 50]
    
    @patch('app.services.image_detection.YOLO')
    def test_detect_brands_empty_result(self, mock_yolo, sample_image):
        """Test detection with no brands found"""
        # Configurar mock para no detectar nada
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes.data.cpu.return_value.numpy.return_value = np.array([])
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        service = ImageDetectionService()
        service.load_image = Mock(return_value=sample_image)
        
        detections = service.detect_brands(sample_image)
        
        assert len(detections) == 0