import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import io
from PIL import Image

class TestDetectionAPI:
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "Bienvenido a la API de Brandlytics" in response.json()["message"]
    
    @patch('app.services.image_detection.ImageDetectionService')
    def test_process_image_upload(self, mock_service, client, sample_image_file):
        """Test image upload processing"""
        # Configurar mock
        mock_service.return_value.detect_brands.return_value = [
            {
                "class_name": "Coca-Cola",
                "confidence": 0.85,
                "box": [10, 10, 50, 50],
                "class_id": 0
            }
        ]
        
        # Simular subida de archivo
        with open(sample_image_file, "rb") as f:
            response = client.post(
                "/api/detection/process-image",
                files={"image_file": ("test.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "detections" in data
        assert len(data["detections"]) > 0
        assert data["detections"][0]["class_name"] == "Coca-Cola"
    
    @patch('app.services.image_detection.ImageDetectionService')
    def test_process_image_url(self, mock_service, client):
        """Test image URL processing"""
        # Configurar mock
        mock_service.return_value.detect_brands.return_value = [
            {
                "class_name": "Pepsi",
                "confidence": 0.78,
                "box": [20, 20, 60, 60],
                "class_id": 1
            }
        ]
        
        response = client.post(
            "/api/detection/process-image",
            data={"image_url": "https://example.com/test.jpg"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "detections" in data
    
    def test_process_image_no_input(self, client):
        """Test processing without image input"""
        response = client.post("/api/detection/process-image")
        
        assert response.status_code == 400
        assert "No image provided" in response.json()["detail"]
    
    def test_process_video_endpoint_exists(self, client):
        """Test that video endpoint exists"""
        # Crear archivo de video falso
        fake_video = io.BytesIO(b"fake video content")
        
        response = client.post(
            "/api/detection/process-video",
            files={"video": ("test.mp4", fake_video, "video/mp4")}
        )
        
        # Debería devolver error 422 o similar, pero no 404
        assert response.status_code != 404