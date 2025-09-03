import sys
import os
from pathlib import Path
import pytest
import asyncio
import tempfile
from unittest.mock import Mock, patch
import cv2
import numpy as np

# Agregar el directorio raíz del proyecto al PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def client():
    """FastAPI test client - will be created per test when needed"""
    # Usar lazy loading para evitar importar la app inmediatamente
    from fastapi.testclient import TestClient
    
    # Mock dependencies antes de importar
    with patch.dict('sys.modules', {
        'supabase': Mock(),
        'huggingface_hub': Mock(),
        'ultralytics': Mock(),
    }):
        try:
            from app.main import app
            return TestClient(app)
        except Exception as e:
            pytest.skip(f"Cannot load FastAPI app: {e}")

@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Crear imagen de prueba de 100x100 pixels
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Agregar algo de contenido
    cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
    return img

@pytest.fixture
def sample_image_file(sample_image):
    """Create a temporary image file"""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, sample_image)
        yield tmp.name
    os.unlink(tmp.name)

@pytest.fixture
def mock_detection_service():
    """Mock detection service for testing"""
    service = Mock()
    service.detect_brands.return_value = [
        {
            "class_name": "Coca-Cola",
            "confidence": 0.85,
            "box": [10, 10, 50, 50],
            "class_id": 0
        }
    ]
    return service

@pytest.fixture
def sample_detection_result():
    """Sample detection result for testing"""
    return {
        "detections": [
            {
                "class_name": "Coca-Cola",
                "confidence": 0.85,
                "box": [10, 10, 50, 50],
                "class_id": 0
            },
            {
                "class_name": "Pepsi",
                "confidence": 0.78,
                "box": [60, 60, 100, 100],
                "class_id": 1
            }
        ],
        "database_status": "saved successfully"
    }

@pytest.fixture
def temp_model_file():
    """Create a temporary model file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        # Crear un archivo vacío que simule un modelo
        tmp.write(b"fake model data")
        yield tmp.name
    os.unlink(tmp.name)
