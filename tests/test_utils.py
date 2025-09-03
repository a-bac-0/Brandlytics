import pytest
import os
import tempfile
from pathlib import Path
import cv2
import numpy as np

class TestUtilities:
    
    def test_image_processing_basic(self, sample_image):
        """Test basic image processing operations"""
        # Test resizing
        resized = cv2.resize(sample_image, (50, 50))
        assert resized.shape == (50, 50, 3)
        
        # Test color conversion
        gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        assert len(gray.shape) == 2
        assert gray.shape == (100, 100)
    
    def test_file_operations(self, sample_image_file):
        """Test file operations with images"""
        # Verificar que el archivo existe
        assert os.path.exists(sample_image_file)
        
        # Leer la imagen
        img = cv2.imread(sample_image_file)
        assert img is not None
        assert isinstance(img, np.ndarray)
    
    def test_directory_structure(self):
        """Test project directory structure"""
        project_root = Path(__file__).parent.parent
        
        # Verificar directorios críticos
        critical_dirs = [
            "app",
            "app/api",
            "app/services",
            "app/config",
            "models",
            "tests"
        ]
        
        for dir_name in critical_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Directory {dir_name} does not exist"
            assert dir_path.is_dir(), f"{dir_name} is not a directory"
    
    def test_model_files_basic(self):
        """Test model files exist and have reasonable sizes"""
        project_root = Path(__file__).parent.parent
        models_dir = project_root / "models"
        
        model_files = list(models_dir.glob("*.pt"))
        assert len(model_files) > 0, "No .pt model files found"
        
        for model_file in model_files:
            # Verificar que el archivo no está vacío
            assert model_file.stat().st_size > 0, f"Model file {model_file.name} is empty"
    
    def test_config_files_exist(self):
        """Test that configuration files exist"""
        project_root = Path(__file__).parent.parent
        
        config_files = [
            "pytest.ini",
            "requirements.txt",
            ".env.example"
        ]
        
        for config_file in config_files:
            file_path = project_root / config_file
            assert file_path.exists(), f"Config file {config_file} does not exist"
    
    def test_temp_file_creation(self):
        """Test temporary file creation and cleanup"""
        with tempfile.NamedTemporaryFile(suffix='.test', delete=False) as tmp:
            test_content = b"test data"
            tmp.write(test_content)
            tmp_path = tmp.name
        
        # Verificar que el archivo se creó
        assert os.path.exists(tmp_path)
        
        # Leer y verificar contenido
        with open(tmp_path, 'rb') as f:
            assert f.read() == test_content
        
        # Limpiar
        os.unlink(tmp_path)
        assert not os.path.exists(tmp_path)
    
    def test_numpy_operations(self, sample_image):
        """Test basic numpy operations on images"""
        # Test shape operations
        height, width, channels = sample_image.shape
        assert height == 100
        assert width == 100
        assert channels == 3
        
        # Test array operations
        mean_value = np.mean(sample_image)
        assert 0 <= mean_value <= 255
        
        # Test array slicing
        roi = sample_image[20:80, 20:80]
        assert roi.shape == (60, 60, 3)
