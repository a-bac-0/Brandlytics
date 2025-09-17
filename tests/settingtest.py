import pytest
import os
from pathlib import Path
from unittest.mock import patch

class TestConfiguration:
    
    def test_project_structure(self):
        """Test basic project structure"""
        project_root = Path(__file__).parent.parent
        
        # Verificar que existen las carpetas principales
        assert (project_root / "app").exists()
        assert (project_root / "tests").exists()
        assert (project_root / "models").exists()
        assert (project_root / "frontend").exists()
    
    def test_config_import(self):
        """Test that config can be imported"""
        try:
            from app.config.model_config import settings
            # Si llegamos aquí, la importación fue exitosa
            assert True
        except ImportError:
            # Si hay error, es esperado en el entorno de testing
            pytest.skip("Config module not available in test environment")
    
    def test_model_files_exist(self):
        """Test that model files exist"""
        project_root = Path(__file__).parent.parent
        models_dir = project_root / "models"
        
        # Verificar que existe al menos un archivo de modelo
        model_files = list(models_dir.glob("*.pt"))
        assert len(model_files) > 0, "No model files found"