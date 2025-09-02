import pytest
import os
from pathlib import Path
from app.config.model_config import Settings

class TestConfiguration:
    
    def test_settings_default_values(self):
        """Test default configuration values"""
        settings = Settings()
        
        assert settings.CONFIDENCE_THRESHOLD == 0.5
        assert settings.IOU_THRESHOLD == 0.6
        assert settings.API_PORT == 8000
        assert isinstance(settings.MODEL_PATH, Path)
    
    def test_settings_from_env(self, monkeypatch):
        """Test configuration from environment variables"""
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.7")
        monkeypatch.setenv("API_PORT", "9000")
        
        settings = Settings()
        
        assert settings.CONFIDENCE_THRESHOLD == 0.7
        assert settings.API_PORT == 9000
    
    def test_model_path_exists(self):
        """Test that model path is configured correctly"""
        settings = Settings()
        # No verificamos que el archivo exista, solo que el path es válido
        assert isinstance(settings.MODEL_PATH, Path)
        assert str(settings.MODEL_PATH).endswith('.pt')