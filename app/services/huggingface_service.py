"""
Hugging Face Hub integration service for Brandlytics.
Handles model downloading, caching, and management from HF Hub.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError
from ultralytics import YOLO

from app.config.model_config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HuggingFaceModelService:
    """Service for managing models from Hugging Face Hub."""
    
    def __init__(self):
        self.api = HfApi(token=settings.HF_TOKEN)
        self.cache_dir = Path(settings.HF_CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
    def list_organization_models(self) -> List[Dict]:
        """List all models in the CV-Brandlytics organization."""
        try:
            models = list(self.api.list_models(author=settings.HF_ORG))
            model_list = []
            
            for model in models:
                model_info = {
                    'id': model.id,
                    'downloads': getattr(model, 'downloads', 0),
                    'likes': getattr(model, 'likes', 0),
                    'tags': getattr(model, 'tags', []),
                    'created_at': getattr(model, 'created_at', None),
                    'last_modified': getattr(model, 'last_modified', None)
                }
                model_list.append(model_info)
                
            logger.info(f"Found {len(model_list)} models in {settings.HF_ORG} organization")
            return model_list
            
        except Exception as e:
            logger.error(f"Failed to list organization models: {e}")
            return []
    
    def download_model(self, repo_id: str, filename: str = None) -> str:
        """
        Download a model from Hugging Face Hub.
        
        Args:
            repo_id: Repository ID (e.g., "CV-Brandlytics/ModelM")
            filename: Specific file to download (optional)
            
        Returns:
            Path to downloaded model file
        """
        try:
            if filename:
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=self.cache_dir,
                    token=settings.HF_TOKEN
                )
                logger.info(f"Downloaded {filename} from {repo_id}")
            else:
                model_path = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=self.cache_dir,
                    token=settings.HF_TOKEN
                )
                logger.info(f"Downloaded complete repository from {repo_id}")
                
            return model_path
            
        except HfHubHTTPError as e:
            logger.error(f"HTTP error downloading model {repo_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to download model {repo_id}: {e}")
            raise
    
    def get_model_info(self, repo_id: str) -> Dict:
        """Get detailed information about a model."""
        try:
            model_info = self.api.model_info(repo_id=repo_id, token=settings.HF_TOKEN)
            
            info_dict = {
                'id': model_info.id,
                'downloads': getattr(model_info, 'downloads', 0),
                'likes': getattr(model_info, 'likes', 0),
                'tags': getattr(model_info, 'tags', []),
                'pipeline_tag': getattr(model_info, 'pipeline_tag', None),
                'created_at': getattr(model_info, 'created_at', None),
                'last_modified': getattr(model_info, 'last_modified', None),
                'siblings': [f.rfilename for f in getattr(model_info, 'siblings', [])]
            }
            
            logger.info(f"Retrieved info for model {repo_id}")
            return info_dict
            
        except Exception as e:
            logger.error(f"Failed to get model info for {repo_id}: {e}")
            return {}
    
    def load_yolo_from_hf(self, repo_id: str, model_filename: str = "best.pt") -> YOLO:
        """
        Load a YOLO model from Hugging Face Hub.
        
        Args:
            repo_id: Repository ID
            model_filename: Name of the model file
            
        Returns:
            Loaded YOLO model
        """
        try:
            model_path = self.download_model(repo_id, model_filename)
            
            model = YOLO(model_path)
            logger.info(f"✅ Successfully loaded YOLO model from {repo_id}/{model_filename}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model from {repo_id}: {e}")
            raise
    
    def upload_model(self, local_model_path: str, repo_id: str, commit_message: str = None) -> bool:
        """
        Upload a local model to Hugging Face Hub.
        
        Args:
            local_model_path: Path to local model file or directory
            repo_id: Target repository ID
            commit_message: Commit message for the upload
            
        Returns:
            Success status
        """
        try:
            if not settings.HF_TOKEN:
                raise ValueError("HF_TOKEN is required for uploading models")
            
            local_path = Path(local_model_path)
            if not local_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {local_model_path}")
            
            commit_msg = commit_message or f"Upload model from {local_path.name}"
            
            if local_path.is_file():
                self.api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=local_path.name,
                    repo_id=repo_id,
                    commit_message=commit_msg,
                    token=settings.HF_TOKEN
                )
            else:
                self.api.upload_folder(
                    folder_path=str(local_path),
                    repo_id=repo_id,
                    commit_message=commit_msg,
                    token=settings.HF_TOKEN
                )
            
            logger.info(f"✅ Successfully uploaded {local_model_path} to {repo_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload model to {repo_id}: {e}")
            return False
    
    def get_available_models(self) -> Dict[str, Dict]:
        """Get information about all available models in the organization."""
        models = self.list_organization_models()
        model_dict = {}
        
        for model in models:
            try:
                detailed_info = self.get_model_info(model['id'])
                model_dict[model['id']] = detailed_info
            except Exception as e:
                logger.warning(f"Could not get detailed info for {model['id']}: {e}")
                model_dict[model['id']] = model
        
        return model_dict
    
    def clear_cache(self) -> bool:
        """Clear the HF model cache."""
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True, parents=True)
                logger.info("✅ HF model cache cleared")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to clear cache: {e}")
            return False


# Global service instance
hf_service = HuggingFaceModelService()


def get_hf_model_service() -> HuggingFaceModelService:
    """Get the global HuggingFace service instance."""
    return hf_service
