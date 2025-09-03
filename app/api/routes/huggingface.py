"""
API routes for Hugging Face Hub model management.
"""

from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from datetime import datetime

from app.services.huggingface_service import get_hf_model_service
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/huggingface", tags=["huggingface"])


class ModelInfo(BaseModel):
    id: str
    downloads: int
    likes: int
    tags: List[str]
    created_at: Optional[str] = None
    last_modified: Optional[str] = None


class ModelUploadRequest(BaseModel):
    local_path: str
    repo_id: str
    commit_message: str = None


def format_model_data(model_data):
    """Convert model data to API-compatible format."""
    formatted = model_data.copy()
    
    # Convert datetime objects to strings
    if 'created_at' in formatted and formatted['created_at']:
        if isinstance(formatted['created_at'], datetime):
            formatted['created_at'] = formatted['created_at'].isoformat()
    
    if 'last_modified' in formatted and formatted['last_modified']:
        if isinstance(formatted['last_modified'], datetime):
            formatted['last_modified'] = formatted['last_modified'].isoformat()
    
    return formatted


@router.get("/models", response_model=List[ModelInfo])
async def list_organization_models():
    """List all models in the CV-Brandlytics organization."""
    try:
        hf_service = get_hf_model_service()
        models = hf_service.list_organization_models()
        
        # Format the model data
        formatted_models = [format_model_data(model) for model in models]
        return formatted_models
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/models/{model_id:path}")
async def get_model_info(model_id: str):
    """Get detailed information about a specific model."""
    try:
        hf_service = get_hf_model_service()
        model_info = hf_service.get_model_info(model_id)
        
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        return model_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@router.post("/models/{model_id:path}/download")
async def download_model(
    model_id: str,
    filename: str = Query(None, description="Specific file to download (optional)")
):
    """Download a model from Hugging Face Hub."""
    try:
        hf_service = get_hf_model_service()
        model_path = hf_service.download_model(model_id, filename)
        
        return {
            "status": "success",
            "message": f"Model downloaded successfully",
            "model_id": model_id,
            "local_path": model_path,
            "filename": filename
        }
    except Exception as e:
        logger.error(f"Failed to download model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")


@router.post("/models/upload")
async def upload_model(request: ModelUploadRequest):
    """Upload a local model to Hugging Face Hub."""
    try:
        hf_service = get_hf_model_service()
        success = hf_service.upload_model(
            local_model_path=request.local_path,
            repo_id=request.repo_id,
            commit_message=request.commit_message
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Model uploaded successfully",
                "repo_id": request.repo_id,
                "local_path": request.local_path
            }
        else:
            raise HTTPException(status_code=500, detail="Upload failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload model: {str(e)}")


@router.get("/models/available")
async def get_available_models():
    """Get all available models with detailed information."""
    try:
        hf_service = get_hf_model_service()
        models = hf_service.get_available_models()
        return models
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get available models: {str(e)}")


@router.delete("/cache")
async def clear_model_cache():
    """Clear the Hugging Face model cache."""
    try:
        hf_service = get_hf_model_service()
        success = hf_service.clear_cache()
        
        if success:
            return {"status": "success", "message": "Model cache cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear cache")
            
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get("/config")
async def get_huggingface_config():
    """Get current Hugging Face configuration."""
    from app.config.model_config import settings
    
    return {
        "hf_org": settings.HF_ORG,
        "hf_model_repo": settings.HF_MODEL_REPO,
        "use_hf_model": settings.USE_HF_MODEL,
        "hf_cache_dir": settings.HF_CACHE_DIR,
        "has_token": bool(settings.HF_TOKEN)
    }
