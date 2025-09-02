from .connection import get_supabase
from app.api.schemas.schemas_detection import DetectionCreate
from typing import List
from app.config.model_config import settings

def save_detections(detections_data: List[DetectionCreate]):

    supabase = get_supabase()
    data_to_insert = [d.model_dump() for d in detections_data]
    
    try:
        response = supabase.from_("detections").insert(data_to_insert).execute()
        
        if response.data is None and response.error is not None:
             raise Exception(f"Error al guardar en Supabase: {response.error.message}")

        print(f"Se guardaron {len(response.data)} detecciones en Supabase.")
        return response.data
    except Exception as e:
        print(f"Error en la operación de base de datos: {e}")
        raise e



def get_video_analytics(video_name: str):
    """Get analytics for a specific video from the database."""
    supabase = get_supabase()
    
    try:
        response = supabase.from_("detections").select("*").eq("video_name", video_name).execute()
        
        if not response.data:
            return {"message": f"No data found for video: {video_name}"}
        
        detections = response.data
        
        # Calculate analytics
        total_detections = len(detections)
        unique_brands = list(set(d['brand_name'] for d in detections))
        avg_confidence = sum(d['confidence'] for d in detections) / total_detections if total_detections > 0 else 0
        
        # Brand breakdown
        brand_counts = {}
        for detection in detections:
            brand = detection['brand_name']
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        return {
            "video_name": video_name,
            "total_detections": total_detections,
            "unique_brands": unique_brands,
            "avg_confidence": avg_confidence,
            "brand_breakdown": brand_counts,
            "sample_detections": detections[:10]  # First 10 as examples
        }
        
    except Exception as e:
        return {"error": f"Failed to get analytics: {str(e)}"}


def upload_public_bytes(bucket: str, path: str, content: bytes, content_type: str = "image/jpeg") -> str:
    supabase = get_supabase()
    resp = supabase.storage.from_(bucket).upload(
        path=path,
        file=content,
        file_options={"content-type": content_type, "x-upsert": "true"}   # <- FIX: content-type
    )
    if isinstance(resp, dict) and resp.get("error"):
        raise Exception(str(resp["error"]))
    if hasattr(resp, "error") and resp.error:
        raise Exception(str(resp.error))

    pub = supabase.storage.from_(bucket).get_public_url(path)
    if isinstance(pub, str):
        return pub
    url = getattr(pub, "public_url", None)
    if not url and isinstance(pub, dict):
        url = pub.get("publicUrl") or pub.get("public_url") or pub.get("data", {}).get("publicUrl")
    if not url:
        base = settings.SUPABASE_URL.rstrip("/")
        url = f"{base}/storage/v1/object/public/{bucket}/{path}"
    return url