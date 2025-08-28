from .connection import get_supabase
from app.api.schemas.schemas_detection import DetectionCreate
from typing import List

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
