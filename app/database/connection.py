from supabase import create_client, Client
from app.config.model_config import settings

supabase_client: Client = None

def get_supabase() -> Client:
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
