# src/app/database/connection.py
import os
from supabase import create_client
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Falta SUPABASE_URL o SUPABASE_KEY en el .env")

# Crear cliente Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
