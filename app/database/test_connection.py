# src/app/database/test_connection.py
from connection import supabase

# Intentar listar los buckets existentes
buckets = supabase.storage.list_buckets()
print("Buckets existentes:", buckets)
