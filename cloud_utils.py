import os
import streamlit as st
from supabase import create_client
from pinecone import Pinecone

# Use st.secrets for Streamlit Cloud, os.environ for local dev
def get_secret(key):
    return st.secrets.get(key) or os.getenv(key)

# Initialize Clients
sb_client = create_client(get_secret("SUPABASE_URL"), get_secret("SUPABASE_KEY"))
pc = Pinecone(api_key=get_secret("PINECONE_API_KEY"))
index_name = "ipo-rhp-index"

# --- SUPABASE STORAGE (PDFs) ---
def get_pdf_from_cloud(ipo_name: str):
    safe_name = ipo_name.replace(" ", "_")
    try:
        # Check if file exists
        data = sb_client.storage.from_("rhp-pdfs").download(f"{safe_name}.pdf")
        local_path = f"/tmp/{safe_name}.pdf"
        with open(local_path, "wb") as f:
            f.write(data)
        return local_path
    except:
        return None

def upload_pdf_to_cloud(ipo_name: str, local_path: str):
    safe_name = ipo_name.replace(" ", "_")
    try:
        with open(local_path, "rb") as f:
        sb_client.storage.from_("rhp-pdfs").upload(
            path=f"{safe_name}.pdf", 
            file=f, 
            file_options={"content-type": "application/pdf", "upsert": "true"}
        )
        print(f"✅ Uploaded {safe_name}.pdf to Supabase")
    except Exception as e:
        # This will prevent the app from crashing even if upload fails
        print(f"⚠️ Cloud Storage Upload Error: {e}")

# --- PINECONE (Vectors) ---
def namespace_exists(ipo_name: str):
    idx = pc.Index(index_name)
    stats = idx.describe_index_stats()
    return ipo_name in stats.get("namespaces", {})

# --- SUPABASE DB (JSON Analysis) ---
def get_cached_profile(ipo_name: str):
    res = sb_client.table("ipo_profiles").select("profile").eq("ipo_name", ipo_name).execute()
    return res.data[0]["profile"] if res.data else None

def save_profile_to_cache(ipo_name: str, profile_dict: dict):
    sb_client.table("ipo_profiles").upsert({"ipo_name": ipo_name, "profile": profile_dict}).execute()
