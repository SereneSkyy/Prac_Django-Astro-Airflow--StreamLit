import requests

DJANGO_BASE_URL = "http://127.0.0.1:8000/api/retrieve"

def preview_data(topic: str):
    """Fetches data by topic (Raw)."""
    response = requests.get(f"{DJANGO_BASE_URL}/{topic}")
    return response.json()

def preview_lang_data(lang):
    """Fetches CLEANED data by language (NLP)."""
    # This hits your backend/api/retrieve/cmtsep/<lang>
    response = requests.get(f"{DJANGO_BASE_URL}/cmtsep/{lang}")
    if response.status_code == 200:
        return response.json()
    return []