import streamlit as st
import requests

DJANGO_INGEST_API = "http://127.0.0.1:8000/api/ingest"

trigger_btn = st.button('Trigger Tasks', on_click=lambda: st.toast(":green[Toast Msg]", duration=2))  

if trigger_btn:
    try:
        result = requests.post(DJANGO_INGEST_API)
        st.write(result.json())
    except Exception as e:
        st.error(e)