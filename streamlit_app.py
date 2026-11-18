"""Streamlit Cloud entry point: starts the FastAPI app in a background thread, then shows the dashboard."""
import os
import runpy
import threading
import time
import urllib.request

import streamlit as st
import uvicorn

os.environ["API_URL"] = "http://127.0.0.1:8000"


@st.cache_resource
def start_api() -> bool:
    from src.api import app
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(120):                       # wait up to ~60 s for the models to load
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


start_api()
runpy.run_path("dashboard/app.py", run_name="__main__")
