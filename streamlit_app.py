"""
OmniPattern - Streamlit Cloud Entry Point
Redirects execution to src/web/app.py
"""
import os
import sys
import runpy

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

app_path = os.path.join(ROOT_DIR, "src", "web", "app.py")
runpy.run_path(app_path, run_name="__main__")
