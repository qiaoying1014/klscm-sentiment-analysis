"""Streamlit Community Cloud entry point for the frozen research dashboard."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["KLSCM_DASHBOARD_BUNDLE"] = str(ROOT / "deploy" / "dashboard_data")
runpy.run_path(str(ROOT / "absa_dashboard.py"), run_name="__main__")
