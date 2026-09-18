"""Streamlit Community Cloud entrypoint; local capture and training stay local."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[1] / 'streamlit_app/app.py'),
               init_globals={'IDS_CLOUD_MODE': True}, run_name='__main__')
