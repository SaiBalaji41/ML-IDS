"""
Sentinel ML-IDS: Streamlit Web Dashboard & SOC Control Room.
1:1 Pixel-Perfect SOC Dashboard with Hybrid 1D-CNN + BiLSTM Deep Learning Intelligence.
Comprehensive 5-Tab Architecture:
- 📊 Overview & Live Stream
- ⚡ Custom Input Studio
- 📈 Model Benchmarks (5 Models)
- 🧠 Architecture & Pipeline (6-Stage Methodology)
- 🔍 SHAP & Threat Glossary (34 Classes)
"""

import html
import io
import json
from pathlib import Path
import sys
import time
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BYTE_LENGTH = 1024
CLOUD_MODE = bool(globals().get('IDS_CLOUD_MODE'))

st.set_page_config(
    page_title='Sentinel ML-IDS — Intelligent Network Security',
    page_icon='🛡️',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# Load custom styles
styles_path = Path(__file__).parent / 'styles.css'
if styles_path.exists():
    styles_css = styles_path.read_text(encoding='utf-8')
    st.html(f'<style>{styles_css}</style>')

# Build Pixel-Perfect Bundled HTML
dashboard_dir = ROOT / 'dashboard'
bundled_html = ''
if dashboard_dir.exists():
    try:
        html_raw = (dashboard_dir / 'index.html').read_text(encoding='utf-8')
        css_raw = (dashboard_dir / 'styles.css').read_text(encoding='utf-8')
        js_raw = (dashboard_dir / 'app.js').read_text(encoding='utf-8')
        bundled_html = html_raw.replace(
            '<link rel="stylesheet" href="styles.css">',
            f'<style>{css_raw}</style>',
        ).replace(
            '<script src="app.js"></script>',
            f'<script>{js_raw}</script>',
        )
    except Exception as e:
        bundled_html = ''


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


# Initialize Default Session State
for key, value in {
    'events': [],
    'total_packets': 30,
    'attack_packets': 30,
    'review_packets': 6,
    'streaming': False,
    'batches': [
        {'batch': '1', 'Benign': 0, 'Attack': 10},
        {'batch': '2', 'Benign': 0, 'Attack': 10},
        {'batch': '3', 'Benign': 0, 'Attack': 10},
    ],
    'packet_payload': None,
    'packet_description': 'No packet selected',
    'model_directory': str(ROOT / 'models/pytorch_packet_ids'),
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==============================================================================
# MAIN VIEW: 1:1 PIXEL-PERFECT EMBEDDED DASHBOARD
# ==============================================================================
if bundled_html:
    components.html(bundled_html, height=1300, scrolling=True)
else:
    st.error("Dashboard assets in 'dashboard/' could not be loaded. Please ensure index.html, styles.css, and app.js exist.")
