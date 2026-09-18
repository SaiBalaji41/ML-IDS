"""
Global Pytest Configuration and Environment Initialization.
Ensures proper module import ordering and platform stability on Windows.
"""

import os
import sys
from pathlib import Path

# Disable oneDNN noisy warnings and set log level
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Safe Windows WMI workaround for Python 3.12 platform.uname() bug
import platform

def _safe_get_machine_win32():
    return os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64")

platform._get_machine_win32 = _safe_get_machine_win32

# Pre-import numpy and pandas to prevent Keras 3 circular C-extension import on Windows
import numpy as np
import pandas as pd
import sklearn
import xgboost

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
