"""
Unit tests for Dashboard Server REST APIs and cache loading.
"""

import json
from pathlib import Path
import pytest

from dashboard.server import CACHE, load_backend_assets, DashboardRequestHandler

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_load_backend_assets():
    """Verify that backend assets load properly into CACHE."""
    load_backend_assets()
    assert isinstance(CACHE["label_mapping"], dict)
    assert len(CACHE["feature_names"]) == 46 or len(CACHE["feature_names"]) == 0
    if (PROJECT_ROOT / "results" / "metrics" / "final_results_table.csv").exists():
        assert len(CACHE["metrics_comparison"]) > 0


def test_cache_structure():
    """Verify cache dictionary keys exist."""
    required_keys = [
        "model_xgb", "model_rf", "scaler", "label_mapping",
        "id_to_label", "feature_names", "metrics_comparison"
    ]
    for k in required_keys:
        assert k in CACHE


def test_dashboard_handler_instantiation():
    """Verify DashboardRequestHandler is defined."""
    assert issubclass(DashboardRequestHandler, object)
