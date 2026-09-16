"""
Unit tests for XGBoost baseline model interface and predictions.
"""

import pytest

try:
    import numpy as np
    from src.models.xgboost_model import XGBoostBaselineModel
    XGB_AVAILABLE = True
except (ImportError, Exception):
    XGB_AVAILABLE = False


def test_xgboost_initialization():
    """Test model hyperparameter initialization."""
    if not XGB_AVAILABLE:
        pytest.skip("XGBoost / NumPy not accessible in current environment")
    xgb = XGBoostBaselineModel(n_estimators=10, max_depth=3, random_state=42)
    assert xgb.n_estimators == 10
    assert xgb.max_depth == 3
    assert xgb.random_state == 42
    assert xgb.model is None


def test_xgboost_fit_predict(tmp_path):
    """Test training, prediction, and serialization interface with mock data."""
    if not XGB_AVAILABLE:
        pytest.skip("XGBoost / NumPy not accessible in current environment")
    np.random.seed(42)
    X = np.random.randn(50, 5).astype(np.float32)
    y = np.random.randint(0, 3, size=50)

    xgb = XGBoostBaselineModel(n_estimators=10, max_depth=3, random_state=42)
    xgb.fit(X, y)
    preds = xgb.predict(X)
    probs = xgb.predict_proba(X)

    assert preds.shape == (50,)
    assert probs.shape == (50, 3)

    # Test feature importance
    feat_names = [f"f_{i}" for i in range(5)]
    importances = xgb.get_feature_importances(feature_names=feat_names, importance_type="gain")
    assert len(importances) == 5

    # Test model save and load
    save_file = tmp_path / "xgb_test.pkl"
    xgb.save(save_file)
    assert save_file.exists()

    xgb_loaded = XGBoostBaselineModel()
    xgb_loaded.load(save_file)
    preds_loaded = xgb_loaded.predict(X)
    assert np.array_equal(preds, preds_loaded)
