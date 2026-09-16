"""
Unit tests for Random Forest baseline model interface and predictions.
"""

import pytest

try:
    import numpy as np
    from src.models.random_forest import RandomForestBaselineModel
    NUMPY_AVAILABLE = True
except (ImportError, Exception):
    NUMPY_AVAILABLE = False


def test_random_forest_initialization():
    """Test model hyperparameter initialization."""
    if not NUMPY_AVAILABLE:
        pytest.skip("NumPy / Scikit-Learn not accessible in current environment")
    rf = RandomForestBaselineModel(n_estimators=10, random_state=42)
    assert rf.n_estimators == 10
    assert rf.random_state == 42
    assert rf.model is None


def test_random_forest_fit_predict(tmp_path):
    """Test training, prediction, and serialization interface with mock data."""
    if not NUMPY_AVAILABLE:
        pytest.skip("NumPy / Scikit-Learn not accessible in current environment")
    np.random.seed(42)
    X = np.random.randn(50, 5).astype(np.float32)
    y = np.random.randint(0, 3, size=50)

    rf = RandomForestBaselineModel(n_estimators=10, random_state=42)
    rf.fit(X, y)
    preds = rf.predict(X)
    probs = rf.predict_proba(X)

    assert preds.shape == (50,)
    assert probs.shape == (50, 3)

    # Test feature importance
    feat_names = [f"f_{i}" for i in range(5)]
    importances = rf.get_feature_importances(feature_names=feat_names)
    assert len(importances) == 5

    # Test model save and load
    save_file = tmp_path / "rf_test.pkl"
    rf.save(save_file)
    assert save_file.exists()

    rf_loaded = RandomForestBaselineModel()
    rf_loaded.load(save_file)
    preds_loaded = rf_loaded.predict(X)
    assert np.array_equal(preds, preds_loaded)
