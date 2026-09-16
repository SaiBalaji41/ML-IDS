"""
Unit and Integration Tests for XGBoost Baseline Model (Phase 7).
"""

import json
import os
from pathlib import Path
import numpy as np
import pytest

from src.models.xgboost_model import XGBoostBaselineModel


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def model_path(project_root) -> Path:
    return project_root / "models" / "xgboost.pkl"


@pytest.fixture(scope="module")
def metadata_path(project_root) -> Path:
    return project_root / "models" / "xgboost_metadata.json"


@pytest.fixture(scope="module")
def test_metrics_path(project_root) -> Path:
    return project_root / "results" / "metrics" / "xgboost_test.json"


@pytest.fixture(scope="module")
def validation_metrics_path(project_root) -> Path:
    return project_root / "results" / "metrics" / "xgboost_validation.json"


@pytest.fixture(scope="module")
def report_path(project_root) -> Path:
    return project_root / "results" / "metrics" / "xgboost_classification_report.csv"


@pytest.fixture(scope="module")
def feat_imp_path(project_root) -> Path:
    return project_root / "results" / "metrics" / "xgboost_feature_importance.csv"


@pytest.fixture(scope="module")
def cm_plot_path(project_root) -> Path:
    return project_root / "results" / "confusion_matrices" / "xgboost_confusion_matrix.png"


@pytest.fixture(scope="module")
def feat_plot_path(project_root) -> Path:
    return project_root / "results" / "graphs" / "xgboost_feature_importance.png"


def test_xgboost_model_file_exists(model_path):
    """Test that serialized XGBoost model exists and is non-empty."""
    assert model_path.exists(), f"Model file not found at {model_path}"
    assert os.path.getsize(model_path) > 0, "Model file is empty"


def test_xgboost_metadata_exists_and_valid(metadata_path):
    """Test that model metadata JSON exists and contains required fields."""
    assert metadata_path.exists(), f"Metadata file not found at {metadata_path}"
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["model_type"] == "XGBoost"
    assert meta["number_of_features"] == 46
    assert meta["number_of_classes"] == 34
    assert meta["random_seed"] == 42
    assert "hyperparameters" in meta
    assert meta["hyperparameters"]["n_estimators"] == 100
    assert "test_metrics" in meta
    assert meta["test_metrics"]["accuracy"] > 0.90


def test_xgboost_model_can_be_loaded(model_path):
    """Test that model can be successfully loaded."""
    xgb_model = XGBoostBaselineModel()
    xgb_model.load(model_path)
    assert xgb_model.model is not None


def test_xgboost_predictions_shape_and_validity(model_path):
    """Test that model predictions have correct shape and valid class bounds."""
    xgb_model = XGBoostBaselineModel()
    xgb_model.load(model_path)

    # 20 synthetic samples with 46 features
    dummy_x = np.random.randn(20, 46).astype(np.float32)
    preds = xgb_model.predict(dummy_x)

    assert preds.shape == (20,), f"Expected shape (20,), got {preds.shape}"
    assert np.all(preds >= 0) and np.all(preds < 34), "Predicted labels out of [0, 33] bound"


def test_xgboost_predict_proba_validity(model_path):
    """Test that predict_proba outputs valid probability distributions summing to 1."""
    xgb_model = XGBoostBaselineModel()
    xgb_model.load(model_path)

    dummy_x = np.random.randn(10, 46).astype(np.float32)
    probs = xgb_model.predict_proba(dummy_x)

    assert probs.shape == (10, 34), f"Expected shape (10, 34), got {probs.shape}"
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-3, err_msg="Probabilities must sum to 1.0")


def test_xgboost_metrics_artifacts_exist(test_metrics_path, validation_metrics_path, report_path, feat_imp_path):
    """Test that metric JSON and CSV files exist and contain valid results."""
    assert test_metrics_path.exists(), "Test metrics JSON missing"
    assert validation_metrics_path.exists(), "Validation metrics JSON missing"
    assert report_path.exists(), "Per-class report CSV missing"
    assert feat_imp_path.exists(), "Feature importance CSV missing"

    with open(test_metrics_path, "r", encoding="utf-8") as f:
        t_meta = json.load(f)
    assert "accuracy" in t_meta
    assert "macro_f1" in t_meta
    assert "weighted_f1" in t_meta
    assert t_meta["accuracy"] > 0.90


def test_xgboost_visual_artifacts_exist(cm_plot_path, feat_plot_path):
    """Test that confusion matrix and feature importance plot images exist."""
    assert cm_plot_path.exists(), "Confusion matrix plot missing"
    assert feat_plot_path.exists(), "Feature importance plot missing"
