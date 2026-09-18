"""
Unit and Integration Tests for BiLSTM Deep Learning Model (Phase 9).
"""

import json
import os
from pathlib import Path
import numpy as np
import pytest

from src.models.bilstm import BiLSTMModel, build_bilstm_model


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def model_path(project_root) -> Path:
    return project_root / "models" / "bilstm" / "best_model.keras"


@pytest.fixture(scope="module")
def metadata_path(project_root) -> Path:
    return project_root / "models" / "bilstm" / "metadata.json"


@pytest.fixture(scope="module")
def test_metrics_path(project_root) -> Path:
    return project_root / "results" / "metrics" / "bilstm_test.json"


@pytest.fixture(scope="module")
def validation_metrics_path(project_root) -> Path:
    return project_root / "results" / "metrics" / "bilstm_validation.json"


@pytest.fixture(scope="module")
def history_path(project_root) -> Path:
    return project_root / "results" / "metrics" / "bilstm_history.csv"


@pytest.fixture(scope="module")
def report_path(project_root) -> Path:
    return project_root / "results" / "metrics" / "bilstm_classification_report.csv"


@pytest.fixture(scope="module")
def cm_plot_path(project_root) -> Path:
    return project_root / "results" / "confusion_matrices" / "bilstm_confusion_matrix.png"


@pytest.fixture(scope="module")
def history_plot_path(project_root) -> Path:
    return project_root / "results" / "graphs" / "bilstm_training_history.png"


@pytest.fixture(scope="module")
def misclass_path(project_root) -> Path:
    return project_root / "results" / "metrics" / "bilstm_misclassifications.csv"


def test_bilstm_model_file_exists(model_path):
    """Test that serialized Keras model exists and is non-empty."""
    assert model_path.exists(), f"Model file not found at {model_path}"
    assert os.path.getsize(model_path) > 0, "Model file is empty"


def test_bilstm_metadata_exists_and_valid(metadata_path):
    """Test that model metadata JSON exists and contains required fields."""
    assert metadata_path.exists(), f"Metadata file not found at {metadata_path}"
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["model_name"] == "BiLSTM"
    assert meta["number_of_features"] == 46
    assert meta["number_of_classes"] == 34
    assert meta["random_seed"] == 42
    assert meta["optimizer"] == "adam"
    assert "test_metrics" in meta
    assert meta["test_metrics"]["accuracy"] > 0.50


def test_bilstm_model_can_be_loaded(model_path):
    """Test that Keras model can be successfully loaded via BiLSTMModel wrapper."""
    bilstm = BiLSTMModel()
    bilstm.load(model_path)
    assert bilstm.model is not None


def test_bilstm_architecture_shapes():
    """Test freshly built BiLSTM architecture input and output tensor dimensions."""
    model = build_bilstm_model(input_shape=(46, 1), num_classes=34)
    assert model.input_shape == (None, 46, 1), f"Expected (None, 46, 1), got {model.input_shape}"
    assert model.output_shape == (None, 34), f"Expected (None, 34), got {model.output_shape}"


def test_bilstm_predictions_shape_and_validity(model_path):
    """Test that model predictions have correct shape and valid class bounds."""
    bilstm = BiLSTMModel()
    bilstm.load(model_path)

    # 20 synthetic samples with 46 features reshaped to (20, 46, 1) or 2D (20, 46)
    dummy_x = np.random.randn(20, 46).astype(np.float32)
    preds = bilstm.predict(dummy_x)

    assert preds.shape == (20,), f"Expected shape (20,), got {preds.shape}"
    assert np.all(preds >= 0) and np.all(preds < 34), "Predicted labels out of [0, 33] bound"


def test_bilstm_predict_proba_validity(model_path):
    """Test that predict_proba outputs valid probability distributions summing to 1.0."""
    bilstm = BiLSTMModel()
    bilstm.load(model_path)

    dummy_x = np.random.randn(10, 46, 1).astype(np.float32)
    probs = bilstm.predict_proba(dummy_x)

    assert probs.shape == (10, 34), f"Expected shape (10, 34), got {probs.shape}"
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-3, err_msg="Probabilities must sum to 1.0")


def test_bilstm_metrics_artifacts_exist(
    test_metrics_path,
    validation_metrics_path,
    history_path,
    report_path,
    misclass_path
):
    """Test that metric JSON and CSV files exist and contain valid results."""
    assert test_metrics_path.exists(), "Test metrics JSON missing"
    assert validation_metrics_path.exists(), "Validation metrics JSON missing"
    assert history_path.exists(), "History CSV missing"
    assert report_path.exists(), "Per-class report CSV missing"
    assert misclass_path.exists(), "Misclassifications CSV missing"

    with open(test_metrics_path, "r", encoding="utf-8") as f:
        t_meta = json.load(f)
    assert "accuracy" in t_meta
    assert "macro_f1" in t_meta
    assert "weighted_f1" in t_meta
    assert t_meta["accuracy"] > 0.50


def test_bilstm_visual_artifacts_exist(cm_plot_path, history_plot_path):
    """Test that confusion matrix and history plot images exist."""
    assert cm_plot_path.exists(), "Confusion matrix plot missing"
    assert history_plot_path.exists(), "History plot missing"
