"""
Unit & Regression Test Suite for Phase 12: Confusion Matrix & Error Analysis.
Validates mathematical integrity, matrix dimensions, sample totals, metric bounds,
and artifact completeness across all 5 evaluated architectures.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def paths():
    return {
        "conf_dir": PROJECT_ROOT / "results" / "confusion_matrices",
        "metrics_dir": PROJECT_ROOT / "results" / "metrics",
        "graphs_dir": PROJECT_ROOT / "results" / "graphs" / "error_analysis",
        "docs_dir": PROJECT_ROOT / "docs",
        "label_mapping": PROJECT_ROOT / "data" / "processed" / "label_mapping.json",
    }


@pytest.fixture(scope="module")
def models():
    return ["random_forest", "xgboost", "cnn_1d", "bilstm", "cnn_bilstm"]


def test_label_mapping_integrity(paths):
    """Verify that label mapping file exists and contains exactly 34 valid classes."""
    assert paths["label_mapping"].exists(), "label_mapping.json missing"
    with open(paths["label_mapping"], "r", encoding="utf-8") as f:
        mapping = json.load(f)
    assert len(mapping) == 34, f"Expected 34 classes, found {len(mapping)}"
    assert "BenignTraffic" in mapping, "BenignTraffic class must be present"


def test_confusion_matrices_dimensions_and_totals(paths, models):
    """Verify each model's confusion matrix has shape (34, 34), non-negative values, and sum == 1,176,851."""
    expected_samples = 1176851
    for m in models:
        npy_path = paths["conf_dir"] / f"{m}_cm.npy"
        assert npy_path.exists(), f"Numpy confusion matrix missing for {m}: {npy_path}"
        cm = np.load(npy_path)

        assert cm.shape == (34, 34), f"Matrix shape for {m} is {cm.shape}, expected (34, 34)"
        assert np.all(cm >= 0), f"Negative values found in confusion matrix for {m}"
        assert cm.sum() == expected_samples, f"Matrix sum for {m} is {cm.sum()}, expected {expected_samples}"


def test_visual_confusion_matrix_images_exist(paths, models):
    """Verify raw and normalized confusion matrix PNG plots exist for all 5 models."""
    for m in models:
        raw_png = paths["conf_dir"] / f"{m}_confusion_matrix.png"
        norm_png = paths["conf_dir"] / f"{m}_confusion_matrix_normalized.png"
        assert raw_png.exists(), f"Raw confusion matrix PNG missing: {raw_png}"
        assert norm_png.exists(), f"Normalized confusion matrix PNG missing: {norm_png}"
        assert raw_png.stat().st_size > 1000, f"Raw PNG file corrupted/empty: {raw_png}"
        assert norm_png.stat().st_size > 1000, f"Normalized PNG file corrupted/empty: {norm_png}"


def test_classwise_error_analysis_csv(paths):
    """Verify classwise_error_analysis.csv exists and contains valid OvR metrics."""
    csv_path = paths["metrics_dir"] / "classwise_error_analysis.csv"
    assert csv_path.exists(), "classwise_error_analysis.csv missing"
    df = pd.read_csv(csv_path)

    expected_cols = ["Model", "Class", "Class_ID", "TP", "TN", "FP", "FN", "Support", "Precision", "Recall", "F1-Score"]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column {col} in classwise_error_analysis.csv"

    assert len(df) == 5 * 34, f"Expected 170 rows (5 models * 34 classes), got {len(df)}"
    assert (df["Precision"] >= 0.0).all() and (df["Precision"] <= 1.0).all()
    assert (df["Recall"] >= 0.0).all() and (df["Recall"] <= 1.0).all()
    assert (df["F1-Score"] >= 0.0).all() and (df["F1-Score"] <= 1.0).all()


def test_misclassification_pairs_csv(paths):
    """Verify misclassification_pairs.csv contains valid error counts and percentages."""
    csv_path = paths["metrics_dir"] / "misclassification_pairs.csv"
    assert csv_path.exists(), "misclassification_pairs.csv missing"
    df = pd.read_csv(csv_path)

    assert "Model" in df.columns
    assert "Actual Class" in df.columns
    assert "Predicted Class" in df.columns
    assert "Error Count" in df.columns
    assert (df["Error Count"] > 0).all()
    assert (df["Actual Class"] != df["Predicted Class"]).all()


def test_benign_vs_attack_analysis_csv(paths):
    """Verify benign_vs_attack_analysis.csv partitions errors correctly."""
    csv_path = paths["metrics_dir"] / "benign_vs_attack_analysis.csv"
    assert csv_path.exists(), "benign_vs_attack_analysis.csv missing"
    df = pd.read_csv(csv_path)

    assert len(df) == 5, f"Expected 5 models in benign_vs_attack_analysis.csv, got {len(df)}"
    assert (df["Benign Support"] == 27709).all(), "Benign support mismatch"
    assert (df["Attack Support"] == 1149142).all(), "Attack support mismatch"
    assert (df["Benign Support"] + df["Attack Support"] == 1176851).all()


def test_model_error_comparison_csv(paths):
    """Verify model_error_comparison.csv contains valid totals and metrics."""
    csv_path = paths["metrics_dir"] / "model_error_comparison.csv"
    assert csv_path.exists(), "model_error_comparison.csv missing"
    df = pd.read_csv(csv_path)

    assert len(df) == 5, f"Expected 5 model rows, got {len(df)}"
    assert (df["Total Test Samples"] == 1176851).all()
    assert (df["Correct Predictions"] + df["Incorrect Predictions"] == 1176851).all()
    assert (df["Error Rate (%)"] >= 0.0).all() and (df["Error Rate (%)"] <= 100.0).all()


def test_research_error_analysis_csv(paths):
    """Verify research_error_analysis.csv format and values."""
    csv_path = paths["metrics_dir"] / "research_error_analysis.csv"
    assert csv_path.exists(), "research_error_analysis.csv missing"
    df = pd.read_csv(csv_path)
    assert len(df) == 5
    assert "Most Confused Class Pair" in df.columns
    assert "Confusion Count" in df.columns


def test_error_analysis_graphs_exist(paths):
    """Verify generated error diagnostic visualization charts exist."""
    required_graphs = [
        "error_count_by_model.png",
        "error_rate_by_model.png",
        "benign_vs_attack_confusion.png",
        "classwise_f1_comparison.png",
    ]
    for g in required_graphs:
        graph_path = paths["graphs_dir"] / g
        assert graph_path.exists(), f"Error diagnostic graph missing: {graph_path}"
        assert graph_path.stat().st_size > 1000, f"Graph image file empty: {graph_path}"


def test_phase12_documentation_artifacts_exist(paths):
    """Verify all Phase 12 documentation reports exist."""
    required_docs = [
        "confusion_matrix_consistency_check.md",
        "deep_learning_training_curve_analysis.md",
        "error_analysis_random_forest.md",
        "error_analysis_xgboost.md",
        "error_analysis_cnn_1d.md",
        "error_analysis_bilstm.md",
        "error_analysis_cnn_bilstm.md",
        "confusion_matrix_analysis.md",
    ]
    for d in required_docs:
        doc_path = paths["docs_dir"] / d
        assert doc_path.exists(), f"Documentation markdown missing: {doc_path}"
        assert doc_path.stat().st_size > 500, f"Documentation file too short: {doc_path}"
