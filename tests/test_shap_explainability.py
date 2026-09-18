"""
Unit & Regression Test Suite for Phase 13: SHAP Explainability & XAI.
Validates existence, dimension integrity, feature alignment, value bounds,
and artifact completeness across all explainable AI components.
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
        "shap_dir": PROJECT_ROOT / "results" / "shap",
        "docs_dir": PROJECT_ROOT / "docs",
        "meta_json": PROJECT_ROOT / "data" / "processed" / "preprocessing_metadata.json",
        "label_mapping": PROJECT_ROOT / "data" / "processed" / "label_mapping.json",
    }


@pytest.fixture(scope="module")
def expected_features(paths):
    with open(paths["meta_json"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta["feature_names"]


def test_shap_directories_exist(paths):
    """Verify that all required Phase 13 SHAP output directories exist."""
    required_dirs = [
        "random_forest",
        "xgboost",
        "cnn_1d",
        "bilstm",
        "cnn_bilstm",
        "class_specific",
        "misclassification_examples",
    ]
    for d in required_dirs:
        dir_path = paths["shap_dir"] / d
        assert dir_path.exists(), f"SHAP directory missing: {dir_path}"
        assert dir_path.is_dir(), f"Expected directory: {dir_path}"


def test_global_feature_importance_csv(paths, expected_features):
    """Verify global_feature_importance.csv has 230 rows (5 models * 46 features) and no NaNs."""
    csv_p = paths["shap_dir"] / "global_feature_importance.csv"
    assert csv_p.exists(), "global_feature_importance.csv missing"
    df = pd.read_csv(csv_p)

    assert len(df) == 5 * 46, f"Expected 230 rows, got {len(df)}"
    assert list(df.columns) == ["Model", "Feature", "Mean_Absolute_SHAP"]
    assert not df["Mean_Absolute_SHAP"].isna().any(), "NaN values found in SHAP importances"
    assert not np.isinf(df["Mean_Absolute_SHAP"]).any(), "Infinite values found in SHAP importances"
    assert (df["Mean_Absolute_SHAP"] >= 0.0).all(), "Negative SHAP values in absolute importance table"

    # Verify features match official feature list
    for model_name in df["Model"].unique():
        m_feats = df[df["Model"] == model_name]["Feature"].tolist()
        assert set(m_feats) == set(expected_features), f"Feature mismatch for {model_name}"


def test_top_features_csv(paths):
    """Verify top_features.csv has 75 rows (5 models * 15 ranks)."""
    csv_p = paths["shap_dir"] / "top_features.csv"
    assert csv_p.exists(), "top_features.csv missing"
    df = pd.read_csv(csv_p)

    assert len(df) == 5 * 15, f"Expected 75 rows, got {len(df)}"
    assert list(df.columns) == ["Model", "Rank", "Feature", "Mean_Absolute_SHAP"]
    assert (df["Rank"] >= 1).all() and (df["Rank"] <= 15).all()


def test_research_shap_summary_csv(paths):
    """Verify research_shap_summary.csv has 50 rows (5 models * 10 ranks)."""
    csv_p = paths["shap_dir"] / "research_shap_summary.csv"
    assert csv_p.exists(), "research_shap_summary.csv missing"
    df = pd.read_csv(csv_p)

    assert len(df) == 5 * 10, f"Expected 50 rows, got {len(df)}"
    assert "Mean Absolute SHAP" in df.columns
    assert "Rank" in df.columns


def test_shap_experiment_metadata_json(paths):
    """Verify metadata JSON contains complete reproducible configuration."""
    json_p = paths["shap_dir"] / "shap_experiment_metadata.json"
    assert json_p.exists(), "shap_experiment_metadata.json missing"
    with open(json_p, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["random_seed"] == 42
    assert meta["feature_count"] == 46
    assert meta["class_count"] == 34
    assert len(meta["models_explained"]) == 5


def test_shap_visual_artifacts_exist(paths):
    """Verify summary and bar plots exist for all models."""
    models = ["random_forest", "xgboost", "cnn_1d", "bilstm", "cnn_bilstm"]
    for m in models:
        m_dir = paths["shap_dir"] / m
        summary_plot = m_dir / f"{m}_shap_summary.png"
        bar_plot = m_dir / f"{m}_shap_bar.png"
        assert summary_plot.exists(), f"Summary plot missing for {m}: {summary_plot}"
        assert bar_plot.exists(), f"Bar plot missing for {m}: {bar_plot}"
        assert summary_plot.stat().st_size > 1000, f"Plot image file empty: {summary_plot}"


def test_class_specific_and_misclassification_plots(paths):
    """Verify class-specific and misclassification example plots exist."""
    class_dir = paths["shap_dir"] / "class_specific"
    misc_dir = paths["shap_dir"] / "misclassification_examples"

    assert len(list(class_dir.glob("*.png"))) >= 5, "Missing class-specific SHAP plots"
    assert len(list(misc_dir.glob("*.png"))) >= 3, "Missing misclassification SHAP plots"


def test_phase13_documentation_artifacts_exist(paths):
    """Verify required markdown documentation files exist."""
    required_docs = [
        "shap_feature_mapping.md",
        "shap_explainability.md",
        "shap_model_comparison.md",
    ]
    for d in required_docs:
        doc_p = paths["docs_dir"] / d
        assert doc_p.exists(), f"Documentation missing: {doc_p}"
        assert doc_p.stat().st_size > 500, f"Documentation too short: {doc_p}"
