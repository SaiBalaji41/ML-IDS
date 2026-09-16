"""
Basic unit tests for ML-IDS repository structure and package configuration.
"""

from pathlib import Path


def test_directory_structure():
    """Verify that all essential directories exist in the project."""
    root = Path(__file__).parent.parent
    expected_dirs = [
        "data/raw",
        "data/processed",
        "data/processed/train",
        "data/processed/validation",
        "data/processed/test",
        "data/sample",
        "notebooks",
        "src/preprocessing",
        "src/models",
        "src/evaluation",
        "src/explainability",
        "src/realtime",
        "models",
        "models/preprocessing",
        "results/metrics",
        "results/graphs/dataset_analysis",
        "results/confusion_matrices",
        "results/shap",
        "dashboard",
        "tests",
        "docs",
        "configs",
        "scripts",
    ]
    for dir_path in expected_dirs:
        full_path = root / dir_path
        assert full_path.exists(), f"Directory missing: {dir_path}"
        assert full_path.is_dir(), f"Expected directory but found file: {dir_path}"


def test_essential_files_exist():
    """Verify that core configuration and documentation files exist."""
    root = Path(__file__).parent.parent
    expected_files = [
        "requirements.txt",
        "README.md",
        ".gitignore",
        ".env.example",
        "docs/PRD.md",
        "docs/preprocessing_decisions.md",
        "docs/preprocessing_report.md",
        "configs/default_config.yaml",
        "src/preprocessing/load_data.py",
        "src/preprocessing/clean_data.py",
        "src/preprocessing/feature_processing.py",
        "src/preprocessing/label_processing.py",
        "src/preprocessing/split_data.py",
        "src/preprocessing/preprocess_pipeline.py",
        "notebooks/02_data_preprocessing.ipynb",
    ]
    for file_path in expected_files:
        full_path = root / file_path
        assert full_path.exists(), f"File missing: {file_path}"
        assert full_path.is_file(), f"Expected file: {file_path}"


def test_package_structure():
    """Verify that src package modules exist with expected file structure."""
    root = Path(__file__).parent.parent / "src"
    assert (root / "__init__.py").exists()
    assert (root / "preprocessing" / "__init__.py").exists()
    assert (root / "models" / "__init__.py").exists()
    assert (root / "evaluation" / "__init__.py").exists()
    assert (root / "explainability" / "__init__.py").exists()
    assert (root / "realtime" / "__init__.py").exists()
