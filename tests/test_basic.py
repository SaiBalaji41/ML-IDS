"""
Basic unit tests for ML-IDS repository structure and import verification.
"""

from pathlib import Path


def test_directory_structure():
    """Verify that all essential directories exist in the project."""
    root = Path(__file__).parent.parent
    expected_dirs = [
        "data/raw",
        "data/processed",
        "data/sample",
        "notebooks",
        "src/preprocessing",
        "src/models",
        "src/evaluation",
        "src/explainability",
        "src/realtime",
        "models",
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
        "configs/default_config.yaml",
    ]
    for file_path in expected_files:
        full_path = root / file_path
        assert full_path.exists(), f"File missing: {file_path}"
        assert full_path.is_file(), f"Expected file: {file_path}"


def test_package_imports():
    """Verify that the src packages can be imported cleanly."""
    import src
    import src.preprocessing
    import src.models
    import src.evaluation
    import src.explainability
    import src.realtime

    assert src.__doc__ is not None
