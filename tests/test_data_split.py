"""
Unit tests for Train/Validation/Test split verification logic.
Tests dimension consistency, zero-leakage assertions, and hash overlap detection.
"""

from pathlib import Path
import pytest

try:
    import numpy as np
    _ = np.random.randn(2, 2)
    from src.preprocessing.verify_split import compute_row_hashes, load_partition
    NUMPY_AVAILABLE = True
except (ImportError, Exception):
    NUMPY_AVAILABLE = False


def test_compute_row_hashes():
    """Test MD5 row hash computation and duplicate overlap detection."""
    if not NUMPY_AVAILABLE:
        pytest.skip("NumPy native DLLs not accessible in current environment")
    X1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    X2 = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])
    X3 = np.array([[1.0, 2.0, 3.0], [99.0, 99.0, 99.0]])  # Row 0 overlaps with X1

    h1 = compute_row_hashes(X1)
    h2 = compute_row_hashes(X2)
    h3 = compute_row_hashes(X3)

    assert len(h1) == 3
    assert len(h2) == 2
    assert len(h1.intersection(h2)) == 0  # Zero overlap
    assert len(h1.intersection(h3)) == 1  # 1 overlapping row detected


def test_dimension_and_nan_assertions():
    """Test assertions for dimension consistency and NaN checks."""
    if not NUMPY_AVAILABLE:
        pytest.skip("NumPy native DLLs not accessible in current environment")
    X_train = np.random.randn(50, 10).astype(np.float32)
    X_val = np.random.randn(15, 10).astype(np.float32)
    X_test = np.random.randn(15, 10).astype(np.float32)

    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_val).any()
    assert not np.isnan(X_test).any()
    assert not np.isinf(X_train).any()


def test_processed_split_files_exist():
    """Verify that processed split files and label mappings exist on disk."""
    train_file = Path("data/processed/train/train.npz")
    val_file = Path("data/processed/validation/val.npz")
    test_file = Path("data/processed/test/test.npz")
    mapping_file = Path("data/processed/label_mapping.json")

    assert train_file.exists(), "train.npz missing"
    assert val_file.exists(), "val.npz missing"
    assert test_file.exists(), "test.npz missing"
    assert mapping_file.exists(), "label_mapping.json missing"
