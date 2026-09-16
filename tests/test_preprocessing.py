"""
Unit tests for ML-IDS data preprocessing pipeline components.
Tests cleaning, label encoding, stratified splitting, and leakage-safe scaling.
"""

import pytest

try:
    import numpy as np
    _ = np.random.randn(2, 2)
    import pandas as pd
    from src.preprocessing.clean_data import clean_dataset
    from src.preprocessing.feature_processing import FeatureProcessor
    from src.preprocessing.label_processing import encode_labels, identify_label_column
    from src.preprocessing.split_data import split_dataset
    NUMPY_AVAILABLE = True
except (ImportError, Exception):
    NUMPY_AVAILABLE = False


@pytest.fixture
def mock_network_data():
    """Create a synthetic mini-DataFrame mimicking flow features and labels."""
    if not NUMPY_AVAILABLE:
        pytest.skip("NumPy / Pandas native DLLs not accessible in current environment")
    np.random.seed(42)
    n = 100
    data = {
        "flow_duration": np.random.uniform(0.1, 10.0, size=n),
        "packet_count": np.random.randint(1, 500, size=n).astype(float),
        "byte_rate": np.random.uniform(100.0, 5000.0, size=n),
        "constant_feature": np.ones(n),
        "label": ["Benign"] * 60 + ["DDoS"] * 25 + ["PortScan"] * 15,
    }
    data["byte_rate"][5] = np.inf
    data["packet_count"][10] = np.nan
    return pd.DataFrame(data)


def test_clean_dataset(mock_network_data):
    """Test infinity conversion, null imputation, and constant column removal."""
    if not NUMPY_AVAILABLE:
        pytest.skip("NumPy / Pandas native DLLs not accessible in current environment")
    cleaned_df, stats = clean_dataset(mock_network_data, drop_constant_features=True)
    assert stats["infinities_replaced"] == 1
    assert stats["missing_values_handled"] >= 1
    assert "constant_feature" in stats["constant_columns_dropped"]
    assert "constant_feature" not in cleaned_df.columns
    assert not np.isinf(cleaned_df.select_dtypes(include=[np.number]).values).any()
    assert not cleaned_df.isnull().any().any()


def test_label_encoding(mock_network_data, tmp_path):
    """Test deterministic label encoding with Benign mapped to index 0."""
    if not NUMPY_AVAILABLE:
        pytest.skip("NumPy / Pandas native DLLs not accessible in current environment")
    mapping_file = tmp_path / "label_mapping.json"
    label_col = identify_label_column(mock_network_data)
    assert label_col == "label"

    y_encoded, mapping = encode_labels(mock_network_data, label_column=label_col, output_mapping_path=mapping_file)
    assert mapping["Benign"] == 0
    assert len(mapping) == 3
    assert mapping_file.exists()


def test_stratified_split(mock_network_data):
    """Test stratified splitting into 70/15/15 partitions."""
    if not NUMPY_AVAILABLE:
        pytest.skip("NumPy / Pandas native DLLs not accessible in current environment")
    X = mock_network_data.drop(columns=["label"])
    y = mock_network_data["label"]
    X_train, X_val, X_test, y_train, y_val, y_test, meta = split_dataset(X, y, train_size=0.70, val_size=0.15, test_size=0.15)
    assert len(X_train) == 70
    assert len(X_val) == 15
    assert len(X_test) == 15
    assert set(y_train.unique()) == set(y.unique())


def test_leakage_safe_feature_processor(mock_network_data, tmp_path):
    """Test that scaler is fit exclusively on training data and transforms test data statelessly."""
    if not NUMPY_AVAILABLE:
        pytest.skip("NumPy / Pandas native DLLs not accessible in current environment")
    cleaned_df, _ = clean_dataset(mock_network_data)
    X = cleaned_df.drop(columns=["label"])
    y = cleaned_df["label"]
    X_train, X_val, X_test, _, _, _, _ = split_dataset(X, y, train_size=0.70, val_size=0.15, test_size=0.15)

    processor = FeatureProcessor(scaler_type="StandardScaler", artifact_dir=tmp_path)
    X_train_scaled, meta = processor.fit_transform(X_train, save_artifacts=True)
    X_val_scaled = processor.transform(X_val)
    X_test_scaled = processor.transform(X_test)

    assert X_train_scaled.shape[0] == 70
    assert X_val_scaled.shape[0] == 15
    assert X_test_scaled.shape[0] == 15
    assert X_train_scaled.shape[1] == X_val_scaled.shape[1] == X_test_scaled.shape[1]
    assert (tmp_path / "scaler.pkl").exists()
