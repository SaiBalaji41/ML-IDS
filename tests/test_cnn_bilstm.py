"""
Unit tests for Proposed Hybrid CNN + BiLSTM deep learning model architecture and interface.
"""

from pathlib import Path
import pytest

try:
    import numpy as np
    _ = np.random.randn(2, 2)
    from src.models.cnn_bilstm import CNNBiLSTMModel, TF_AVAILABLE
    DL_AVAILABLE = TF_AVAILABLE
except (ImportError, Exception):
    DL_AVAILABLE = False
    np = None


def test_cnn_bilstm_initialization():
    """Test hybrid model hyperparameter initialization."""
    if not DL_AVAILABLE:
        pytest.skip("TensorFlow / NumPy C-extensions not accessible in current environment")

    model_wrapper = CNNBiLSTMModel(
        num_features=46,
        num_classes=8,
        conv_filters=64,
        kernel_size=3,
        lstm_units=64,
        dense_units=64,
        dropout_rate=0.3,
        learning_rate=0.001,
    )
    assert model_wrapper.num_features == 46
    assert model_wrapper.num_classes == 8
    assert model_wrapper.conv_filters == 64
    assert model_wrapper.kernel_size == 3
    assert model_wrapper.lstm_units == 64
    assert model_wrapper.dense_units == 64
    assert model_wrapper.dropout_rate == 0.3
    assert model_wrapper.learning_rate == 0.001
    assert model_wrapper.model is None


def test_cnn_bilstm_architecture_build():
    """Test hybrid model construction, tensor dimensions, and parameter counts."""
    if not DL_AVAILABLE:
        pytest.skip("TensorFlow / NumPy C-extensions not accessible in current environment")

    model_wrapper = CNNBiLSTMModel(num_features=46, num_classes=8)
    keras_model = model_wrapper.build_model()

    assert keras_model is not None
    assert keras_model.input_shape == (None, 46, 1)
    assert keras_model.output_shape == (None, 8)
    assert keras_model.count_params() > 0


def test_cnn_bilstm_fit_predict(tmp_path):
    """Test mock training, inference, and serialization."""
    if not DL_AVAILABLE:
        pytest.skip("TensorFlow / NumPy C-extensions not accessible in current environment")

    np.random.seed(42)
    X_train = np.random.randn(32, 46).astype(np.float32)
    y_train = np.random.randint(0, 4, size=32)
    X_val = np.random.randn(16, 46).astype(np.float32)
    y_val = np.random.randint(0, 4, size=16)

    model_wrapper = CNNBiLSTMModel(num_features=46, num_classes=4)
    model_wrapper.build_model()

    # Train for 1 epoch on mock data
    history = model_wrapper.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=1,
        batch_size=16,
    )
    assert "loss" in history.history
    assert "val_loss" in history.history

    # Predictions
    preds = model_wrapper.predict(X_val)
    probs = model_wrapper.predict_proba(X_val)

    assert preds.shape == (16,)
    assert probs.shape == (16, 4)

    # Save and Load
    save_file = tmp_path / "best_model.keras"
    model_wrapper.save(save_file)
    assert save_file.exists()

    loaded_wrapper = CNNBiLSTMModel(num_features=46, num_classes=4)
    loaded_wrapper.load(save_file)
    loaded_preds = loaded_wrapper.predict(X_val)
    assert np.array_equal(preds, loaded_preds)
