"""
Proposed Hybrid CNN + BiLSTM Deep Learning Model Architecture for ML-IDS.
Combines 1D Convolutional feature extraction with Bidirectional LSTM sequence modeling.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    TF_AVAILABLE = True
except (ImportError, Exception):
    TF_AVAILABLE = False


class CNNBiLSTMModel:
    """
    Proposed Hybrid 1D-CNN + BiLSTM architecture for network traffic classification:
    1. Conv1D extracts local feature correlations across adjacent feature dimensions.
    2. BatchNormalization & MaxPooling1D stabilize and downsample activations.
    3. Bidirectional LSTM captures long-range bidirectional dependencies across the feature sequence.
    4. Dense layer and Dropout provide regularized classification.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        conv_filters: int = 64,
        kernel_size: int = 3,
        lstm_units: int = 64,
        dense_units: int = 64,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        loss: Optional[str] = None,
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.loss = loss
        self.model = None

    def build_model(self):
        """Construct and compile the hybrid CNN + BiLSTM architecture."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required to build the CNN + BiLSTM model.")

        input_layer = layers.Input(shape=(self.num_features, 1), name="flow_feature_input")

        # 1. Convolutional Local Feature Extraction
        x = layers.Conv1D(
            filters=self.conv_filters,
            kernel_size=self.kernel_size,
            padding="same",
            activation="relu",
            name="conv1d_local_features",
        )(input_layer)
        x = layers.BatchNormalization(name="bn_conv")(x)
        x = layers.MaxPooling1D(pool_size=2, name="maxpool_conv")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_conv")(x)

        # 2. Bidirectional Sequential Dependency Modeling
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=False),
            name="bidirectional_lstm",
        )(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_bilstm")(x)

        # 3. Dense Classification Head
        x = layers.Dense(self.dense_units, activation="relu", name="dense_features")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_dense")(x)

        # 4. Output Layer
        if self.num_classes > 2:
            output_layer = layers.Dense(self.num_classes, activation="softmax", name="output_multiclass")(x)
            loss_fn = self.loss or "sparse_categorical_crossentropy"
            metrics = ["accuracy"]
        else:
            output_layer = layers.Dense(1, activation="sigmoid", name="output_binary")(x)
            loss_fn = self.loss or "binary_crossentropy"
            metrics = ["accuracy"]

        self.model = models.Model(inputs=input_layer, outputs=output_layer, name="ML_IDS_Proposed_CNN_BiLSTM")
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        return self.model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 30,
        batch_size: int = 128,
        callbacks_list: Optional[list] = None,
        class_weight: Optional[Dict[int, float]] = None,
    ):
        """Train the hybrid CNN-BiLSTM on training data and evaluate on validation data."""
        if self.model is None:
            self.build_model()

        # Reshape to (samples, features, 1) if necessary
        if X_train.ndim == 2:
            X_train = np.expand_dims(X_train, axis=-1)
        if X_val.ndim == 2:
            X_val = np.expand_dims(X_val, axis=-1)

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            class_weight=class_weight,
            verbose=1,
        )
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate class predictions."""
        if self.model is None:
            raise RuntimeError("Model has not been built or trained.")
        if X.ndim == 2:
            X = np.expand_dims(X, axis=-1)
        probs = self.model.predict(X, verbose=0)
        if self.num_classes > 2:
            return np.argmax(probs, axis=1)
        return (probs > 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate class prediction probabilities."""
        if self.model is None:
            raise RuntimeError("Model has not been built or trained.")
        if X.ndim == 2:
            X = np.expand_dims(X, axis=-1)
        return self.model.predict(X, verbose=0)

    def save(self, filepath: Union[str, Path] = "models/cnn_bilstm/best_model.keras"):
        """Save model checkpoint to disk."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(out_path)

    def load(self, filepath: Union[str, Path] = "models/cnn_bilstm/best_model.keras"):
        """Load saved model checkpoint from disk."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required to load model.")
        self.model = tf.keras.models.load_model(filepath)
        return self


# Backward-compatible alias
HybridCNNBiLSTMModel = CNNBiLSTMModel
