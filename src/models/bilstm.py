"""
Bidirectional LSTM (BiLSTM) Deep Learning Model Architecture for ML-IDS (Phase 9).

Implements a 2-layer Bidirectional LSTM network operating on ordered tabular feature
vectors (reshaped to [batch_size, 46, 1]) for sequential pattern extraction and classification.

NOTE: The BiLSTM treats the ordered feature vector as a sequence. The feature order
does not necessarily represent a real physical or temporal sequence, but allows the model
to capture bidirectional inter-feature contextual dependencies across the feature space.
"""

import os
import sys
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


class BiLSTMModel:
    """
    BiLSTM architecture for network traffic feature vector classification.
    Input Tensor Shape: (batch_size, num_features, 1)
    """

    def __init__(
        self,
        num_features: int = 46,
        num_classes: int = 34,
        lstm_units: Tuple[int, int] = (64, 32),
        dense_units: int = 64,
        dropout_rate: float = 0.30,
        learning_rate: float = 0.001,
        loss: Optional[str] = None,
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.loss = loss
        self.model: Optional[tf.keras.Model] = None

    def build_model(self) -> tf.keras.Model:
        """
        Construct and compile the BiLSTM architecture according to Phase 9 specifications.
        """
        tf.random.set_seed(42)
        np.random.seed(42)

        input_layer = layers.Input(shape=(self.num_features, 1), name="input_features")

        # Layer 1: BiLSTM (return_sequences=True) + Dropout
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units[0], return_sequences=True),
            name="bilstm_1",
        )(input_layer)
        x = layers.Dropout(self.dropout_rate, name="dropout_1")(x)

        # Layer 2: BiLSTM (return_sequences=False) + Dropout
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units[1], return_sequences=False),
            name="bilstm_2",
        )(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_2")(x)

        # Dense layer + Dropout
        x = layers.Dense(self.dense_units, activation="relu", name="dense_64")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_dense")(x)

        # Output Layer
        if self.num_classes > 2:
            output_layer = layers.Dense(self.num_classes, activation="softmax", name="output_softmax")(x)
            loss_fn = self.loss or "sparse_categorical_crossentropy"
            metrics = ["accuracy"]
        else:
            output_layer = layers.Dense(1, activation="sigmoid", name="output_sigmoid")(x)
            loss_fn = self.loss or "binary_crossentropy"
            metrics = ["accuracy"]

        self.model = models.Model(inputs=input_layer, outputs=output_layer, name="ML_IDS_BiLSTM")
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        return self.model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 25,
        batch_size: int = 512,
        callbacks_list: Optional[list] = None,
        class_weight: Optional[Dict[int, float]] = None,
    ):
        """Train the BiLSTM on training data with validation monitoring."""
        if self.model is None:
            self.build_model()

        if X_train.ndim == 2:
            X_train = np.expand_dims(X_train, axis=-1)
        if X_val.ndim == 2:
            X_val = np.expand_dims(X_val, axis=-1)

        return self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            class_weight=class_weight,
            verbose=1,
        )

    def predict(self, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        """Generate class index predictions."""
        if self.model is None:
            raise RuntimeError("Model has not been initialized.")
        if X.ndim == 2:
            X = np.expand_dims(X, axis=-1)
        probs = self.model.predict(X, batch_size=batch_size, verbose=0)
        if self.num_classes > 2:
            return np.argmax(probs, axis=1)
        return (probs > 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        """Generate class probability predictions."""
        if self.model is None:
            raise RuntimeError("Model has not been initialized.")
        if X.ndim == 2:
            X = np.expand_dims(X, axis=-1)
        return self.model.predict(X, batch_size=batch_size, verbose=0)

    def save(self, filepath: Union[str, Path] = "models/bilstm/best_model.keras"):
        """Save Keras model checkpoint to disk."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(out_path)

    def load(self, filepath: Union[str, Path] = "models/bilstm/best_model.keras") -> "BiLSTMModel":
        """Load saved Keras model checkpoint from disk."""
        self.model = tf.keras.models.load_model(filepath)
        return self


# Helper function
def build_bilstm_model(
    input_shape: Tuple[int, int] = (46, 1),
    num_classes: int = 34,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Helper function to directly build and return a compiled BiLSTM tf.keras.Model."""
    model_wrapper = BiLSTMModel(
        num_features=input_shape[0],
        num_classes=num_classes,
        learning_rate=learning_rate,
    )
    return model_wrapper.build_model()
