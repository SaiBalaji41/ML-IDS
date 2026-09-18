"""
1D-CNN Deep Learning Model Architecture for ML-IDS (Phase 8).

Implements a 1D Convolutional Neural Network operating on ordered tabular feature
vectors (reshaped to [batch_size, 46, 1]) for local pattern extraction and classification.
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


class Conv1DModel:
    """
    1D-CNN architecture for network traffic feature vector classification.
    Input Tensor Shape: (batch_size, num_features, 1)
    """

    def __init__(
        self,
        num_features: int = 46,
        num_classes: int = 34,
        filters: Tuple[int, int] = (64, 128),
        kernel_size: int = 3,
        dense_units: int = 128,
        dropout_rate: float = 0.30,
        learning_rate: float = 0.001,
        loss: Optional[str] = None,
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        self.filters = filters
        self.kernel_size = kernel_size
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.loss = loss
        self.model: Optional[tf.keras.Model] = None

    def build_model(self) -> tf.keras.Model:
        """
        Construct and compile the 1D-CNN architecture according to Phase 8 specifications.
        """
        tf.random.set_seed(42)
        np.random.seed(42)

        input_layer = layers.Input(shape=(self.num_features, 1), name="input_features")

        # Conv Block 1
        x = layers.Conv1D(
            filters=self.filters[0],
            kernel_size=self.kernel_size,
            padding="same",
            name="conv1d_1",
        )(input_layer)
        x = layers.BatchNormalization(name="bn_1")(x)
        x = layers.ReLU(name="relu_1")(x)
        x = layers.MaxPooling1D(pool_size=2, name="maxpool_1")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_1")(x)

        # Conv Block 2
        x = layers.Conv1D(
            filters=self.filters[1],
            kernel_size=self.kernel_size,
            padding="same",
            name="conv1d_2",
        )(x)
        x = layers.BatchNormalization(name="bn_2")(x)
        x = layers.ReLU(name="relu_2")(x)
        x = layers.MaxPooling1D(pool_size=2, name="maxpool_2")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_2")(x)

        # Aggregation & Classification Head: GlobalAveragePooling1D is chosen over Flatten
        # to drastically reduce parameter count, avoid spatial dimension memorization, and enhance generalization.
        x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
        x = layers.Dense(self.dense_units, activation="relu", name="dense_128")(x)
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

        self.model = models.Model(inputs=input_layer, outputs=output_layer, name="ML_IDS_1D_CNN")
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
        batch_size: int = 512,
        callbacks_list: Optional[list] = None,
        class_weight: Optional[Dict[int, float]] = None,
    ):
        """Train the 1D-CNN on training data with validation monitoring."""
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

    def save(self, filepath: Union[str, Path] = "models/cnn_1d/best_model.keras"):
        """Save Keras model checkpoint to disk."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(out_path)

    def load(self, filepath: Union[str, Path] = "models/cnn_1d/best_model.keras") -> "Conv1DModel":
        """Load saved Keras model checkpoint from disk."""
        self.model = tf.keras.models.load_model(filepath)
        return self


# Aliases and helper functions
CNN1DModel = Conv1DModel


def build_cnn_1d_model(
    input_shape: Tuple[int, int] = (46, 1),
    num_classes: int = 34,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Helper function to directly build and return a compiled 1D-CNN tf.keras.Model."""
    model_wrapper = Conv1DModel(
        num_features=input_shape[0],
        num_classes=num_classes,
        learning_rate=learning_rate,
    )
    return model_wrapper.build_model()

