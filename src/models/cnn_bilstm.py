"""
Proposed Hybrid CNN + BiLSTM Deep Learning Model Architecture for ML-IDS (Phase 10).

Combines 1D Convolutional Neural Network (for local spatial pattern extraction over ordered features)
with a Bidirectional LSTM (for capturing bidirectional contextual dependencies across extracted representations).

NOTE: The input consists of 46 standardized tabular network flow features reshaped into [batch_size, 46, 1].
The sequence dimension corresponds to the ordered feature vector and NOT necessarily a real temporal packet sequence.
"""

import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


class CNNBiLSTMModel:
    """
    Proposed Hybrid CNN + BiLSTM architecture for network intrusion detection.
    Input Tensor Shape: (batch_size, num_features, 1)
    """

    def __init__(
        self,
        num_features: int = 46,
        num_classes: int = 34,
        conv_filters: int = 64,
        kernel_size: int = 3,
        pool_size: int = 2,
        lstm_units: int = 64,
        dense_units: int = 64,
        dropout_rate: float = 0.30,
        learning_rate: float = 0.001,
        loss: Optional[str] = None,
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.loss = loss
        self.model: Optional[tf.keras.Model] = None

    def build_model(self) -> tf.keras.Model:
        """
        Construct and compile the hybrid CNN-BiLSTM architecture according to Phase 10 specifications.
        """
        tf.random.set_seed(42)
        np.random.seed(42)

        input_layer = layers.Input(shape=(self.num_features, 1), name="input_features")

        # 1. CNN Feature Extractor Block
        x = layers.Conv1D(
            filters=self.conv_filters,
            kernel_size=self.kernel_size,
            padding="same",
            name="conv1d_local_patterns",
        )(input_layer)
        x = layers.BatchNormalization(name="bn_1")(x)
        x = layers.ReLU(name="relu_1")(x)
        x = layers.MaxPooling1D(pool_size=self.pool_size, name="maxpool_1")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_cnn")(x)

        # 2. BiLSTM Contextual Sequence Processing Block
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=False),
            name="bilstm_context",
        )(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_bilstm")(x)

        # 3. Dense Classification Head
        x = layers.Dense(self.dense_units, activation="relu", name="dense_64")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_dense")(x)

        # 4. Output Layer
        if self.num_classes > 2:
            output_layer = layers.Dense(self.num_classes, activation="softmax", name="output_softmax")(x)
            loss_fn = self.loss or "sparse_categorical_crossentropy"
            metrics = ["accuracy"]
        else:
            output_layer = layers.Dense(1, activation="sigmoid", name="output_sigmoid")(x)
            loss_fn = self.loss or "binary_crossentropy"
            metrics = ["accuracy"]

        self.model = models.Model(inputs=input_layer, outputs=output_layer, name="ML_IDS_Hybrid_CNN_BiLSTM")
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
        """Train the hybrid CNN-BiLSTM on training data with validation monitoring."""
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

    def save(self, filepath: Union[str, Path] = "models/cnn_bilstm/best_model.keras"):
        """Save Keras model checkpoint to disk."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(out_path)

    def load(self, filepath: Union[str, Path] = "models/cnn_bilstm/best_model.keras") -> "CNNBiLSTMModel":
        """Load saved Keras model checkpoint from disk."""
        self.model = tf.keras.models.load_model(filepath)
        return self


# Aliases and helper functions
HybridCNNBiLSTMModel = CNNBiLSTMModel


def build_cnn_bilstm_model(
    input_shape: Tuple[int, int] = (46, 1),
    num_classes: int = 34,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Helper function to directly build and return a compiled hybrid CNN-BiLSTM tf.keras.Model."""
    model_wrapper = CNNBiLSTMModel(
        num_features=input_shape[0],
        num_classes=num_classes,
        learning_rate=learning_rate,
    )
    return model_wrapper.build_model()

