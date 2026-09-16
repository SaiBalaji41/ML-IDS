"""
1D-CNN Deep Learning Model Architecture for ML-IDS.
Implements a 1D Convolutional Neural Network for local pattern extraction across ordered flow features.
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


class Conv1DModel:
    """
    1D-CNN architecture for network traffic feature sequence classification.
    Input Tensor Shape: (batch_size, num_features, 1)
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        filters: Tuple[int, int] = (64, 128),
        kernel_size: int = 3,
        dense_units: int = 128,
        dropout_rate: float = 0.3,
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
        self.model = None

    def build_model(self):
        """Construct and compile the 1D-CNN architecture."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required to build the 1D-CNN model.")

        input_layer = layers.Input(shape=(self.num_features, 1), name="flow_feature_input")

        # Conv Block 1
        x = layers.Conv1D(
            filters=self.filters[0],
            kernel_size=self.kernel_size,
            padding="same",
            activation="relu",
            name="conv1d_block1",
        )(input_layer)
        x = layers.BatchNormalization(name="bn_block1")(x)
        x = layers.MaxPooling1D(pool_size=2, name="maxpool_block1")(x)

        # Conv Block 2
        x = layers.Conv1D(
            filters=self.filters[1],
            kernel_size=self.kernel_size,
            padding="same",
            activation="relu",
            name="conv1d_block2",
        )(x)
        x = layers.BatchNormalization(name="bn_block2")(x)
        x = layers.MaxPooling1D(pool_size=2, name="maxpool_block2")(x)
        x = layers.Dropout(self.dropout_rate, name="spatial_dropout")(x)

        # Aggregation & Classification Head
        x = layers.GlobalAveragePooling1D(name="global_avg_pooling")(x)
        x = layers.Dense(self.dense_units, activation="relu", name="dense_features")(x)
        x = layers.Dropout(self.dropout_rate, name="dense_dropout")(x)

        # Output Layer
        if self.num_classes > 2:
            output_layer = layers.Dense(self.num_classes, activation="softmax", name="output_multiclass")(x)
            loss_fn = self.loss or "sparse_categorical_crossentropy"
            metrics = ["accuracy"]
        else:
            output_layer = layers.Dense(1, activation="sigmoid", name="output_binary")(x)
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
        batch_size: int = 128,
        callbacks_list: Optional[list] = None,
        class_weight: Optional[Dict[int, float]] = None,
    ):
        """Train the 1D-CNN on training data and evaluate on validation data."""
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

    def save(self, filepath: Union[str, Path] = "models/cnn_1d/best_model.keras"):
        """Save model checkpoint to disk."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(out_path)

    def load(self, filepath: Union[str, Path] = "models/cnn_1d/best_model.keras"):
        """Load saved model checkpoint from disk."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required to load model.")
        self.model = tf.keras.models.load_model(filepath)
        return self
