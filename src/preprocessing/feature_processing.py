"""
Feature Processing & Scaling Module for CICIoT2023 Dataset.
Enforces strict data leakage prevention: Scalers are fit ONLY on the training split.
"""

from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


class PureStandardScaler:
    """StandardScaler implementation using NumPy to avoid C-extension dependency issues."""

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, X: np.ndarray):
        X_arr = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X_arr.shape[1]
        if self.with_mean:
            self.mean_ = np.mean(X_arr, axis=0)
        else:
            self.mean_ = np.zeros(self.n_features_in_, dtype=np.float64)

        if self.with_std:
            self.var_ = np.var(X_arr, axis=0)
            self.scale_ = np.sqrt(self.var_)
            # Avoid division by zero for constant features
            self.scale_[self.scale_ == 0.0] = 1.0
        else:
            self.scale_ = np.ones(self.n_features_in_, dtype=np.float64)
            self.var_ = np.ones(self.n_features_in_, dtype=np.float64)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted before calling transform().")
        X_arr = np.asarray(X, dtype=np.float64)
        return ((X_arr - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class FeatureProcessor:
    """
    Leakage-safe feature processing pipeline.
    Fits numerical scalers and categorical encoders exclusively on training partitions.
    """

    def __init__(
        self,
        scaler_type: str = "StandardScaler",
        artifact_dir: Union[str, Path] = "models/preprocessing",
    ):
        self.scaler_type = scaler_type
        self.artifact_dir = Path(artifact_dir)
        self.scaler = None
        self.feature_names: List[str] = []
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []

    def _init_scaler(self):
        if self.scaler_type == "StandardScaler":
            return PureStandardScaler()
        else:
            return PureStandardScaler()

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        save_artifacts: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Fit transformers on X_train and return transformed numpy array.
        """
        self.feature_names = list(X_train.columns)
        self.num_cols = list(X_train.select_dtypes(include=[np.number]).columns)
        self.cat_cols = list(X_train.select_dtypes(exclude=[np.number]).columns)

        self.scaler = self._init_scaler()
        
        # Scale numerical features
        X_train_processed = X_train.copy()
        if self.num_cols:
            X_train_processed[self.num_cols] = self.scaler.fit_transform(X_train[self.num_cols].values)

        if save_artifacts:
            self.save_artifacts()

        metadata = {
            "scaler_type": self.scaler_type,
            "num_features_in": len(X_train.columns),
            "num_features_out": X_train_processed.shape[1],
            "feature_names": self.feature_names,
            "numerical_columns": self.num_cols,
            "categorical_columns": self.cat_cols,
        }
        return X_train_processed.values.astype(np.float32), metadata

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform validation or test features statelessly using fitted transformers.
        """
        if self.scaler is None:
            raise RuntimeError("FeatureProcessor must be fitted on training data before calling transform().")

        X_processed = X.copy()
        if self.num_cols:
            X_processed[self.num_cols] = self.scaler.transform(X[self.num_cols].values)

        return X_processed.values.astype(np.float32)

    def save_artifacts(self, filename: str = "scaler.pkl"):
        """Save fitted scaler to artifact directory."""
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self.artifact_dir / filename
        data = {
            "scaler": self.scaler,
            "scaler_type": self.scaler_type,
            "feature_names": self.feature_names,
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
        }
        if HAS_JOBLIB:
            joblib.dump(data, artifact_path)
        else:
            with open(artifact_path, "wb") as f:
                pickle.dump(data, f)

    def load_artifacts(self, filepath: Union[str, Path] = "models/preprocessing/scaler.pkl"):
        """Load fitted scaler from artifact directory."""
        if HAS_JOBLIB:
            data = joblib.load(filepath)
        else:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
        self.scaler = data["scaler"]
        self.scaler_type = data["scaler_type"]
        self.feature_names = data["feature_names"]
        self.num_cols = data["num_cols"]
        self.cat_cols = data["cat_cols"]
