"""
Feature Processing & Scaling Module for CICIoT2023 Dataset.
Enforces strict data leakage prevention: Scalers/Encoders are fit ONLY on the training split.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


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
            return StandardScaler()
        elif self.scaler_type == "MinMaxScaler":
            return MinMaxScaler()
        elif self.scaler_type == "RobustScaler":
            return RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler_type: {self.scaler_type}")

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
            X_train_processed[self.num_cols] = self.scaler.fit_transform(X_train[self.num_cols])

        # Handle any residual categorical columns with dummy encoding if needed
        if self.cat_cols:
            X_train_processed = pd.get_dummies(X_train_processed, columns=self.cat_cols, drop_first=True)
            self.feature_names = list(X_train_processed.columns)

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
            X_processed[self.num_cols] = self.scaler.transform(X[self.num_cols])

        if self.cat_cols:
            X_processed = pd.get_dummies(X_processed, columns=self.cat_cols, drop_first=True)
            # Align columns with training feature names
            X_processed = X_processed.reindex(columns=self.feature_names, fill_value=0)

        return X_processed.values.astype(np.float32)

    def save_artifacts(self, filename: str = "scaler.pkl"):
        """Save fitted scaler to artifact directory."""
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self.artifact_dir / filename
        joblib.dump(
            {
                "scaler": self.scaler,
                "scaler_type": self.scaler_type,
                "feature_names": self.feature_names,
                "num_cols": self.num_cols,
                "cat_cols": self.cat_cols,
            },
            artifact_path,
        )

    def load_artifacts(self, filepath: Union[str, Path] = "models/preprocessing/scaler.pkl"):
        """Load fitted scaler from artifact directory."""
        data = joblib.load(filepath)
        self.scaler = data["scaler"]
        self.scaler_type = data["scaler_type"]
        self.feature_names = data["feature_names"]
        self.num_cols = data["num_cols"]
        self.cat_cols = data["cat_cols"]
