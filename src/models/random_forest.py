"""
Random Forest Baseline Classifier Module for ML-IDS.
Wraps Scikit-Learn's RandomForestClassifier with reproducibility and logging support.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestBaselineModel:
    """
    Random Forest baseline classifier for network intrusion detection.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        class_weight: Optional[str] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model: Optional[RandomForestClassifier] = None

    def build_model(self) -> RandomForestClassifier:
        """Instantiate the scikit-learn RandomForestClassifier."""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        return self.model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaselineModel":
        """Fit the model on training data."""
        if self.model is None:
            self.build_model()
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate class predictions."""
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate class probability estimates."""
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        return self.model.predict_proba(X)

    def get_feature_importances(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Extract Gini-based feature importances."""
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            raise RuntimeError("Model must be trained before extracting feature importances.")
        importances = self.model.feature_importances_
        if feature_names is not None and len(feature_names) == len(importances):
            return dict(zip(feature_names, importances))
        return {f"feature_{i}": float(val) for i, val in enumerate(importances)}

    def save(self, filepath: Union[str, Path] = "models/random_forest.pkl"):
        """Serialize fitted model to disk."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out_path)

    def load(self, filepath: Union[str, Path] = "models/random_forest.pkl"):
        """Load serialized model from disk."""
        self.model = joblib.load(filepath)
        return self
