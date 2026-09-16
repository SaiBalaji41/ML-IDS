"""
Random Forest Baseline Classifier Module for ML-IDS.
Encapsulates Scikit-Learn's RandomForestClassifier with robust training,
evaluation, persistence, and feature importance extraction capabilities.
"""

from pathlib import Path
import pickle
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from sklearn.ensemble import RandomForestClassifier


class RandomForestBaselineModel:
    """
    Random Forest baseline classifier for network intrusion detection.
    
    Parameters
    ----------
    n_estimators : int, default=200
        Number of decision trees in the ensemble.
    max_depth : int or None, default=25
        Maximum depth of each tree. Constrained to balance expressiveness and memory.
    min_samples_split : int, default=5
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=2
        Minimum number of samples required to be at a leaf node.
    max_features : str or int, default="sqrt"
        Number of features to consider when looking for the best split.
    class_weight : str or None, default="balanced"
        Weights associated with classes to handle severe IoT attack class imbalance.
    max_samples : float or int or None, default=None
        Fraction or number of samples to draw from X to train each base estimator.
    random_state : int, default=42
        Fixed random seed for deterministic reproducibility.
    n_jobs : int, default=-1
        Number of parallel CPU workers.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 25,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: Union[str, int, float] = "sqrt",
        class_weight: Optional[str] = "balanced",
        max_samples: Optional[Union[int, float]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model: Optional[RandomForestClassifier] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.training_time_sec: float = 0.0

    def build_model(self) -> RandomForestClassifier:
        """Instantiate the underlying scikit-learn RandomForestClassifier."""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        return self.model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaselineModel":
        """
        Fit the Random Forest classifier on training data.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Standardized feature matrix.
        y : np.ndarray of shape (n_samples,)
            Integer-encoded ground-truth target vector.
        """
        if self.model is None:
            self.build_model()
            
        t0 = time.time()
        self.model.fit(X, y)
        self.training_time_sec = round(time.time() - t0, 2)
        
        self.feature_importances_ = self.model.feature_importances_
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X."""
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        return self.model.predict_proba(X)

    def get_feature_importances(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Extract Gini-based feature importances mapped to feature names.
        
        Parameters
        ----------
        feature_names : list of str, optional
            List of column names matching the feature dimension.
            
        Returns
        -------
        dict : Mapping from feature name to float importance value.
        """
        if self.feature_importances_ is None:
            raise RuntimeError("Model must be fitted before retrieving feature importances.")
            
        importances = self.feature_importances_
        if feature_names is not None and len(feature_names) == len(importances):
            return {name: float(imp) for name, imp in zip(feature_names, importances)}
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}

    def save(self, filepath: Union[str, Path] = "models/random_forest.pkl"):
        """Serialize the fitted model and attributes to a pickle file."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_importances_": self.feature_importances_,
                    "classes_": self.classes_,
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "class_weight": self.class_weight,
                    "random_state": self.random_state,
                    "training_time_sec": self.training_time_sec,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(self, filepath: Union[str, Path] = "models/random_forest.pkl") -> "RandomForestBaselineModel":
        """Load a serialized model from disk."""
        with open(filepath, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.feature_importances_ = payload["feature_importances_"]
        self.classes_ = payload["classes_"]
        self.n_estimators = payload["n_estimators"]
        self.max_depth = payload["max_depth"]
        self.class_weight = payload.get("class_weight", None)
        self.random_state = payload["random_state"]
        self.training_time_sec = payload.get("training_time_sec", 0.0)
        return self
