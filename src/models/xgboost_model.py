"""
XGBoost Baseline Classifier Module for ML-IDS (Phase 7).

Wraps XGBoost's XGBClassifier with support for:
- Multiclass softprob objective with mlogloss evaluation
- Balanced sample weighting on training partitions
- Feature importance extraction via average gain and weight
- Native pickle and booster artifact serialization
"""

from pathlib import Path
import pickle
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import xgboost as xgb


class XGBoostBaselineModel:
    """
    XGBoost gradient boosted decision trees baseline for network intrusion detection.
    
    Parameters
    ----------
    n_estimators : int, default=300
        Number of gradient boosted trees.
    max_depth : int, default=6
        Maximum tree depth for base learners.
    learning_rate : float, default=0.1
        Boosting learning rate (shrinkage).
    subsample : float, default=0.8
        Subsample ratio of the training instances.
    colsample_bytree : float, default=0.8
        Subsample ratio of columns when constructing each tree.
    random_state : int, default=42
        Random number seed.
    n_jobs : int, default=-1
        Number of parallel threads.
    tree_method : str, default="hist"
        Fast histogram-optimized tree construction method.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        objective: Optional[str] = None,
        eval_metric: Optional[str] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        tree_method: str = "hist",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.tree_method = tree_method

        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.training_time_sec: float = 0.0

    def build_model(self, num_classes: int = 34) -> xgb.XGBClassifier:
        """Instantiate XGBClassifier configured for multiclass/binary IDS classification."""
        if self.objective is None:
            obj = "multi:softprob" if num_classes > 2 else "binary:logistic"
            metric = "mlogloss" if num_classes > 2 else "logloss"
        else:
            obj = self.objective
            metric = self.eval_metric or ("mlogloss" if num_classes > 2 else "logloss")

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective=obj,
            eval_metric=metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method=self.tree_method,
        )
        return self.model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "XGBoostBaselineModel":
        """
        Fit the XGBoost classifier on training data.
        
        Parameters
        ----------
        X_train : np.ndarray
            Standardized feature matrix.
        y_train : np.ndarray
            Integer encoded target labels.
        eval_set : list of (X, y) tuples, optional
            Validation sets for early tracking/diagnostics.
        sample_weight : np.ndarray, optional
            Per-sample training weights for class imbalance management.
        """
        num_classes = len(np.unique(y_train))
        if self.model is None:
            self.build_model(num_classes=num_classes)

        t0 = time.time()
        if eval_set:
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                sample_weight=sample_weight,
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
        self.training_time_sec = round(time.time() - t0, 2)
        
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate class label predictions."""
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate class probability estimates."""
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        return self.model.predict_proba(X)

    def get_feature_importances(
        self,
        feature_names: Optional[List[str]] = None,
        importance_type: str = "gain",
    ) -> Dict[str, float]:
        """
        Extract feature importances mapped to feature names.
        
        Parameters
        ----------
        feature_names : list of str, optional
            Feature column names.
        importance_type : str, default="gain"
            Importance metric: 'gain', 'weight', 'cover', 'total_gain', or 'total_cover'.
            
        Returns
        -------
        dict : Mapping from feature name to float importance score.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before extracting feature importances.")
            
        booster = self.model.get_booster()
        score_dict = booster.get_score(importance_type=importance_type)
        
        if feature_names is not None:
            result = {}
            for i, name in enumerate(feature_names):
                f_key = f"f{i}"
                result[name] = float(score_dict.get(f_key, score_dict.get(name, 0.0)))
            return result
        return {k: float(v) for k, v in score_dict.items()}

    def save(self, filepath: Union[str, Path] = "models/xgboost.pkl"):
        """Serialize fitted model to disk."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                    "subsample": self.subsample,
                    "colsample_bytree": self.colsample_bytree,
                    "random_state": self.random_state,
                    "training_time_sec": self.training_time_sec,
                    "xgboost_version": xgb.__version__,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(self, filepath: Union[str, Path] = "models/xgboost.pkl") -> "XGBoostBaselineModel":
        """Load serialized model from disk."""
        with open(filepath, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.n_estimators = payload["n_estimators"]
        self.max_depth = payload["max_depth"]
        self.learning_rate = payload["learning_rate"]
        self.subsample = payload["subsample"]
        self.colsample_bytree = payload["colsample_bytree"]
        self.random_state = payload["random_state"]
        self.training_time_sec = payload.get("training_time_sec", 0.0)
        return self
