"""
XGBoost Baseline Classifier Module for ML-IDS.
Wraps XGBoost's XGBClassifier with dynamic multiclass/binary objective handling and logging.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except (ImportError, Exception):
    XGB_AVAILABLE = False

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


class XGBoostBaselineModel:
    """
    XGBoost gradient boosted decision trees baseline for network intrusion detection.
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
        tree_method: str = "auto",
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
        self.model = None

    def build_model(self, num_classes: int = 2):
        """Instantiate XGBClassifier with appropriate objective."""
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is required to build XGBoostBaselineModel.")

        if self.objective is None:
            if num_classes > 2:
                obj = "multi:softprob"
                metric = "mlogloss"
            else:
                obj = "binary:logistic"
                metric = "logloss"
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
        eval_set: Optional[List[tuple]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "XGBoostBaselineModel":
        """Fit XGBoost model on training data."""
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is required to train XGBoostBaselineModel.")

        num_classes = len(np.unique(y_train))
        if self.model is None:
            self.build_model(num_classes=num_classes)

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

    def get_feature_importances(
        self,
        feature_names: Optional[List[str]] = None,
        importance_type: str = "gain",
    ) -> Dict[str, float]:
        """Extract feature importances based on gain or weight."""
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
        import pickle
        with open(out_path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath: Union[str, Path] = "models/xgboost.pkl"):
        """Load serialized model from disk."""
        import pickle
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        return self
