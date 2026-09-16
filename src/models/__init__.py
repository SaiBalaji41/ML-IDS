"""
ML-IDS Model Architecture Package.
"""

from src.models.random_forest import RandomForestBaselineModel
from src.models.xgboost_model import XGBoostBaselineModel

__all__ = [
    "RandomForestBaselineModel",
    "XGBoostBaselineModel",
]
