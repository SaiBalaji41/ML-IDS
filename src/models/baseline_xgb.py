"""
XGBoost Baseline Classifier Module.
Aliased from src.models.xgboost_model for backwards compatibility.
"""

from src.models.xgboost_model import XGBoostBaselineModel

# Alias class name
XGBoostBaseline = XGBoostBaselineModel

__all__ = ["XGBoostBaseline", "XGBoostBaselineModel"]
