"""
Random Forest Baseline Classifier Module.
Aliased from src.models.random_forest for backwards compatibility.
"""

from src.models.random_forest import RandomForestBaselineModel

# Alias class name
RandomForestBaseline = RandomForestBaselineModel

__all__ = ["RandomForestBaseline", "RandomForestBaselineModel"]
