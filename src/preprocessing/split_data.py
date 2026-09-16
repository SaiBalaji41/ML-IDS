"""
Data Splitting Module for CICIoT2023 Dataset.
Executes stratified Train / Validation / Test dataset partitioning.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def _pure_stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int = 42):
    """Pure pandas/numpy stratified split fallback."""
    rng = np.random.default_rng(random_state)
    train_indices = []
    test_indices = []
    
    for label, group in y.groupby(y):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        n_test = max(1, int(len(indices) * test_size))
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])
        
    return X.loc[train_indices], X.loc[test_indices], y.loc[train_indices], y.loc[test_indices]


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, Dict]:
    """
    Split feature matrix X and target y into stratified Train, Validation, and Test sets.
    """
    assert abs((train_size + val_size + test_size) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    temp_size = val_size + test_size
    if HAS_SKLEARN:
        stratify_target = y if stratify else None
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=temp_size, random_state=random_state, stratify=stratify_target
        )
        val_ratio_in_temp = val_size / temp_size
        stratify_temp = y_temp if stratify else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1.0 - val_ratio_in_temp), random_state=random_state, stratify=stratify_temp
        )
    else:
        X_train, X_temp, y_train, y_temp = _pure_stratified_split(X, y, test_size=temp_size, random_state=random_state)
        val_ratio_in_temp = val_size / temp_size
        X_val, X_test, y_val, y_test = _pure_stratified_split(X_temp, y_temp, test_size=(1.0 - val_ratio_in_temp), random_state=random_state)

    metadata = {
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "train_ratio": train_size,
        "val_ratio": val_size,
        "test_ratio": test_size,
        "random_state": random_state,
        "stratified": stratify,
    }
    return X_train, X_val, X_test, y_train, y_val, y_test, metadata
