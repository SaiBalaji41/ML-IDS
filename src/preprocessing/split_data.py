"""
Data Splitting Module for CICIoT2023 Dataset.
Executes stratified Train / Validation / Test dataset partitioning.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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
    
    Args:
        X: Feature DataFrame.
        y: Target label Series.
        train_size: Proportion for training (default: 0.70).
        val_size: Proportion for validation (default: 0.15).
        test_size: Proportion for testing (default: 0.15).
        random_state: Random seed.
        stratify: Whether to maintain class proportions across splits.
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, split_metadata
    """
    assert abs((train_size + val_size + test_size) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    stratify_target = y if stratify else None

    # Step 1: Split into Train and Temp (Val + Test)
    temp_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    # Step 2: Split Temp into Validation and Test
    stratify_temp = y_temp if stratify else None
    val_ratio_in_temp = val_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_ratio_in_temp),
        random_state=random_state,
        stratify=stratify_temp,
    )

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
