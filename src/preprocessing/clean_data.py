"""
Data Cleaning Module for CICIoT2023 Dataset.
Handles missing values, infinite values, duplicate records, and constant features.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def clean_dataset(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    drop_constant_features: bool = True,
    impute_missing_strategy: str = "median",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean the input DataFrame by handling infinities, nulls, duplicates, and constants.
    
    Args:
        df: Input raw DataFrame.
        drop_duplicates: Whether to drop exact duplicate rows.
        drop_constant_features: Whether to drop features with zero variance.
        impute_missing_strategy: Strategy for missing value imputation ('median', 'mean', 'drop').
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Cleaned DataFrame and dictionary of cleaning statistics.
    """
    cleaned_df = df.copy()
    stats = {
        "initial_rows": len(cleaned_df),
        "initial_columns": len(cleaned_df.columns),
        "infinities_replaced": 0,
        "missing_values_handled": 0,
        "duplicates_removed": 0,
        "constant_columns_dropped": [],
    }

    # 1. Standardize column names (strip whitespace)
    cleaned_df.columns = cleaned_df.columns.str.strip()

    # 2. Identify numerical columns and replace inf / -inf with NaN
    num_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(cleaned_df[num_cols].values)
    stats["infinities_replaced"] = int(np.sum(inf_mask))
    if stats["infinities_replaced"] > 0:
        cleaned_df[num_cols] = cleaned_df[num_cols].replace([np.inf, -np.inf], np.nan)

    # 3. Handle Missing Values
    null_counts = cleaned_df.isnull().sum()
    total_nulls = int(null_counts.sum())
    stats["missing_values_handled"] = total_nulls

    if total_nulls > 0:
        if impute_missing_strategy == "drop":
            cleaned_df = cleaned_df.dropna().reset_index(drop=True)
        elif impute_missing_strategy in ["median", "mean"]:
            for col in num_cols:
                if cleaned_df[col].isnull().any():
                    val = cleaned_df[col].median() if impute_missing_strategy == "median" else cleaned_df[col].mean()
                    cleaned_df[col] = cleaned_df[col].fillna(val)
            # Categorical fill with mode
            cat_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
            for col in cat_cols:
                if cleaned_df[col].isnull().any():
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])

    # 4. Handle Duplicates
    if drop_duplicates:
        dup_count = int(cleaned_df.duplicated().sum())
        stats["duplicates_removed"] = dup_count
        if dup_count > 0:
            cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

    # 5. Drop Constant Features (Zero variance)
    if drop_constant_features:
        constant_cols = [col for col in num_cols if col in cleaned_df.columns and cleaned_df[col].nunique() <= 1]
        if constant_cols:
            cleaned_df = cleaned_df.drop(columns=constant_cols)
            stats["constant_columns_dropped"] = constant_cols

    stats["final_rows"] = len(cleaned_df)
    stats["final_columns"] = len(cleaned_df.columns)
    return cleaned_df, stats
