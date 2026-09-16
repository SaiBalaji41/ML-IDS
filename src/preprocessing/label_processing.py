"""
Label Processing Module for CICIoT2023 Dataset.
Extracts target column, creates deterministic label mappings, and saves mappings to JSON.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd


def identify_label_column(
    df: pd.DataFrame,
    candidate_names: Optional[List[str]] = None,
) -> str:
    """
    Identify the target label column from candidate names or string categorical search.
    """
    if candidate_names is None:
        candidate_names = ["label", "Label", "attack_type", "Attack", "class", "Class", "target", "Target"]

    # Check candidate names (case-insensitive)
    col_map = {c.lower(): c for c in df.columns}
    for candidate in candidate_names:
        if candidate.lower() in col_map:
            return col_map[candidate.lower()]

    # If not found by candidate name, find last non-numeric column
    non_num_cols = df.select_dtypes(exclude=["number"]).columns
    if len(non_num_cols) > 0:
        return non_num_cols[-1]

    raise ValueError(
        "Could not automatically identify the label column. "
        f"Available columns: {list(df.columns)}"
    )


def encode_labels(
    df: pd.DataFrame,
    label_column: Optional[str] = None,
    output_mapping_path: Union[str, Path] = "data/processed/label_mapping.json",
) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Encode target label strings into deterministic integer codes.
    
    Args:
        df: Input DataFrame.
        label_column: Optional explicit label column name.
        output_mapping_path: Path to save the JSON mapping.
        
    Returns:
        Tuple[pd.Series, Dict[str, int]]: Encoded integer series and the label mapping dictionary.
    """
    if label_column is None:
        label_column = identify_label_column(df)

    # Sort unique class names deterministically, prioritizing 'Benign' as 0 if present
    unique_classes = sorted(df[label_column].astype(str).unique())
    if "Benign" in unique_classes:
        unique_classes.remove("Benign")
        unique_classes = ["Benign"] + unique_classes
    elif "benign" in unique_classes:
        unique_classes.remove("benign")
        unique_classes = ["benign"] + unique_classes

    label_mapping = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
    encoded_series = df[label_column].astype(str).map(label_mapping)

    # Save mapping to file
    out_path = Path(output_mapping_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=4)

    return encoded_series, label_mapping
