"""
Data Loading Module for CICIoT2023 Dataset.
Discovers and loads dataset files from data/raw/ using memory-efficient methods.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def discover_raw_files(raw_dir: Union[str, Path] = "data/raw") -> List[Path]:
    """
    Discover all readable raw dataset files in the raw directory (recursively).
    Supports CSV and Parquet formats.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")

    files = [
        f for f in raw_path.rglob("*")
        if f.is_file() and f.suffix.lower() in [".csv", ".parquet"]
    ]
    return sorted(files)


def load_raw_dataset(
    raw_dir: Union[str, Path] = "data/raw",
    sample_frac: Optional[float] = None,
    chunk_size: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load dataset files from data/raw/ into a consolidated DataFrame.
    
    Args:
        raw_dir: Path to directory containing raw files.
        sample_frac: Optional fraction to sample for memory conservation.
        chunk_size: Optional chunk size for large CSV stream reading.
        random_state: Random seed for reproducible sampling.
        
    Returns:
        pd.DataFrame: Consolidated raw DataFrame.
    """
    files = discover_raw_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No dataset files (.csv or .parquet) found in '{raw_dir}'. "
            "Please place the CICIoT2023 dataset in 'data/raw/'."
        )

    dfs = []
    for file_path in files:
        if file_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            if chunk_size:
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                    if sample_frac and sample_frac < 1.0:
                        chunk = chunk.sample(frac=sample_frac, random_state=random_state)
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(file_path, low_memory=False)
                if sample_frac and sample_frac < 1.0:
                    df = df.sample(frac=sample_frac, random_state=random_state)
        dfs.append(df)

    consolidated_df = pd.concat(dfs, ignore_index=True)
    # Strip any leading/trailing whitespace in column names
    consolidated_df.columns = consolidated_df.columns.str.strip()
    return consolidated_df
