"""
ML-IDS Preprocessing and Feature Engineering Package.
"""

from src.preprocessing.clean_data import clean_dataset
from src.preprocessing.feature_processing import FeatureProcessor
from src.preprocessing.label_processing import encode_labels, identify_label_column
from src.preprocessing.load_data import discover_raw_files, load_raw_dataset
from src.preprocessing.preprocess_pipeline import run_pipeline
from src.preprocessing.split_data import split_dataset

__all__ = [
    "discover_raw_files",
    "load_raw_dataset",
    "clean_dataset",
    "identify_label_column",
    "encode_labels",
    "FeatureProcessor",
    "split_dataset",
    "run_pipeline",
]
