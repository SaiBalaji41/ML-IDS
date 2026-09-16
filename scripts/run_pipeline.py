"""
ML-IDS Pipeline Execution Script Entrypoint.
This script will coordinate end-to-end preprocessing, training, evaluation, and explainability in future phases.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="ML-Powered Intrusion Detection System (IDS) Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to pipeline configuration YAML file",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["preprocess", "train", "evaluate", "explain", "all"],
        default="all",
        help="Pipeline phase to execute (Active only in future implementation phases)",
    )

    args = parser.parse_args()
    print(f"ML-IDS Pipeline initialized with config: {args.config}")
    print(f"Target phase: {args.phase}")
    print("STATUS: Pipeline implementation scheduled for subsequent project phases.")


if __name__ == "__main__":
    main()
