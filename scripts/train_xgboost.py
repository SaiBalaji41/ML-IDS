"""
Training and Evaluation Script for XGBoost Baseline Model.

Usage:
    python scripts/train_xgboost.py --data-dir data/processed --n-estimators 300 --max-depth 6
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Support running directly or as module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.xgboost_model import XGBoostBaselineModel
from src.preprocessing.verify_split import load_partition


def train_and_evaluate_xgb(
    data_dir: str = "data/processed",
    models_dir: str = "models",
    metrics_dir: str = "results/metrics",
    graphs_dir: str = "results/graphs",
    confusion_dir: str = "results/confusion_matrices",
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Dict:
    """
    Train and evaluate the XGBoost baseline model.
    """
    base_path = Path(data_dir)
    model_path = Path(models_dir)
    metric_path = Path(metrics_dir)
    graph_path = Path(graphs_dir)
    conf_path = Path(confusion_dir)

    model_path.mkdir(parents=True, exist_ok=True)
    metric_path.mkdir(parents=True, exist_ok=True)
    graph_path.mkdir(parents=True, exist_ok=True)
    conf_path.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("      ML-IDS: XGBOOST BASELINE MODEL TRAINING & EVALUATION")
    print("=" * 75)

    # 1. Load Processed Partitions
    print("[1/7] Loading processed Train, Validation, and Test datasets...")
    X_train, y_train = load_partition(base_path / "train", "train")
    X_val, y_val = load_partition(base_path / "validation", "val")
    X_test, y_test = load_partition(base_path / "test", "test")

    # Load label mapping
    mapping_file = base_path / "label_mapping.json"
    label_mapping = {}
    if mapping_file.exists():
        with open(mapping_file, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)

    # Load feature names from scaler artifact if present
    feature_names = []
    scaler_artifact = model_path / "preprocessing" / "scaler.pkl"
    if scaler_artifact.exists():
        scaler_data = joblib.load(scaler_artifact)
        feature_names = scaler_data.get("feature_names", [])
    if not feature_names:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    inv_mapping = {v: k for k, v in label_mapping.items()} if label_mapping else {}
    target_names = [inv_mapping.get(i, f"Class_{i}") for i in sorted(np.unique(y_train))]
    num_classes = len(target_names)

    print(f"      Train Samples: {len(X_train):,} | Features: {X_train.shape[1]} | Classes: {num_classes}")
    print(f"      Val Samples:   {len(X_val):,}")
    print(f"      Test Samples:  {len(X_test):,}")

    # 2. Train Model
    print(f"[2/7] Training XGBoost (n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate})...")
    xgb_model = XGBoostBaselineModel(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    t_train_start = time.time()
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    t_train_end = time.time()
    train_duration = round(t_train_end - t_train_start, 4)
    print(f"      Training completed in {train_duration:.2f} seconds.")

    # 3. Validation Evaluation
    print("[3/7] Evaluating on Validation set...")
    t_val_start = time.time()
    y_val_pred = xgb_model.predict(X_val)
    t_val_end = time.time()
    val_latency = round(t_val_end - t_val_start, 4)

    val_acc = float(accuracy_score(y_val, y_val_pred))
    val_prec_w = float(precision_score(y_val, y_val_pred, average="weighted", zero_division=0))
    val_rec_w = float(recall_score(y_val, y_val_pred, average="weighted", zero_division=0))
    val_f1_w = float(f1_score(y_val, y_val_pred, average="weighted", zero_division=0))
    val_f1_m = float(f1_score(y_val, y_val_pred, average="macro", zero_division=0))

    val_metrics = {
        "model": "XGBoost",
        "split": "validation",
        "accuracy": val_acc,
        "weighted_precision": val_prec_w,
        "weighted_recall": val_rec_w,
        "weighted_f1": val_f1_w,
        "macro_f1": val_f1_m,
        "inference_time_sec": val_latency,
    }
    with open(metric_path / "xgboost_validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=4)

    val_report_dict = classification_report(y_val, y_val_pred, target_names=target_names, output_dict=True, zero_division=0)
    pd.DataFrame(val_report_dict).transpose().to_csv(metric_path / "xgboost_validation_report.csv")
    print(f"      Val Accuracy: {val_acc*100:.2f}% | Val Weighted F1: {val_f1_w:.4f} | Val Macro F1: {val_f1_m:.4f}")

    # 4. Test Evaluation
    print("[4/7] Evaluating on Test set (Single final evaluation)...")
    t_test_start = time.time()
    y_test_pred = xgb_model.predict(X_test)
    t_test_end = time.time()
    test_latency = round(t_test_end - t_test_start, 4)

    test_acc = float(accuracy_score(y_test, y_test_pred))
    test_prec_w = float(precision_score(y_test, y_test_pred, average="weighted", zero_division=0))
    test_rec_w = float(recall_score(y_test, y_test_pred, average="weighted", zero_division=0))
    test_f1_w = float(f1_score(y_test, y_test_pred, average="weighted", zero_division=0))

    test_prec_m = float(precision_score(y_test, y_test_pred, average="macro", zero_division=0))
    test_rec_m = float(recall_score(y_test, y_test_pred, average="macro", zero_division=0))
    test_f1_m = float(f1_score(y_test, y_test_pred, average="macro", zero_division=0))

    test_metrics = {
        "model": "XGBoost",
        "split": "test",
        "accuracy": test_acc,
        "weighted_precision": test_prec_w,
        "weighted_recall": test_rec_w,
        "weighted_f1": test_f1_w,
        "macro_precision": test_prec_m,
        "macro_recall": test_rec_m,
        "macro_f1": test_f1_m,
        "training_time_sec": train_duration,
        "test_inference_time_sec": test_latency,
    }
    with open(metric_path / "xgboost_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)

    test_report_dict = classification_report(y_test, y_test_pred, target_names=target_names, output_dict=True, zero_division=0)
    test_report_df = pd.DataFrame(test_report_dict).transpose()
    test_report_df.to_csv(metric_path / "xgboost_test_report.csv")
    test_report_df.to_csv(metric_path / "xgboost_per_class.csv")
    print(f"      Test Accuracy: {test_acc*100:.2f}% | Test Weighted F1: {test_f1_w:.4f} | Test Macro F1: {test_f1_m:.4f}")

    # 5. Confusion Matrices & Visualizations
    print("[5/7] Generating and saving confusion matrix heatmaps...")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Validation Confusion Matrix
        cm_val = confusion_matrix(y_val, y_val_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_val, annot=True, fmt="d", cmap="Greens", xticklabels=target_names, yticklabels=target_names)
        plt.title("XGBoost — Validation Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(conf_path / "xgboost_validation.png", dpi=300)
        plt.close()

        # Test Confusion Matrix
        cm_test = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens", xticklabels=target_names, yticklabels=target_names)
        plt.title("XGBoost Baseline — Test Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(conf_path / "xgboost_test.png", dpi=300)
        plt.close()
        print(f"      Confusion matrices saved to {conf_path}")
    except Exception as e:
        print(f"      [WARN] Confusion matrix plotting skipped: {e}")

    # 6. Feature Importance Extraction (Gain-based)
    print("[6/7] Extracting gain-based feature importances...")
    feat_imp = xgb_model.get_feature_importances(feature_names=feature_names, importance_type="gain")
    feat_imp_df = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance_Gain"]).sort_values(by="Importance_Gain", ascending=False)
    feat_imp_df.to_csv(metric_path / "xgboost_feature_importance.csv", index=False)

    try:
        import matplotlib.pyplot as plt
        top20 = feat_imp_df.head(20)
        plt.figure(figsize=(10, 6))
        plt.barh(top20["Feature"][::-1], top20["Importance_Gain"][::-1], color="#2d7f5e")
        plt.title("XGBoost — Top 20 Feature Importances (Gain)")
        plt.xlabel("Gain")
        plt.tight_layout()
        plt.savefig(graph_path / "xgboost_feature_importance.png", dpi=300)
        plt.close()
        print(f"      Feature importance plot saved to {graph_path / 'xgboost_feature_importance.png'}")
    except Exception as e:
        print(f"      [WARN] Feature importance plotting skipped: {e}")

    # Misclassifications analysis
    misclassified_indices = np.where(y_test != y_test_pred)[0]
    misclass_df = pd.DataFrame({
        "Sample_Index": misclassified_indices,
        "True_Class": [inv_mapping.get(y_test[i], str(y_test[i])) for i in misclassified_indices],
        "Predicted_Class": [inv_mapping.get(y_test_pred[i], str(y_test_pred[i])) for i in misclassified_indices],
    })
    misclass_df.to_csv(metric_path / "xgboost_misclassifications.csv", index=False)

    # Runtime records
    runtime_records = {
        "model": "XGBoost",
        "training_time_seconds": train_duration,
        "val_inference_time_seconds": val_latency,
        "test_inference_time_seconds": test_latency,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
    }
    with open(metric_path / "xgboost_runtime.json", "w", encoding="utf-8") as f:
        json.dump(runtime_records, f, indent=4)

    # Update Model Comparison Table (Preserving Random Forest)
    comp_file = metric_path / "model_comparison.csv"
    comp_row = {
        "Model": "XGBoost",
        "Accuracy": round(test_acc, 4),
        "Precision": round(test_prec_w, 4),
        "Recall": round(test_rec_w, 4),
        "F1": round(test_f1_w, 4),
        "Macro Precision": round(test_prec_m, 4),
        "Macro Recall": round(test_rec_m, 4),
        "Macro F1": round(test_f1_m, 4),
        "Weighted Precision": round(test_prec_w, 4),
        "Weighted Recall": round(test_rec_w, 4),
        "Weighted F1": round(test_f1_w, 4),
    }
    if comp_file.exists():
        comp_df = pd.read_csv(comp_file)
        comp_df = comp_df[comp_df["Model"] != "XGBoost"]
        comp_df = pd.concat([comp_df, pd.DataFrame([comp_row])], ignore_index=True)
    else:
        comp_df = pd.DataFrame([comp_row])
    comp_df.to_csv(comp_file, index=False)

    # 7. Save Model & Metadata
    print("[7/7] Serializing trained model and metadata...")
    xgb_model.save(model_path / "xgboost.pkl")
    metadata = {
        "model_type": "XGBoost",
        "timestamp": datetime.now().isoformat(),
        "random_state": random_state,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "n_features": int(X_train.shape[1]),
        "feature_names": feature_names,
        "n_classes": num_classes,
        "class_names": target_names,
        "training_samples": len(X_train),
        "test_accuracy": test_acc,
        "test_weighted_f1": test_f1_w,
        "test_macro_f1": test_f1_m,
    }
    with open(model_path / "xgboost_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"      Model saved to: {model_path / 'xgboost.pkl'}")
    print(f"      Metadata saved to: {model_path / 'xgboost_metadata.json'}")
    print("=" * 75)
    print("STATUS: XGBoost Baseline Training & Evaluation Completed.")
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate XGBoost Baseline")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed dataset directory")
    parser.add_argument("--models-dir", type=str, default="models", help="Model saving directory")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics", help="Metrics output directory")
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of boosting rounds")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--random-seed", type=int, default=42, help="Random state seed")

    args = parser.parse_args()
    try:
        train_and_evaluate_xgb(
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            metrics_dir=args.metrics_dir,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=args.random_seed,
        )
    except Exception as e:
        print(f"\n[ERROR] XGBoost training halted: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
