"""
Training and Evaluation Script for Random Forest Baseline Model (Phase 6).

Executes reproducible training, validation diagnostics, single final test evaluation,
confusion matrix generation, feature importance ranking, and model artifact persistence.

Usage:
    python scripts/train_random_forest.py --data-dir data/processed --n-estimators 200
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.random_forest import RandomForestBaselineModel
from src.preprocessing.verify_split import load_partition


def calculate_metrics_and_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str]
) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray]:
    """
    Calculate accuracy, macro & weighted precision, recall, F1, and per-class reports.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.
    target_names : List[str]
        Ordered list of class names corresponding to integer indices [0, num_classes-1].
        
    Returns
    -------
    metrics : Dict[str, float]
        Overall classification metrics.
    report_df : pd.DataFrame
        Per-class classification metrics including Precision, Recall, F1, and Support.
    cm : np.ndarray
        Confusion matrix of shape (num_classes, num_classes).
    """
    num_classes = len(target_names)
    total_samples = len(y_true)
    accuracy = float(np.mean(y_true == y_pred))

    # Initialize confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    per_class_rows = []
    precisions = []
    recalls = []
    f1_scores = []
    supports = []

    for c in range(num_classes):
        class_name = target_names[c]
        tp = int(cm[c, c])
        fp = int(np.sum(cm[:, c]) - tp)
        fn = int(np.sum(cm[c, :]) - tp)
        support = int(np.sum(cm[c, :]))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)

        per_class_rows.append({
            "Class_Index": c,
            "Class_Name": class_name,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1_Score": round(f1, 4),
            "Support": support,
        })

    # Macro Averages (unweighted mean across all classes)
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1_scores))

    # Weighted Averages (weighted by support of each class)
    weights = np.array(supports, dtype=np.float64) / total_samples if total_samples > 0 else np.zeros(num_classes)
    weighted_precision = float(np.sum(np.array(precisions) * weights))
    weighted_recall = float(np.sum(np.array(recalls) * weights))
    weighted_f1 = float(np.sum(np.array(f1_scores) * weights))

    metrics = {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_precision, 4),
        "weighted_recall": round(weighted_recall, 4),
        "weighted_f1": round(weighted_f1, 4),
        "total_samples": total_samples,
    }

    report_df = pd.DataFrame(per_class_rows)
    return metrics, report_df, cm


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    """Generate and save a confusion matrix heatmap visualization."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(18, 15))
    
    # Normalize confusion matrix for color intensity while showing counts
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    sns.heatmap(
        cm_norm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Prediction Ratio'}
    )
    plt.title("Random Forest Baseline — Test Set Confusion Matrix (Normalized)", fontsize=16, pad=15)
    plt.xlabel("Predicted Label", fontsize=13, labelpad=10)
    plt.ylabel("Ground Truth Label", fontsize=13, labelpad=10)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"      [SAVED] Confusion matrix plot -> {save_path}")


def plot_feature_importance(feat_imp_df: pd.DataFrame, save_path: Path, top_n: int = 25):
    """Generate and save horizontal bar chart of top Gini feature importances."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    top_df = feat_imp_df.head(top_n).iloc[::-1]  # Invert so highest is at top
    
    plt.figure(figsize=(12, 10))
    bars = plt.barh(top_df["Feature"], top_df["Importance"], color="#1f77b4", edgecolor="#0e4370")
    plt.title(f"Top {top_n} Most Discriminative Features — Random Forest Baseline", fontsize=15, pad=15)
    plt.xlabel("Gini Feature Importance", fontsize=12, labelpad=10)
    plt.ylabel("Network Flow Feature", fontsize=12, labelpad=10)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2, f"{width:.4f}", va='center', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"      [SAVED] Feature importance plot -> {save_path}")


def train_and_evaluate_random_forest(
    data_dir: str = "data/processed",
    models_dir: str = "models",
    metrics_dir: str = "results/metrics",
    graphs_dir: str = "results/graphs",
    confusion_dir: str = "results/confusion_matrices",
    n_estimators: int = 200,
    max_depth: int = 25,
    max_samples: float = 0.1,
    class_weight: str = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Dict:
    """
    Execute full training and evaluation of Random Forest baseline.
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

    print("=" * 80)
    print("      ML-IDS: RANDOM FOREST BASELINE MODEL TRAINING & EVALUATION (PHASE 6)")
    print("=" * 80)

    # 1. Verification of Data Partitions
    print("\n[STEP 1/8] Verifying and loading processed datasets...")
    train_file = base_path / "train" / "train.npz"
    val_file = base_path / "validation" / "val.npz"
    test_file = base_path / "test" / "test.npz"

    for p_name, p_file in [("Train", train_file), ("Validation", val_file), ("Test", test_file)]:
        if not p_file.exists():
            raise FileNotFoundError(f"Missing required partition: {p_file}")

    X_train, y_train = load_partition(base_path / "train", "train")
    X_val, y_val = load_partition(base_path / "validation", "val")
    X_test, y_test = load_partition(base_path / "test", "test")

    # Load label mapping
    mapping_file = base_path / "label_mapping.json"
    if not mapping_file.exists():
        raise FileNotFoundError(f"Missing label mapping file: {mapping_file}")
    with open(mapping_file, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)

    # Load metadata
    meta_file = base_path / "preprocessing_metadata.json"
    feature_names = []
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            p_meta = json.load(f)
            feature_names = p_meta.get("feature_names", [])
    if not feature_names:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    inv_mapping = {v: k for k, v in label_mapping.items()}
    target_names = [inv_mapping[i] for i in range(len(label_mapping))]

    print(f"      Classification Type: Multiclass Classification ({len(target_names)} distinct classes)")
    print(f"      Training Set:        {len(X_train):,} samples | {X_train.shape[1]} features")
    print(f"      Validation Set:      {len(X_val):,} samples | {X_val.shape[1]} features")
    print(f"      Test Set:            {len(X_test):,} samples | {X_test.shape[1]} features")

    # 2. Configure Random Forest
    print(f"\n[STEP 2/8] Configuring Random Forest Classifier...")
    print(f"      - n_estimators:      {n_estimators}")
    print(f"      - max_depth:         {max_depth}")
    print(f"      - max_samples:       {max_samples} (subsample per tree)")
    print(f"      - class_weight:      {class_weight} (Justified by 6,000:1 class imbalance)")
    print(f"      - random_state:      {random_state}")
    print(f"      - n_jobs:            {n_jobs}")

    rf_model = RandomForestBaselineModel(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight=class_weight,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    # 3. Train Model strictly on Training Set
    print("\n[STEP 3/8] Training Random Forest model on training partition (X_train, y_train)...")
    t0_train = time.time()
    rf_model.fit(X_train, y_train)
    train_duration = round(time.time() - t0_train, 2)
    print(f"      Training successfully completed in {train_duration:.2f} seconds.")

    # 4. Diagnostic Evaluation on Validation Set
    print("\n[STEP 4/8] Evaluating on Validation set (diagnostic evaluation)...")
    val_sample_size = len(X_val)
    t0_val = time.time()
    y_val_pred = rf_model.predict(X_val)
    t1_val = time.time()
    val_inference_time = round(t1_val - t0_val, 4)

    val_metrics, val_report_df, cm_val = calculate_metrics_and_report(y_val, y_val_pred, target_names)
    val_metrics["model"] = "Random Forest"
    val_metrics["split"] = "validation"
    val_metrics["inference_time_sec"] = val_inference_time
    val_metrics["samples_evaluated"] = val_sample_size

    val_json_path = metric_path / "random_forest_validation.json"
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=4)
    print(f"      [SAVED] Validation metrics -> {val_json_path}")
    print(f"      Validation Accuracy:          {val_metrics['accuracy'] * 100:.2f}%")
    print(f"      Validation Weighted F1-Score: {val_metrics['weighted_f1']:.4f}")
    print(f"      Validation Macro F1-Score:    {val_metrics['macro_f1']:.4f}")
    print(f"      Validation Inference Latency: {val_inference_time:.4f}s ({val_sample_size:,} samples)")

    # 5. Final Single Evaluation on Test Set
    print("\n[STEP 5/8] Performing single final evaluation on Test set (X_test, y_test)...")
    test_sample_size = len(X_test)
    t0_test = time.time()
    y_test_pred = rf_model.predict(X_test)
    t1_test = time.time()
    test_inference_time = round(t1_test - t0_test, 4)

    test_metrics, test_report_df, cm_test = calculate_metrics_and_report(y_test, y_test_pred, target_names)
    test_metrics["model"] = "Random Forest"
    test_metrics["split"] = "test"
    test_metrics["training_time_sec"] = train_duration
    test_metrics["inference_time_sec"] = test_inference_time
    test_metrics["samples_evaluated"] = test_sample_size

    test_json_path = metric_path / "random_forest_test.json"
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)
    print(f"      [SAVED] Test metrics -> {test_json_path}")

    # Save per-class classification report
    class_report_path = metric_path / "random_forest_classification_report.csv"
    test_report_df.to_csv(class_report_path, index=False)
    print(f"      [SAVED] Per-class classification report -> {class_report_path}")

    print(f"      Test Accuracy:          {test_metrics['accuracy'] * 100:.2f}%")
    print(f"      Test Weighted Precision:{test_metrics['weighted_precision']:.4f}")
    print(f"      Test Weighted Recall:   {test_metrics['weighted_recall']:.4f}")
    print(f"      Test Weighted F1-Score: {test_metrics['weighted_f1']:.4f}")
    print(f"      Test Macro Precision:   {test_metrics['macro_precision']:.4f}")
    print(f"      Test Macro Recall:      {test_metrics['macro_recall']:.4f}")
    print(f"      Test Macro F1-Score:    {test_metrics['macro_f1']:.4f}")
    print(f"      Test Inference Latency: {test_inference_time:.4f}s ({test_sample_size:,} samples)")

    # 6. Confusion Matrix & Misclassifications
    print("\n[STEP 6/8] Generating Confusion Matrix & Misclassification analysis...")
    cm_png_path = conf_path / "random_forest_confusion_matrix.png"
    plot_confusion_matrix(cm_test, target_names, cm_png_path)

    # Extract misclassified samples
    misclassified_mask = y_test != y_test_pred
    misclassified_indices = np.where(misclassified_mask)[0]
    
    # Save a manageable, informative summary of misclassifications
    max_misclass_to_save = min(len(misclassified_indices), 50000)
    sampled_misclass_idx = misclassified_indices[:max_misclass_to_save]
    
    misclass_df = pd.DataFrame({
        "Test_Sample_Index": sampled_misclass_idx,
        "True_Label_Index": y_test[sampled_misclass_idx],
        "True_Class_Name": [inv_mapping[int(idx)] for idx in y_test[sampled_misclass_idx]],
        "Predicted_Label_Index": y_test_pred[sampled_misclass_idx],
        "Predicted_Class_Name": [inv_mapping[int(idx)] for idx in y_test_pred[sampled_misclass_idx]],
    })
    misclass_csv_path = metric_path / "random_forest_misclassifications.csv"
    misclass_df.to_csv(misclass_csv_path, index=False)
    print(f"      [SAVED] Misclassifications ({len(misclassified_indices):,} total errors, {len(misclass_df):,} saved) -> {misclass_csv_path}")

    # 7. Feature Importance Extraction
    print("\n[STEP 7/8] Calculating Gini feature importances...")
    feat_importances = rf_model.get_feature_importances(feature_names=feature_names)
    feat_imp_df = pd.DataFrame(
        list(feat_importances.items()),
        columns=["Feature", "Importance"]
    ).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    feat_imp_csv_path = metric_path / "random_forest_feature_importance.csv"
    feat_imp_df.to_csv(feat_imp_csv_path, index=False)
    print(f"      [SAVED] Feature importances -> {feat_imp_csv_path}")

    feat_imp_png_path = graph_path / "random_forest_feature_importance.png"
    plot_feature_importance(feat_imp_df, feat_imp_png_path, top_n=25)

    # 8. Model Serialization, Metadata & Comparison Table Update
    print("\n[STEP 8/8] Serializing Model, Metadata, and updating Model Comparison Table...")
    model_file_path = model_path / "random_forest.pkl"
    rf_model.save(model_file_path)
    model_size_mb = round(os.path.getsize(model_file_path) / (1024 * 1024), 2)

    meta_payload = {
        "model_type": "Random Forest",
        "library": "scikit-learn",
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "random_seed": random_state,
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": class_weight,
            "max_samples": max_samples,
            "n_jobs": n_jobs,
        },
        "number_of_features": int(X_train.shape[1]),
        "feature_names": feature_names,
        "number_of_classes": len(target_names),
        "label_mapping": label_mapping,
        "training_sample_count": len(X_train),
        "validation_sample_count": len(X_val),
        "test_sample_count": len(X_test),
        "training_time_seconds": train_duration,
        "validation_inference_time_seconds": val_inference_time,
        "test_inference_time_seconds": test_inference_time,
        "model_file_size_mb": model_size_mb,
        "preprocessing_artifacts": {
            "scaler": "models/preprocessing/scaler.pkl",
            "label_mapping": "data/processed/label_mapping.json",
            "preprocessing_metadata": "data/processed/preprocessing_metadata.json"
        },
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_precision": test_metrics["weighted_precision"],
            "weighted_recall": test_metrics["weighted_recall"],
            "weighted_f1": test_metrics["weighted_f1"],
        }
    }
    meta_file_path = model_path / "random_forest_metadata.json"
    with open(meta_file_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=4)
    print(f"      [SAVED] Model artifact -> {model_file_path} ({model_size_mb} MB)")
    print(f"      [SAVED] Model metadata -> {meta_file_path}")

    # Model comparison table (ONLY Random Forest row)
    comp_file_path = metric_path / "model_comparison.csv"
    comp_row = {
        "Model": "Random Forest",
        "Accuracy": round(test_metrics["accuracy"], 4),
        "Precision Macro": round(test_metrics["macro_precision"], 4),
        "Recall Macro": round(test_metrics["macro_recall"], 4),
        "F1 Macro": round(test_metrics["macro_f1"], 4),
        "Precision Weighted": round(test_metrics["weighted_precision"], 4),
        "Recall Weighted": round(test_metrics["weighted_recall"], 4),
        "F1 Weighted": round(test_metrics["weighted_f1"], 4),
        "Training Time": round(train_duration, 2),
        "Inference Time": round(test_inference_time, 4),
    }

    # Filter out previous Random Forest row if exists, maintain exact schema
    if comp_file_path.exists():
        comp_df = pd.read_csv(comp_file_path)
        comp_df = comp_df[comp_df["Model"] != "Random Forest"]
        comp_df = pd.concat([pd.DataFrame([comp_row]), comp_df], ignore_index=True)
    else:
        comp_df = pd.DataFrame([comp_row])

    # Reorder columns explicitly as required
    required_cols = [
        "Model",
        "Accuracy",
        "Precision Macro",
        "Recall Macro",
        "F1 Macro",
        "Precision Weighted",
        "Recall Weighted",
        "F1 Weighted",
        "Training Time",
        "Inference Time"
    ]
    # Keep only required columns
    comp_df = comp_df[[c for c in required_cols if c in comp_df.columns]]
    comp_df.to_csv(comp_file_path, index=False)
    print(f"      [SAVED] Model comparison table -> {comp_file_path}")

    print("\n" + "=" * 80)
    print("STATUS: Phase 6 Random Forest Baseline Training & Evaluation Completed Successfully.")
    print("=" * 80)
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate Random Forest Baseline for ML-IDS")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed dataset directory")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics", help="Directory to save metrics")
    parser.add_argument("--graphs-dir", type=str, default="results/graphs", help="Directory to save graphs")
    parser.add_argument("--confusion-dir", type=str, default="results/confusion_matrices", help="Directory to save confusion matrices")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of decision trees (default: 200)")
    parser.add_argument("--max-depth", type=int, default=25, help="Maximum tree depth (default: 25)")
    parser.add_argument("--max-samples", type=float, default=0.1, help="Bootstrap subsample fraction per tree (default: 0.1, i.e. 549k samples per tree)")
    parser.add_argument("--class-weight", type=str, default="balanced", help="Class weighting scheme ('balanced' or None)")
    parser.add_argument("--random-seed", type=int, default=42, help="Fixed random state (default: 42)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of CPU worker jobs (-1 for all cores)")

    args = parser.parse_args()

    class_wt = None if args.class_weight.lower() in ("none", "null") else args.class_weight

    try:
        train_and_evaluate_random_forest(
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            metrics_dir=args.metrics_dir,
            graphs_dir=args.graphs_dir,
            confusion_dir=args.confusion_dir,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            max_samples=args.max_samples,
            class_weight=class_wt,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
        )
    except Exception as e:
        print(f"\n[FATAL ERROR] Random Forest pipeline execution failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
