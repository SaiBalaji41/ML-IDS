"""
Training and Evaluation Script for XGBoost Baseline Model (Phase 7).

Executes reproducible training, validation diagnostics, single final test evaluation,
confusion matrix generation, feature importance ranking, misclassification analysis,
and model artifact persistence.

Usage:
    python scripts/train_xgboost.py --data-dir data/processed --n-estimators 100 --max-samples 750000
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
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.xgboost_model import XGBoostBaselineModel
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
        Ordered list of class names.
        
    Returns
    -------
    metrics : Dict[str, float]
        Overall classification metrics.
    report_df : pd.DataFrame
        Per-class classification metrics.
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

    # Macro Averages
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1_scores))

    # Weighted Averages
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
    """Generate and save normalized confusion matrix heatmap visualization."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(18, 15))
    
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    sns.heatmap(
        cm_norm,
        annot=False,
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Prediction Ratio'}
    )
    plt.title("XGBoost Baseline — Test Set Confusion Matrix (Normalized)", fontsize=16, pad=15)
    plt.xlabel("Predicted Label", fontsize=13, labelpad=10)
    plt.ylabel("Ground Truth Label", fontsize=13, labelpad=10)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"      [SAVED] Confusion matrix plot -> {save_path}")


def plot_feature_importance(feat_imp_df: pd.DataFrame, save_path: Path, top_n: int = 25):
    """Generate and save horizontal bar chart of top XGBoost feature importances (Gain)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    top_df = feat_imp_df.head(top_n).iloc[::-1]
    
    plt.figure(figsize=(12, 10))
    bars = plt.barh(top_df["Feature"], top_df["Importance_Gain"], color="#2ca02c", edgecolor="#1b611b")
    plt.title(f"Top {top_n} Most Discriminative Features — XGBoost (Average Gain)", fontsize=15, pad=15)
    plt.xlabel("Average Feature Gain Across Splits", fontsize=12, labelpad=10)
    plt.ylabel("Network Flow Feature", fontsize=12, labelpad=10)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + (0.01 * width), bar.get_y() + bar.get_height() / 2, f"{width:.2f}", va='center', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"      [SAVED] Feature importance plot -> {save_path}")


def train_and_evaluate_xgboost(
    data_dir: str = "data/processed",
    models_dir: str = "models",
    metrics_dir: str = "results/metrics",
    graphs_dir: str = "results/graphs",
    confusion_dir: str = "results/confusion_matrices",
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    max_train_samples: int = 750000,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Dict:
    """
    Execute training and evaluation of XGBoost baseline.
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
    print("      ML-IDS: XGBOOST BASELINE MODEL TRAINING & EVALUATION (PHASE 7)")
    print("=" * 80)

    # 1. Verification of Data Partitions
    print("\n[STEP 1/8] Verifying and loading processed datasets...")
    train_file = base_path / "train" / "train.npz"
    val_file = base_path / "validation" / "val.npz"
    test_file = base_path / "test" / "test.npz"

    for p_name, p_file in [("Train", train_file), ("Validation", val_file), ("Test", test_file)]:
        if not p_file.exists():
            raise FileNotFoundError(f"Missing required partition: {p_file}")

    X_train_full, y_train_full = load_partition(base_path / "train", "train")
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
        feature_names = [f"feature_{i}" for i in range(X_train_full.shape[1])]

    inv_mapping = {v: k for k, v in label_mapping.items()}
    target_names = [inv_mapping[i] for i in range(len(label_mapping))]

    print(f"      XGBoost Version:     {xgb.__version__}")
    print(f"      Classification Type: Multiclass Classification ({len(target_names)} distinct classes)")
    print(f"      Total Train Samples: {len(X_train_full):,} | Features: {X_train_full.shape[1]}")
    print(f"      Validation Samples:  {len(X_val):,} | Features: {X_val.shape[1]}")
    print(f"      Test Samples:        {len(X_test):,} | Features: {X_test.shape[1]}")

    # 2. Stratified Subsampling & Class Imbalance Weighting (Training Set Only)
    print(f"\n[STEP 2/8] Preparing stratified training partition and balanced sample weights...")
    if max_train_samples and max_train_samples < len(X_train_full):
        rng = np.random.default_rng(random_state)
        sample_indices = []
        for c in range(len(target_names)):
            c_idx = np.where(y_train_full == c)[0]
            if len(c_idx) == 0:
                continue
            n_target = max(min(len(c_idx), 150), int(len(c_idx) * (max_train_samples / len(y_train_full))))
            sample_indices.extend(rng.choice(c_idx, size=min(n_target, len(c_idx)), replace=False))
        sample_indices = np.array(sample_indices)
        rng.shuffle(sample_indices)
        X_train = X_train_full[sample_indices]
        y_train = y_train_full[sample_indices]
        print(f"      Stratified training sample: {len(X_train):,} samples across all {len(target_names)} classes.")
    else:
        X_train, y_train = X_train_full, y_train_full

    # Compute balanced weights strictly on training set
    train_sample_weights = compute_sample_weight("balanced", y_train)
    print(f"      Balanced sample weights computed exclusively on training set (Min weight: {train_sample_weights.min():.4f}, Max weight: {train_sample_weights.max():.4f}).")

    # 3. Configure XGBoost Baseline Model
    print(f"\n[STEP 3/8] Configuring XGBoost Baseline Classifier...")
    print(f"      - n_estimators:      {n_estimators}")
    print(f"      - max_depth:         {max_depth}")
    print(f"      - learning_rate:     {learning_rate}")
    print(f"      - subsample:         {subsample}")
    print(f"      - colsample_bytree:  {colsample_bytree}")
    print(f"      - objective:         multi:softprob")
    print(f"      - eval_metric:       mlogloss")
    print(f"      - tree_method:       hist (Fast CPU Histogram)")
    print(f"      - random_state:      {random_state}")
    print(f"      - n_jobs:            {n_jobs}")

    xgb_model = XGBoostBaselineModel(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",
    )

    # 4. Train Model on Training Partition
    print("\n[STEP 4/8] Fitting XGBoost model on training set...")
    t0_train = time.time()
    xgb_model.fit(X_train, y_train, sample_weight=train_sample_weights)
    train_duration = round(time.time() - t0_train, 2)
    print(f"      Training completed in {train_duration:.2f} seconds.")

    # 5. Diagnostic Evaluation on Validation Set
    print("\n[STEP 5/8] Evaluating on Validation set (diagnostic evaluation)...")
    val_sample_size = len(X_val)
    t0_val = time.time()
    y_val_pred = xgb_model.predict(X_val)
    t1_val = time.time()
    val_inference_time = round(t1_val - t0_val, 4)

    val_metrics, val_report_df, cm_val = calculate_metrics_and_report(y_val, y_val_pred, target_names)
    val_metrics["model"] = "XGBoost"
    val_metrics["split"] = "validation"
    val_metrics["inference_time_sec"] = val_inference_time
    val_metrics["samples_evaluated"] = val_sample_size

    val_json_path = metric_path / "xgboost_validation.json"
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=4)
    print(f"      [SAVED] Validation metrics -> {val_json_path}")
    print(f"      Validation Accuracy:          {val_metrics['accuracy'] * 100:.2f}%")
    print(f"      Validation Weighted F1-Score: {val_metrics['weighted_f1']:.4f}")
    print(f"      Validation Macro F1-Score:    {val_metrics['macro_f1']:.4f}")
    print(f"      Validation Inference Latency: {val_inference_time:.4f}s ({val_sample_size:,} samples)")

    # 6. Single Final Evaluation on Test Set
    print("\n[STEP 6/8] Performing single final evaluation on Test set (X_test, y_test)...")
    test_sample_size = len(X_test)
    t0_test = time.time()
    y_test_pred = xgb_model.predict(X_test)
    t1_test = time.time()
    test_inference_time = round(t1_test - t0_test, 4)

    test_metrics, test_report_df, cm_test = calculate_metrics_and_report(y_test, y_test_pred, target_names)
    test_metrics["model"] = "XGBoost"
    test_metrics["split"] = "test"
    test_metrics["training_time_sec"] = train_duration
    test_metrics["inference_time_sec"] = test_inference_time
    test_metrics["samples_evaluated"] = test_sample_size

    test_json_path = metric_path / "xgboost_test.json"
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)
    print(f"      [SAVED] Test metrics -> {test_json_path}")

    class_report_path = metric_path / "xgboost_classification_report.csv"
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

    # 7. Confusion Matrix & Misclassification Analysis
    print("\n[STEP 7/8] Generating Confusion Matrix & Misclassification breakdown...")
    cm_png_path = conf_path / "xgboost_confusion_matrix.png"
    plot_confusion_matrix(cm_test, target_names, cm_png_path)

    misclassified_mask = y_test != y_test_pred
    misclassified_indices = np.where(misclassified_mask)[0]
    
    max_misclass_to_save = min(len(misclassified_indices), 50000)
    sampled_misclass_idx = misclassified_indices[:max_misclass_to_save]
    
    misclass_df = pd.DataFrame({
        "Test_Sample_Index": sampled_misclass_idx,
        "True_Label_Index": y_test[sampled_misclass_idx],
        "True_Class_Name": [inv_mapping[int(idx)] for idx in y_test[sampled_misclass_idx]],
        "Predicted_Label_Index": y_test_pred[sampled_misclass_idx],
        "Predicted_Class_Name": [inv_mapping[int(idx)] for idx in y_test_pred[sampled_misclass_idx]],
    })
    misclass_csv_path = metric_path / "xgboost_misclassifications.csv"
    misclass_df.to_csv(misclass_csv_path, index=False)
    print(f"      [SAVED] Misclassifications ({len(misclassified_indices):,} total errors, {len(misclass_df):,} saved) -> {misclass_csv_path}")

    # Identify most frequent confusion pairs
    top_confusion_pairs = misclass_df.groupby(["True_Class_Name", "Predicted_Class_Name"]).size().reset_index(name="Count").sort_values(by="Count", ascending=False).head(10)
    print(f"      Top Confusion Pairs:\n{top_confusion_pairs.to_string(index=False)}")

    # 8. Feature Importance (Gain and Weight)
    print("\n[STEP 8/8] Calculating Feature Importances & Updating Comparison Table...")
    feat_importances_gain = xgb_model.get_feature_importances(feature_names=feature_names, importance_type="gain")
    feat_importances_weight = xgb_model.get_feature_importances(feature_names=feature_names, importance_type="weight")

    feat_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance_Gain": [feat_importances_gain.get(f, 0.0) for f in feature_names],
        "Importance_Weight": [feat_importances_weight.get(f, 0.0) for f in feature_names],
    }).sort_values(by="Importance_Gain", ascending=False).reset_index(drop=True)

    feat_imp_csv_path = metric_path / "xgboost_feature_importance.csv"
    feat_imp_df.to_csv(feat_imp_csv_path, index=False)
    print(f"      [SAVED] Feature importances -> {feat_imp_csv_path}")

    feat_imp_png_path = graph_path / "xgboost_feature_importance.png"
    plot_feature_importance(feat_imp_df, feat_imp_png_path, top_n=25)

    # Save Model Artifact & Metadata
    model_file_path = model_path / "xgboost.pkl"
    xgb_model.save(model_file_path)
    model_size_mb = round(os.path.getsize(model_file_path) / (1024 * 1024), 2)

    meta_payload = {
        "model_type": "XGBoost",
        "xgboost_version": xgb.__version__,
        "timestamp": datetime.now().isoformat(),
        "random_seed": random_state,
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "n_jobs": n_jobs,
        },
        "number_of_features": int(X_train_full.shape[1]),
        "feature_names": feature_names,
        "number_of_classes": len(target_names),
        "label_mapping": label_mapping,
        "training_sample_count": len(X_train),
        "total_training_available": len(X_train_full),
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
    meta_file_path = model_path / "xgboost_metadata.json"
    with open(meta_file_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=4)
    print(f"      [SAVED] Model artifact -> {model_file_path} ({model_size_mb} MB)")
    print(f"      [SAVED] Model metadata -> {meta_file_path}")

    # Update Model Comparison Table (Preserving Random Forest, adding XGBoost)
    comp_file_path = metric_path / "model_comparison.csv"
    comp_row = {
        "Model": "XGBoost",
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

    if comp_file_path.exists():
        comp_df = pd.read_csv(comp_file_path)
        comp_df = comp_df[comp_df["Model"] != "XGBoost"]
        comp_df = pd.concat([comp_df, pd.DataFrame([comp_row])], ignore_index=True)
    else:
        comp_df = pd.DataFrame([comp_row])

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
    comp_df = comp_df[[c for c in required_cols if c in comp_df.columns]]
    comp_df.to_csv(comp_file_path, index=False)
    print(f"      [SAVED] Updated Model Comparison Table -> {comp_file_path}")
    print(f"\n{comp_df.to_string(index=False)}")

    print("\n" + "=" * 80)
    print("STATUS: Phase 7 XGBoost Baseline Training & Evaluation Completed Successfully.")
    print("=" * 80)
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate XGBoost Baseline for ML-IDS")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed dataset directory")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics", help="Directory to save metrics")
    parser.add_argument("--graphs-dir", type=str, default="results/graphs", help="Directory to save graphs")
    parser.add_argument("--confusion-dir", type=str, default="results/confusion_matrices", help="Directory to save confusion matrices")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of boosted trees (default: 100)")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth (default: 6)")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Boosting learning rate (default: 0.1)")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample ratio (default: 0.8)")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, help="Column subsample ratio (default: 0.8)")
    parser.add_argument("--max-train-samples", type=int, default=750000, help="Stratified sample count for training (default: 750000)")
    parser.add_argument("--random-seed", type=int, default=42, help="Fixed random seed (default: 42)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of CPU worker jobs (-1 for all cores)")

    args = parser.parse_args()

    try:
        train_and_evaluate_xgboost(
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            metrics_dir=args.metrics_dir,
            graphs_dir=args.graphs_dir,
            confusion_dir=args.confusion_dir,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            max_train_samples=args.max_train_samples,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
        )
    except Exception as e:
        print(f"\n[FATAL ERROR] XGBoost pipeline execution failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
