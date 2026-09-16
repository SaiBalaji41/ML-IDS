"""
Training and Evaluation Script for 1D-CNN Deep Learning Baseline Model (Phase 8).

Executes reproducible 1D-CNN training on ordered feature vectors with early stopping,
learning rate annealing, history logging, confusion matrix visualization, and artifact persistence.

Usage:
    python scripts/train_cnn_1d.py --data-dir data/processed --epochs 25 --batch-size 512
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
from sklearn.utils.class_weight import compute_class_weight

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras import callbacks

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_1d import Conv1DModel
from src.preprocessing.verify_split import load_partition


def calculate_metrics_and_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str]
) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray]:
    """Calculate exact accuracy, macro/weighted precision, recall, F1, and per-class reports."""
    num_classes = len(target_names)
    total_samples = len(y_true)
    accuracy = float(np.mean(y_true == y_pred))

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

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1_scores))

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


def plot_training_history(history_df: pd.DataFrame, save_path: Path):
    """Plot training and validation loss and accuracy curves."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history_df) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Loss Curve
    ax1.plot(epochs, history_df["loss"], "b-o", label="Training Loss", linewidth=2)
    ax1.plot(epochs, history_df["val_loss"], "r--s", label="Validation Loss", linewidth=2)
    ax1.set_title("1D-CNN Loss Curve", fontsize=14, pad=10)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss (Sparse Categorical Crossentropy)", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy Curve
    ax2.plot(epochs, history_df["accuracy"], "b-o", label="Training Accuracy", linewidth=2)
    ax2.plot(epochs, history_df["val_accuracy"], "g--^", label="Validation Accuracy", linewidth=2)
    ax2.set_title("1D-CNN Accuracy Curve", fontsize=14, pad=10)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"      [SAVED] Training history plot -> {save_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    """Plot normalized test confusion matrix."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(18, 15))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    sns.heatmap(
        cm_norm,
        annot=False,
        cmap="Purples",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Prediction Ratio'}
    )
    plt.title("1D-CNN Deep Learning Baseline — Test Confusion Matrix (Normalized)", fontsize=16, pad=15)
    plt.xlabel("Predicted Label", fontsize=13, labelpad=10)
    plt.ylabel("Ground Truth Label", fontsize=13, labelpad=10)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"      [SAVED] Confusion matrix plot -> {save_path}")


def train_and_evaluate_cnn_1d(
    data_dir: str = "data/processed",
    models_dir: str = "models/cnn_1d",
    metrics_dir: str = "results/metrics",
    graphs_dir: str = "results/graphs",
    confusion_dir: str = "results/confusion_matrices",
    epochs: int = 25,
    batch_size: int = 512,
    learning_rate: float = 0.001,
    max_train_samples: int = 500000,
    random_state: int = 42,
) -> Dict:
    """
    Execute 1D-CNN baseline model training, evaluation, and logging.
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
    print("      ML-IDS: 1D-CNN DEEP LEARNING BASELINE TRAINING & EVALUATION (PHASE 8)")
    print("=" * 80)

    # 1. Load Data Partitions
    print("\n[STEP 1/8] Loading processed datasets...")
    X_train_full, y_train_full = load_partition(base_path / "train", "train")
    X_val, y_val = load_partition(base_path / "validation", "val")
    X_test, y_test = load_partition(base_path / "test", "test")

    mapping_file = base_path / "label_mapping.json"
    with open(mapping_file, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)

    inv_mapping = {v: k for k, v in label_mapping.items()}
    target_names = [inv_mapping[i] for i in range(len(label_mapping))]

    print(f"      Data Representation: Tabular statistical feature vectors (ordered sequence)")
    print(f"      Input Reshape:       (samples, {X_train_full.shape[1]}, 1)")
    print(f"      Classes:             {len(target_names)} distinct classes")
    print(f"      Total Train Samples: {len(X_train_full):,}")
    print(f"      Validation Samples:  {len(X_val):,}")
    print(f"      Test Samples:        {len(X_test):,}")

    # 2. Stratified Subsampling & Class Weight Calculation (Training Partition Only)
    print(f"\n[STEP 2/8] Preparing stratified training subset and computing balanced class weights...")
    if max_train_samples and max_train_samples < len(X_train_full):
        rng = np.random.default_rng(random_state)
        sample_indices = []
        for c in range(len(target_names)):
            c_idx = np.where(y_train_full == c)[0]
            if len(c_idx) == 0: continue
            n_sub = max(min(len(c_idx), 100), int(len(c_idx) * (max_train_samples / len(y_train_full))))
            sample_indices.extend(rng.choice(c_idx, size=min(n_sub, len(c_idx)), replace=False))
        sample_indices = np.array(sample_indices)
        rng.shuffle(sample_indices)
        X_train = X_train_full[sample_indices]
        y_train = y_train_full[sample_indices]
        print(f"      Stratified training sample: {len(X_train):,} samples.")
    else:
        X_train, y_train = X_train_full, y_train_full

    raw_weights = compute_class_weight("balanced", classes=np.arange(len(target_names)), y=y_train)
    # Clip extreme weights to prevent gradient instability
    clipped_weights = np.clip(raw_weights, 0.2, 50.0)
    class_weight_dict = {i: float(w) for i, w in enumerate(clipped_weights)}
    print(f"      Class weights computed strictly on training data (Min: {min(class_weight_dict.values()):.4f}, Max: {max(class_weight_dict.values()):.4f}).")

    # 3. Reshape for Conv1D: (N, 46, 1)
    X_train_3d = np.expand_dims(X_train, axis=-1).astype(np.float32)
    X_val_3d = np.expand_dims(X_val, axis=-1).astype(np.float32)
    X_test_3d = np.expand_dims(X_test, axis=-1).astype(np.float32)

    # 4. Build 1D-CNN Architecture
    print(f"\n[STEP 3/8] Constructing 1D-CNN architecture...")
    cnn_wrapper = Conv1DModel(
        num_features=X_train.shape[1],
        num_classes=len(target_names),
        filters=(64, 128),
        kernel_size=3,
        dense_units=128,
        dropout_rate=0.30,
        learning_rate=learning_rate,
    )
    keras_model = cnn_wrapper.build_model()
    keras_model.summary(print_fn=lambda x: print(f"      {x}"))
    total_params = keras_model.count_params()

    # 5. Callbacks
    best_model_path = model_path / "best_model.keras"
    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5, verbose=1),
        callbacks.ModelCheckpoint(filepath=str(best_model_path), monitor="val_loss", save_best_only=True, verbose=0),
    ]

    # 6. Train Model
    print(f"\n[STEP 4/8] Training 1D-CNN (max_epochs={epochs}, batch_size={batch_size})...")
    t0_train = time.time()
    history = keras_model.fit(
        X_train_3d,
        y_train,
        validation_data=(X_val_3d, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        class_weight=class_weight_dict,
        verbose=1,
    )
    train_duration = round(time.time() - t0_train, 2)
    epochs_completed = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    print(f"      Training completed in {train_duration:.2f} seconds ({epochs_completed} epochs, best epoch: {best_epoch}).")

    # Save History
    history_df = pd.DataFrame(history.history)
    history_csv_path = metric_path / "cnn_1d_history.csv"
    history_df.to_csv(history_csv_path, index=False)
    print(f"      [SAVED] Training history -> {history_csv_path}")

    history_png_path = graph_path / "cnn_1d_training_history.png"
    plot_training_history(history_df, history_png_path)

    # 7. Diagnostic Validation Evaluation
    print("\n[STEP 5/8] Evaluating on Validation set...")
    val_sample_size = len(X_val)
    t0_val = time.time()
    val_probs = keras_model.predict(X_val_3d, batch_size=2048, verbose=0)
    y_val_pred = np.argmax(val_probs, axis=1)
    val_inference_time = round(time.time() - t0_val, 4)

    val_metrics, val_report_df, _ = calculate_metrics_and_report(y_val, y_val_pred, target_names)
    val_metrics["model"] = "1D-CNN"
    val_metrics["split"] = "validation"
    val_metrics["inference_time_sec"] = val_inference_time
    val_metrics["samples_evaluated"] = val_sample_size

    val_json_path = metric_path / "cnn_1d_validation.json"
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=4)
    print(f"      [SAVED] Validation metrics -> {val_json_path}")
    print(f"      Validation Accuracy:          {val_metrics['accuracy'] * 100:.2f}%")
    print(f"      Validation Weighted F1-Score: {val_metrics['weighted_f1']:.4f}")
    print(f"      Validation Macro F1-Score:    {val_metrics['macro_f1']:.4f}")
    print(f"      Validation Inference Latency: {val_inference_time:.4f}s ({val_sample_size:,} samples)")

    # 8. Single Final Evaluation on Test Set
    print("\n[STEP 6/8] Performing single final evaluation on Test set...")
    test_sample_size = len(X_test)
    t0_test = time.time()
    test_probs = keras_model.predict(X_test_3d, batch_size=2048, verbose=0)
    y_test_pred = np.argmax(test_probs, axis=1)
    test_inference_time = round(time.time() - t0_test, 4)

    test_metrics, test_report_df, cm_test = calculate_metrics_and_report(y_test, y_test_pred, target_names)
    test_metrics["model"] = "1D-CNN"
    test_metrics["split"] = "test"
    test_metrics["training_time_sec"] = train_duration
    test_metrics["inference_time_sec"] = test_inference_time
    test_metrics["samples_evaluated"] = test_sample_size
    test_metrics["epochs_completed"] = epochs_completed
    test_metrics["best_epoch"] = best_epoch

    test_json_path = metric_path / "cnn_1d_test.json"
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)
    print(f"      [SAVED] Test metrics -> {test_json_path}")

    class_report_path = metric_path / "cnn_1d_classification_report.csv"
    test_report_df.to_csv(class_report_path, index=False)
    print(f"      [SAVED] Per-class report -> {class_report_path}")

    print(f"      Test Accuracy:          {test_metrics['accuracy'] * 100:.2f}%")
    print(f"      Test Weighted Precision:{test_metrics['weighted_precision']:.4f}")
    print(f"      Test Weighted Recall:   {test_metrics['weighted_recall']:.4f}")
    print(f"      Test Weighted F1-Score: {test_metrics['weighted_f1']:.4f}")
    print(f"      Test Macro Precision:   {test_metrics['macro_precision']:.4f}")
    print(f"      Test Macro Recall:      {test_metrics['macro_recall']:.4f}")
    print(f"      Test Macro F1-Score:    {test_metrics['macro_f1']:.4f}")
    print(f"      Test Inference Latency: {test_inference_time:.4f}s ({test_sample_size:,} samples)")

    # 9. Confusion Matrix & Misclassification Analysis
    print("\n[STEP 7/8] Generating Confusion Matrix & Misclassifications...")
    cm_png_path = conf_path / "cnn_1d_confusion_matrix.png"
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
    misclass_csv_path = metric_path / "cnn_1d_misclassifications.csv"
    misclass_df.to_csv(misclass_csv_path, index=False)
    print(f"      [SAVED] Misclassifications ({len(misclassified_indices):,} total errors, {len(misclass_df):,} saved) -> {misclass_csv_path}")

    # 10. Metadata & Comparison Table Update
    print("\n[STEP 8/8] Serializing Metadata and updating Model Comparison Table...")
    model_size_mb = round(os.path.getsize(best_model_path) / (1024 * 1024), 2)

    meta_payload = {
        "model_type": "1D-CNN",
        "framework": "TensorFlow / Keras",
        "tensorflow_version": tf.__version__,
        "timestamp": datetime.now().isoformat(),
        "random_seed": random_state,
        "input_shape": [46, 1],
        "number_of_features": int(X_train_full.shape[1]),
        "number_of_classes": len(target_names),
        "label_mapping": label_mapping,
        "architecture": {
            "conv1d_1": {"filters": 64, "kernel_size": 3, "padding": "same", "activation": "relu"},
            "batch_norm_1": True,
            "maxpool_1": 2,
            "dropout_1": 0.30,
            "conv1d_2": {"filters": 128, "kernel_size": 3, "padding": "same", "activation": "relu"},
            "batch_norm_2": True,
            "maxpool_2": 2,
            "dropout_2": 0.30,
            "pooling": "GlobalAveragePooling1D",
            "dense_1": {"units": 128, "activation": "relu"},
            "dropout_dense": 0.30,
            "output_layer": {"units": 34, "activation": "softmax"}
        },
        "total_parameters": total_params,
        "optimizer": "Adam",
        "initial_learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": epochs,
        "actual_epochs": epochs_completed,
        "best_epoch": best_epoch,
        "class_weights_used": True,
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
    meta_file_path = model_path / "metadata.json"
    with open(meta_file_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=4)
    print(f"      [SAVED] Model checkpoint -> {best_model_path} ({model_size_mb} MB)")
    print(f"      [SAVED] Metadata -> {meta_file_path}")

    # Update Model Comparison Table (Preserving Random Forest and XGBoost, appending 1D-CNN)
    comp_file_path = metric_path / "model_comparison.csv"
    comp_row = {
        "Model": "1D-CNN",
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
        comp_df = comp_df[comp_df["Model"] != "1D-CNN"]
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
    print("STATUS: Phase 8 1D-CNN Deep Learning Model Training & Evaluation Completed Successfully.")
    print("=" * 80)
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate 1D-CNN for ML-IDS")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed dataset directory")
    parser.add_argument("--models-dir", type=str, default="models/cnn_1d", help="Directory to save models")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics", help="Directory to save metrics")
    parser.add_argument("--graphs-dir", type=str, default="results/graphs", help="Directory to save graphs")
    parser.add_argument("--confusion-dir", type=str, default="results/confusion_matrices", help="Directory to save confusion matrices")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size (default: 512)")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Initial learning rate (default: 0.001)")
    parser.add_argument("--max-train-samples", type=int, default=500000, help="Stratified sample count for training (default: 500000)")
    parser.add_argument("--random-seed", type=int, default=42, help="Fixed random seed (default: 42)")

    args = parser.parse_args()

    try:
        train_and_evaluate_cnn_1d(
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            metrics_dir=args.metrics_dir,
            graphs_dir=args.graphs_dir,
            confusion_dir=args.confusion_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_train_samples=args.max_train_samples,
            random_state=args.random_seed,
        )
    except Exception as e:
        print(f"\n[FATAL ERROR] 1D-CNN pipeline execution failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
