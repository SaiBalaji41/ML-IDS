"""
Training and Evaluation Pipeline for Proposed Hybrid CNN + BiLSTM Model (Phase 10).

Executes end-to-end training of the hybrid Convolutional-Recurrent network on processed CICIoT2023 partitions,
tracks convergence, evaluates test generalization, exports all metric artifacts, and generates the extended comparison table.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Dict, Tuple

def _safe_get_machine_win32():
    return os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64")
platform._get_machine_win32 = _safe_get_machine_win32

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_bilstm import CNNBiLSTMModel
from src.preprocessing.verify_split import load_partition


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate Hybrid CNN-BiLSTM for ML-IDS.")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed data directory.")
    parser.add_argument("--models-dir", type=str, default="models/cnn_bilstm", help="Path to save model artifacts.")
    parser.add_argument("--results-dir", type=str, default="results", help="Root results directory.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial Adam learning rate.")
    parser.add_argument("--max-train-samples", type=int, default=400000, help="Subsample train partition if needed for memory/speed.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def plot_training_history(history_df: pd.DataFrame, output_path: Path):
    """Plot and save training and validation loss and accuracy curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    axes[0].plot(history_df["epoch"], history_df["loss"], label="Train Loss", color="#1f77b4", lw=2)
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    axes[0].set_title("Hybrid CNN-BiLSTM — Training vs. Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss (Sparse Categorical Crossentropy)", fontsize=10)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Accuracy Curve
    axes[1].plot(history_df["epoch"], history_df["accuracy"], label="Train Accuracy", color="#1f77b4", lw=2)
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="Val Accuracy", color="#2ca02c", lw=2)
    axes[1].set_title("Hybrid CNN-BiLSTM — Training vs. Validation Accuracy", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Accuracy", fontsize=10)
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved training history curves to {output_path}")


def evaluate_partition(
    model: CNNBiLSTMModel,
    X: np.ndarray,
    y: np.ndarray,
    partition_name: str,
    batch_size: int = 4096,
) -> Tuple[Dict[str, float], np.ndarray, float]:
    """Calculate standard classification metrics on given partition."""
    print(f"\nRunning model inference on {partition_name} partition ({len(X):,} samples, batch_size={batch_size})...")
    sys.stdout.flush()
    t0 = time.time()
    y_pred = model.predict(X, batch_size=batch_size)
    infer_time = time.time() - t0

    acc = accuracy_score(y, y_pred)
    p_macro = precision_score(y, y_pred, average="macro", zero_division=0)
    r_macro = recall_score(y, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y, y_pred, average="macro", zero_division=0)
    p_weighted = precision_score(y, y_pred, average="weighted", zero_division=0)
    r_weighted = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y, y_pred, average="weighted", zero_division=0)

    metrics = {
        "partition": partition_name,
        "sample_count": int(len(y)),
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
        "inference_time_seconds": float(infer_time),
    }

    print(f"\n[{partition_name.upper()}] Evaluation Metrics:")
    print(f"  Accuracy:          {acc*100:.2f}%")
    print(f"  Macro F1-Score:    {f1_macro:.4f}")
    print(f"  Weighted F1-Score: {f1_weighted:.4f}")
    print(f"  Inference Time:    {infer_time:.2f}s ({infer_time/len(y)*1e6:.2f} µs/sample)")
    return metrics, y_pred, infer_time


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = results_dir / "metrics"
    graphs_dir = results_dir / "graphs"
    conf_dir = results_dir / "confusion_matrices"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    conf_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 10: PROPOSED HYBRID CNN + BiLSTM MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load label mapping
    mapping_file = data_dir / "label_mapping.json"
    if not mapping_file.exists():
        raise FileNotFoundError(f"Label mapping not found at {mapping_file}")
    with open(mapping_file, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    inv_mapping = {v: k for k, v in label_mapping.items()}
    num_classes = len(label_mapping)
    target_names = [inv_mapping.get(i, f"Class_{i}") for i in range(num_classes)]

    # 2. Load dataset partitions
    print("\nLoading dataset partitions...")
    X_train, y_train = load_partition(data_dir / "train", "train")
    X_val, y_val = load_partition(data_dir / "validation", "val")
    X_test, y_test = load_partition(data_dir / "test", "test")

    print(f"Loaded Partitions:")
    print(f"  Train:      X={X_train.shape}, y={y_train.shape}")
    print(f"  Validation: X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:       X={X_test.shape}, y={y_test.shape}")

    # Subsample training data if specified for training efficiency
    if args.max_train_samples and len(X_train) > args.max_train_samples:
        print(f"\nSubsampling training data to {args.max_train_samples:,} samples (stratified)...")
        from sklearn.model_selection import train_test_split
        _, X_train_sub, _, y_train_sub = train_test_split(
            X_train,
            y_train,
            test_size=args.max_train_samples,
            stratify=y_train,
            random_state=args.random_seed,
        )
        X_train, y_train = X_train_sub, y_train_sub
        print(f"Subsampled train set: X={X_train.shape}, y={y_train.shape}")

    num_features = X_train.shape[1]

    # 3. Calculate class weights strictly on training partition
    print("\nCalculating balanced class weights on training split...")
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weights_clipped = np.clip(weights, 0.2, 50.0)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights_clipped)}
    print(f"Computed weights for {len(class_weight_dict)} classes (clipped range: [{weights_clipped.min():.4f}, {weights_clipped.max():.4f}])")

    # 4. Build Hybrid Architecture
    print("\nInitializing Proposed Hybrid CNN-BiLSTM architecture...")
    hybrid_wrapper = CNNBiLSTMModel(
        num_features=num_features,
        num_classes=num_classes,
        conv_filters=64,
        kernel_size=3,
        pool_size=2,
        lstm_units=64,
        dense_units=64,
        dropout_rate=0.30,
        learning_rate=args.lr,
    )
    keras_model = hybrid_wrapper.build_model()
    keras_model.summary()

    total_params = keras_model.count_params()

    # 5. Callbacks Setup
    best_model_path = models_dir / "best_model.keras"
    cb_list = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 6. Training Execution
    print(f"\nStarting Hybrid CNN-BiLSTM training (max {args.epochs} epochs, batch size {args.batch_size})...")
    val_sub_n = min(50000, len(X_val))
    X_val_train, y_val_train = X_val[:val_sub_n], y_val[:val_sub_n]
    t0_train = time.time()
    history = hybrid_wrapper.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val_train,
        y_val=y_val_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks_list=cb_list,
        class_weight=class_weight_dict,
    )
    train_duration = time.time() - t0_train
    print(f"\nHybrid CNN-BiLSTM training completed in {train_duration:.2f} seconds.")

    # 7. Save and Plot Training History
    hist_dict = history.history
    epochs_ran = len(hist_dict["loss"])
    best_epoch = int(np.argmin(hist_dict["val_loss"]) + 1)
    history_df = pd.DataFrame({
        "epoch": list(range(1, epochs_ran + 1)),
        "loss": hist_dict["loss"],
        "accuracy": hist_dict["accuracy"],
        "val_loss": hist_dict["val_loss"],
        "val_accuracy": hist_dict["val_accuracy"],
        "lr": hist_dict.get("learning_rate", [args.lr] * epochs_ran),
    })
    history_csv = metrics_dir / "cnn_bilstm_history.csv"
    history_df.to_csv(history_csv, index=False)
    print(f"Saved training history to {history_csv}")

    history_plot = graphs_dir / "cnn_bilstm_training_history.png"
    plot_training_history(history_df, history_plot)

    # 8. Load Best Checkpoint for Evaluation
    if best_model_path.exists():
        print(f"\nLoading best checkpoint from {best_model_path} (best epoch: {best_epoch})...")
        hybrid_wrapper.load(best_model_path)

    # 9. Validation Evaluation
    val_metrics, y_val_pred, val_infer_time = evaluate_partition(
        hybrid_wrapper, X_val, y_val, "validation", batch_size=4096
    )
    val_json = metrics_dir / "cnn_bilstm_validation.json"
    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)
    print(f"Saved validation metrics to {val_json}")

    # 10. Test Evaluation (Single Unbiased Pass)
    test_metrics, y_test_pred, test_infer_time = evaluate_partition(
        hybrid_wrapper, X_test, y_test, "test", batch_size=4096
    )
    test_json = metrics_dir / "cnn_bilstm_test.json"
    with open(test_json, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Saved test metrics to {test_json}")

    # 11. Per-Class Classification Report
    report_dict = classification_report(
        y_test,
        y_test_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv = metrics_dir / "cnn_bilstm_classification_report.csv"
    report_df.to_csv(report_csv, index=True)
    print(f"Saved per-class report to {report_csv}")

    # 12. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_test_pred)
    cm_plot = conf_dir / "cnn_bilstm_confusion_matrix.png"
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Purples",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title("Hybrid CNN-BiLSTM — Test Set Confusion Matrix (34 Classes)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(cm_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot to {cm_plot}")

    # 13. Misclassifications Analysis
    err_idx = np.where(y_test != y_test_pred)[0]
    print(f"Total misclassified test samples: {len(err_idx)} ({len(err_idx)/len(y_test)*100:.2f}%)")
    misclass_df = pd.DataFrame({
        "sample_index": err_idx[:50000],  # cap for disk efficiency
        "true_label_idx": y_test[err_idx[:50000]],
        "true_class_name": [inv_mapping.get(y_test[i], str(y_test[i])) for i in err_idx[:50000]],
        "pred_label_idx": y_test_pred[err_idx[:50000]],
        "pred_class_name": [inv_mapping.get(y_test_pred[i], str(y_test_pred[i])) for i in err_idx[:50000]],
    })
    misclass_csv = metrics_dir / "cnn_bilstm_misclassifications.csv"
    misclass_df.to_csv(misclass_csv, index=False)
    print(f"Saved misclassification analysis to {misclass_csv}")

    # 14. Save Metadata JSON
    model_size_mb = os.path.getsize(best_model_path) / (1024 * 1024) if best_model_path.exists() else 0.0
    gpus = tf.config.list_physical_devices("GPU")
    metadata = {
        "model_name": "CNN-BiLSTM",
        "model_type": "Hybrid Convolutional Bidirectional LSTM",
        "framework": f"TensorFlow {tf.__version__}",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "gpu_count": len(gpus),
            "gpu_devices": [g.name for g in gpus],
        },
        "dataset_source": "CICIoT2023 Processed Partitions",
        "data_representation": "Ordered 46-dimensional tabular network flow feature vectors reshaped to [samples, 46, 1]. Sequence order corresponds to feature index.",
        "number_of_features": num_features,
        "number_of_classes": num_classes,
        "parameter_count": total_params,
        "model_size_mb": round(model_size_mb, 2),
        "optimizer": "adam",
        "initial_learning_rate": args.lr,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "actual_epochs_completed": epochs_ran,
        "best_epoch": best_epoch,
        "random_seed": args.random_seed,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "training_time_seconds": round(train_duration, 2),
        "test_inference_time_seconds": round(test_infer_time, 2),
        "class_weights_used": True,
        "class_weights": class_weight_dict,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "label_mapping": label_mapping,
    }
    meta_json = models_dir / "metadata.json"
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to {meta_json}")

    # 15. Update model_comparison.csv & model_comparison_extended.csv
    comp_file = metrics_dir / "model_comparison.csv"
    if comp_file.exists():
        comp_df = pd.read_csv(comp_file)
    else:
        comp_df = pd.DataFrame(columns=[
            "Model", "Accuracy", "Precision Macro", "Recall Macro", "F1 Macro",
            "Precision Weighted", "Recall Weighted", "F1 Weighted",
            "Training Time", "Inference Time"
        ])

    hybrid_row = {
        "Model": "CNN + BiLSTM",
        "Accuracy": round(test_metrics["accuracy"], 4),
        "Precision Macro": round(test_metrics["macro_precision"], 4),
        "Recall Macro": round(test_metrics["macro_recall"], 4),
        "F1 Macro": round(test_metrics["macro_f1"], 4),
        "Precision Weighted": round(test_metrics["weighted_precision"], 4),
        "Recall Weighted": round(test_metrics["weighted_recall"], 4),
        "F1 Weighted": round(test_metrics["weighted_f1"], 4),
        "Training Time": f"{round(train_duration, 2)} s",
        "Inference Time": f"{round(test_infer_time, 2)} s",
    }

    comp_df = comp_df[comp_df["Model"] != "CNN + BiLSTM"]
    comp_df = pd.concat([comp_df, pd.DataFrame([hybrid_row])], ignore_index=True)
    comp_df.to_csv(comp_file, index=False)
    print(f"\nUpdated Standard Model Comparison Table at {comp_file}:")
    print(comp_df.to_string(index=False))

    # Extended Comparison
    ext_file = metrics_dir / "model_comparison_extended.csv"
    ext_rows = []
    # Helper to lookup param and model size if metadata exists
    model_meta_map = {
        "Random Forest": ("models/random_forest_metadata.json", "models/random_forest.joblib", "N/A"),
        "XGBoost": ("models/xgboost_metadata.json", "models/xgboost.pkl", "N/A"),
        "1D-CNN": ("models/cnn_1d/metadata.json", "models/cnn_1d/best_model.keras", "params"),
        "BiLSTM": ("models/bilstm/metadata.json", "models/bilstm/best_model.keras", "params"),
        "CNN + BiLSTM": ("models/cnn_bilstm/metadata.json", "models/cnn_bilstm/best_model.keras", "params"),
    }

    for _, row in comp_df.iterrows():
        m_name = row["Model"]
        meta_p, bin_p, p_type = model_meta_map.get(m_name, (None, None, None))
        p_count = "N/A"
        m_size = "N/A"
        if meta_p and Path(meta_p).exists():
            with open(meta_p, "r", encoding="utf-8") as mf:
                m_data = json.load(mf)
                if "parameter_count" in m_data:
                    p_count = f"{m_data['parameter_count']:,}"
                elif "hyperparameters" in m_data:
                    p_count = f"Trees={m_data['hyperparameters'].get('n_estimators', '100')}"
        if bin_p and Path(bin_p).exists():
            sz = os.path.getsize(bin_p) / (1024 * 1024)
            m_size = f"{sz:.2f} MB"

        ext_rows.append({
            "Model": m_name,
            "Accuracy": row["Accuracy"],
            "Precision Macro": row["Precision Macro"],
            "Recall Macro": row["Recall Macro"],
            "F1 Macro": row["F1 Macro"],
            "Precision Weighted": row["Precision Weighted"],
            "Recall Weighted": row["Recall Weighted"],
            "F1 Weighted": row["F1 Weighted"],
            "Training Time": row["Training Time"],
            "Inference Time": row["Inference Time"],
            "Parameter Count": p_count,
            "Model Size": m_size,
        })
    ext_df = pd.DataFrame(ext_rows)
    ext_df.to_csv(ext_file, index=False)
    print(f"\nGenerated Extended Comparison Table at {ext_file}:")
    print(ext_df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
