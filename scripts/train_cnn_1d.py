"""
Training and Evaluation Script for 1D-CNN Deep Learning Model.

Usage:
    python scripts/train_cnn_1d.py --data-dir data/processed --epochs 30 --batch-size 128
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

from src.models.cnn_1d import Conv1DModel
from src.preprocessing.verify_split import load_partition


def train_and_evaluate_cnn(
    data_dir: str = "data/processed",
    models_dir: str = "models/cnn_1d",
    metrics_dir: str = "results/metrics",
    graphs_dir: str = "results/graphs",
    confusion_dir: str = "results/confusion_matrices",
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    random_state: int = 42,
) -> Dict:
    """
    Train and evaluate the 1D-CNN deep learning model.
    """
    import tensorflow as tf

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

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
    print("      ML-IDS: 1D-CNN DEEP LEARNING MODEL TRAINING & EVALUATION")
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

    inv_mapping = {v: k for k, v in label_mapping.items()} if label_mapping else {}
    target_names = [inv_mapping.get(i, f"Class_{i}") for i in sorted(np.unique(y_train))]
    num_classes = len(target_names)
    num_features = X_train.shape[1]

    print(f"      Train Samples: {len(X_train):,} | Features: {num_features} | Classes: {num_classes}")
    print(f"      Reshaping input for Conv1D: (samples, {num_features}, 1)")

    # 2. Build 1D-CNN Model
    print("[2/7] Constructing 1D-CNN architecture...")
    cnn_wrapper = Conv1DModel(
        num_features=num_features,
        num_classes=num_classes,
        learning_rate=learning_rate,
    )
    keras_model = cnn_wrapper.build_model()
    keras_model.summary()

    # 3. Setup Callbacks
    checkpoint_file = model_path / "best_model.keras"
    cb_list = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_file), monitor="val_loss", save_best_only=True, verbose=1),
    ]

    # 4. Train Model
    print(f"[3/7] Training 1D-CNN (epochs={epochs}, batch_size={batch_size}, lr={learning_rate})...")
    t_train_start = time.time()
    history = cnn_wrapper.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        callbacks_list=cb_list,
    )
    t_train_end = time.time()
    train_duration = round(t_train_end - t_train_start, 4)
    print(f"      Training completed in {train_duration:.2f} seconds.")

    # 5. Save History & Training Curves
    print("[4/7] Saving training history and loss/accuracy curves...")
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(metric_path / "cnn_1d_history.csv", index_label="epoch")

    try:
        import matplotlib.pyplot as plt

        # Loss Curve
        plt.figure(figsize=(8, 5))
        plt.plot(hist_df["loss"], label="Train Loss", color="#2b5c8f", lw=2)
        plt.plot(hist_df["val_loss"], label="Val Loss", color="#e27c3e", lw=2)
        plt.title("1D-CNN — Training vs. Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(graph_path / "cnn_1d_training_loss.png", dpi=300)
        plt.close()

        # Accuracy Curve
        plt.figure(figsize=(8, 5))
        plt.plot(hist_df["accuracy"], label="Train Accuracy", color="#2b5c8f", lw=2)
        plt.plot(hist_df["val_accuracy"], label="Val Accuracy", color="#3ca35d", lw=2)
        plt.title("1D-CNN — Training vs. Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(graph_path / "cnn_1d_training_accuracy.png", dpi=300)
        plt.close()
        print(f"      Training curves saved to {graph_path}")
    except Exception as e:
        print(f"      [WARN] Graph plotting skipped: {e}")

    # 6. Validation Evaluation
    print("[5/7] Evaluating on Validation set...")
    t_val_start = time.time()
    y_val_pred = cnn_wrapper.predict(X_val)
    t_val_end = time.time()
    val_latency = round(t_val_end - t_val_start, 4)

    val_acc = float(accuracy_score(y_val, y_val_pred))
    val_prec_w = float(precision_score(y_val, y_val_pred, average="weighted", zero_division=0))
    val_rec_w = float(recall_score(y_val, y_val_pred, average="weighted", zero_division=0))
    val_f1_w = float(f1_score(y_val, y_val_pred, average="weighted", zero_division=0))
    val_f1_m = float(f1_score(y_val, y_val_pred, average="macro", zero_division=0))

    val_metrics = {
        "model": "1D-CNN",
        "split": "validation",
        "accuracy": val_acc,
        "weighted_precision": val_prec_w,
        "weighted_recall": val_rec_w,
        "weighted_f1": val_f1_w,
        "macro_f1": val_f1_m,
        "inference_time_sec": val_latency,
    }
    with open(metric_path / "cnn_1d_validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=4)

    val_report_dict = classification_report(y_val, y_val_pred, target_names=target_names, output_dict=True, zero_division=0)
    pd.DataFrame(val_report_dict).transpose().to_csv(metric_path / "cnn_1d_validation_report.csv")
    print(f"      Val Accuracy: {val_acc*100:.2f}% | Val Weighted F1: {val_f1_w:.4f} | Val Macro F1: {val_f1_m:.4f}")

    # 7. Test Evaluation
    print("[6/7] Evaluating on Test set (Single final evaluation)...")
    t_test_start = time.time()
    y_test_pred = cnn_wrapper.predict(X_test)
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
        "model": "1D-CNN",
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
        "trainable_parameters": int(keras_model.count_params()),
    }
    with open(metric_path / "cnn_1d_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)

    test_report_dict = classification_report(y_test, y_test_pred, target_names=target_names, output_dict=True, zero_division=0)
    test_report_df = pd.DataFrame(test_report_dict).transpose()
    test_report_df.to_csv(metric_path / "cnn_1d_test_report.csv")
    test_report_df.to_csv(metric_path / "cnn_1d_per_class.csv")
    print(f"      Test Accuracy: {test_acc*100:.2f}% | Test Weighted F1: {test_f1_w:.4f} | Test Macro F1: {test_f1_m:.4f}")

    # Confusion Matrices
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Validation Confusion Matrix
        cm_val = confusion_matrix(y_val, y_val_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_val, annot=True, fmt="d", cmap="Purples", xticklabels=target_names, yticklabels=target_names)
        plt.title("1D-CNN — Validation Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(conf_path / "cnn_1d_validation.png", dpi=300)
        plt.close()

        # Test Confusion Matrix
        cm_test = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_test, annot=True, fmt="d", cmap="Purples", xticklabels=target_names, yticklabels=target_names)
        plt.title("1D-CNN Deep Learning — Test Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(conf_path / "cnn_1d_test.png", dpi=300)
        plt.close()
        print(f"      Confusion matrices saved to {conf_path}")
    except Exception as e:
        print(f"      [WARN] Confusion matrix plotting skipped: {e}")

    # Misclassifications analysis
    misclassified_indices = np.where(y_test != y_test_pred)[0]
    misclass_df = pd.DataFrame({
        "Sample_Index": misclassified_indices,
        "True_Class": [inv_mapping.get(y_test[i], str(y_test[i])) for i in misclassified_indices],
        "Predicted_Class": [inv_mapping.get(y_test_pred[i], str(y_test_pred[i])) for i in misclassified_indices],
    })
    misclass_df.to_csv(metric_path / "cnn_1d_misclassifications.csv", index=False)

    # Runtime records
    runtime_records = {
        "model": "1D-CNN",
        "training_time_seconds": train_duration,
        "val_inference_time_seconds": val_latency,
        "test_inference_time_seconds": test_latency,
        "epochs_trained": len(hist_df),
        "total_parameters": int(keras_model.count_params()),
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
    }
    with open(metric_path / "cnn_1d_runtime.json", "w", encoding="utf-8") as f:
        json.dump(runtime_records, f, indent=4)

    # Update Model Comparison Table (Preserving Random Forest & XGBoost)
    comp_file = metric_path / "model_comparison.csv"
    comp_row = {
        "Model": "1D-CNN",
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
        comp_df = comp_df[comp_df["Model"] != "1D-CNN"]
        comp_df = pd.concat([comp_df, pd.DataFrame([comp_row])], ignore_index=True)
    else:
        comp_df = pd.DataFrame([comp_row])
    comp_df.to_csv(comp_file, index=False)

    # 8. Save Metadata
    print("[7/7] Saving model metadata...")
    metadata = {
        "model_type": "1D-CNN",
        "timestamp": datetime.now().isoformat(),
        "random_state": random_state,
        "epochs_trained": len(hist_df),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "input_shape": [num_features, 1],
        "n_features": num_features,
        "n_classes": num_classes,
        "class_names": target_names,
        "total_parameters": int(keras_model.count_params()),
        "test_accuracy": test_acc,
        "test_weighted_f1": test_f1_w,
        "test_macro_f1": test_f1_m,
    }
    with open(model_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"      Model checkpoint saved to: {checkpoint_file}")
    print(f"      Metadata saved to: {model_path / 'metadata.json'}")
    print("=" * 75)
    print("STATUS: 1D-CNN Training & Evaluation Completed.")
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate 1D-CNN Model")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed dataset directory")
    parser.add_argument("--models-dir", type=str, default="models/cnn_1d", help="Model saving directory")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics", help="Metrics output directory")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("--random-seed", type=int, default=42, help="Random state seed")

    args = parser.parse_args()
    try:
        train_and_evaluate_cnn(
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            metrics_dir=args.metrics_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            random_state=args.random_seed,
        )
    except Exception as e:
        print(f"\n[ERROR] 1D-CNN training halted: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
