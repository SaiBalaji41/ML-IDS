"""
Consolidated Model Evaluation and Comparison Generator (Phase 11).

Performs rigorous empirical comparative analysis across all 5 models:
1. Random Forest Baseline
2. XGBoost Baseline
3. 1D-CNN Deep Learning Model
4. BiLSTM Deep Learning Model
5. Proposed Hybrid CNN + BiLSTM Model

Generates comparative metric tables, class-wise reports, confusion matrix breakdowns,
error analyses, and publication-quality comparative plots.
"""

import json
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    graphs_dir = PROJECT_ROOT / "results" / "graphs" / "model_comparison"
    curves_dir = PROJECT_ROOT / "results" / "graphs" / "training_curves"
    conf_dir = PROJECT_ROOT / "results" / "confusion_matrices"
    docs_dir = PROJECT_ROOT / "docs"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 11: COMPREHENSIVE MODEL EVALUATION & COMPARISON")
    print("=" * 70)

    # 1. Load Model Comparison
    comp_file = metrics_dir / "model_comparison.csv"
    if not comp_file.exists():
        print(f"Error: {comp_file} not found.")
        return

    comp_df = pd.read_csv(comp_file)
    print("\nCurrent Model Benchmarks:")
    print(comp_df.to_string(index=False))

    # 2. Build Final Model Comparison Table
    # Model, Accuracy, Precision Macro, Recall Macro, F1 Macro, Precision Weighted, Recall Weighted, F1 Weighted, Training Time, Inference Time, Parameter Count, Model Size
    rf_bin = PROJECT_ROOT / "models" / "random_forest.pkl" if (PROJECT_ROOT / "models" / "random_forest.pkl").exists() else PROJECT_ROOT / "models" / "random_forest.joblib"
    model_metadata_map = {
        "Random Forest": (PROJECT_ROOT / "models" / "random_forest_metadata.json", rf_bin),
        "XGBoost": (PROJECT_ROOT / "models" / "xgboost_metadata.json", PROJECT_ROOT / "models" / "xgboost.pkl"),
        "1D-CNN": (PROJECT_ROOT / "models" / "cnn_1d" / "metadata.json", PROJECT_ROOT / "models" / "cnn_1d" / "best_model.keras"),
        "BiLSTM": (PROJECT_ROOT / "models" / "bilstm" / "metadata.json", PROJECT_ROOT / "models" / "bilstm" / "best_model.keras"),
        "CNN + BiLSTM": (PROJECT_ROOT / "models" / "cnn_bilstm" / "metadata.json", PROJECT_ROOT / "models" / "cnn_bilstm" / "best_model.keras"),
        "Hybrid CNN-BiLSTM": (PROJECT_ROOT / "models" / "cnn_bilstm" / "metadata.json", PROJECT_ROOT / "models" / "cnn_bilstm" / "best_model.keras"),
        "CNN-BiLSTM": (PROJECT_ROOT / "models" / "cnn_bilstm" / "metadata.json", PROJECT_ROOT / "models" / "cnn_bilstm" / "best_model.keras"),
    }

    final_rows = []
    for _, row in comp_df.iterrows():
        m_name = row["Model"]
        meta_path, bin_path = model_metadata_map.get(m_name, (None, None))
        p_count = "N/A"
        m_size = "N/A"

        if meta_path and meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                if "parameter_count" in meta:
                    p_count = f"{meta['parameter_count']:,}"
                elif "total_parameters" in meta:
                    p_count = f"{meta['total_parameters']:,}"
                elif "hyperparameters" in meta:
                    p_count = f"Trees={meta['hyperparameters'].get('n_estimators', 100)}"

        if bin_path and bin_path.exists():
            sz_mb = os.path.getsize(bin_path) / (1024 * 1024)
            m_size = f"{sz_mb:.2f} MB"

        final_rows.append({
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

    final_df = pd.DataFrame(final_rows)
    final_csv = metrics_dir / "final_model_comparison.csv"
    final_df.to_csv(final_csv, index=False)
    print(f"\n[SAVED] Final comparison table -> {final_csv}")

    # 3. Generate Research Paper Table
    # Model, Accuracy (%), Precision (%), Recall (%), F1-score (%), Training Time (s), Inference Time (ms/sample)
    test_samples = 1176851
    research_rows = []
    for _, row in comp_df.iterrows():
        inf_str = str(row["Inference Time"]).replace("s", "").strip()
        inf_sec = float(inf_str) if inf_str else 0.0
        inf_ms_sample = (inf_sec / test_samples) * 1000

        tr_str = str(row["Training Time"]).replace("s", "").strip()
        tr_sec = float(tr_str) if tr_str else 0.0

        research_rows.append({
            "Model": row["Model"],
            "Accuracy (%)": round(float(row["Accuracy"]) * 100, 2),
            "Weighted Precision (%)": round(float(row["Precision Weighted"]) * 100, 2),
            "Weighted Recall (%)": round(float(row["Recall Weighted"]) * 100, 2),
            "Weighted F1 (%)": round(float(row["F1 Weighted"]) * 100, 2),
            "Macro F1 (%)": round(float(row["F1 Macro"]) * 100, 2),
            "Training Time (s)": round(tr_sec, 2),
            "Inference Time (ms/sample)": round(inf_ms_sample, 5),
        })

    research_df = pd.DataFrame(research_rows)
    research_csv = metrics_dir / "research_model_comparison.csv"
    research_df.to_csv(research_csv, index=False)
    print(f"[SAVED] Research paper table -> {research_csv}")

    # 4. Generate Performance Comparison Graphs
    palette = ["#2b5c8f", "#3ca35d", "#e27c3e", "#8c564b", "#9467bd"]

    # Graph 1: Accuracy Comparison
    plt.figure(figsize=(9, 5))
    acc_vals = [float(r["Accuracy"]) * 100 for _, r in comp_df.iterrows()]
    bars = plt.bar(comp_df["Model"], acc_vals, color=palette[:len(comp_df)], width=0.55, edgecolor="black")
    plt.title("Model Accuracy Comparison (Test Set: 1,176,851 Samples)", fontsize=13, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
    plt.tight_layout()
    plt.savefig(graphs_dir / "accuracy_comparison.png", dpi=300)
    plt.close()

    # Graph 2: Macro Precision Comparison
    plt.figure(figsize=(9, 5))
    p_vals = [float(r["Precision Macro"]) for _, r in comp_df.iterrows()]
    bars = plt.bar(comp_df["Model"], p_vals, color=palette[:len(comp_df)], width=0.55, edgecolor="black")
    plt.title("Macro Precision Comparison Across 34 Classes", fontsize=13, fontweight="bold")
    plt.ylabel("Macro Precision", fontsize=11)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    plt.tight_layout()
    plt.savefig(graphs_dir / "precision_macro_comparison.png", dpi=300)
    plt.close()

    # Graph 3: Macro Recall Comparison
    plt.figure(figsize=(9, 5))
    r_vals = [float(r["Recall Macro"]) for _, r in comp_df.iterrows()]
    bars = plt.bar(comp_df["Model"], r_vals, color=palette[:len(comp_df)], width=0.55, edgecolor="black")
    plt.title("Macro Recall Comparison Across 34 Classes", fontsize=13, fontweight="bold")
    plt.ylabel("Macro Recall", fontsize=11)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    plt.tight_layout()
    plt.savefig(graphs_dir / "recall_macro_comparison.png", dpi=300)
    plt.close()

    # Graph 4: Macro F1 Comparison
    plt.figure(figsize=(9, 5))
    f1_vals = [float(r["F1 Macro"]) for _, r in comp_df.iterrows()]
    bars = plt.bar(comp_df["Model"], f1_vals, color=palette[:len(comp_df)], width=0.55, edgecolor="black")
    plt.title("Macro F1-Score Comparison Across 34 Classes", fontsize=13, fontweight="bold")
    plt.ylabel("Macro F1-Score", fontsize=11)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    plt.tight_layout()
    plt.savefig(graphs_dir / "f1_macro_comparison.png", dpi=300)
    plt.close()

    # Graph 5: Training Time Comparison
    plt.figure(figsize=(9, 5))
    tr_times = [float(str(r["Training Time"]).replace("s", "").strip()) for _, r in comp_df.iterrows()]
    bars = plt.bar(comp_df["Model"], tr_times, color=palette[:len(comp_df)], width=0.55, edgecolor="black")
    plt.title("Training Time Comparison (Wall-Clock Seconds)", fontsize=13, fontweight="bold")
    plt.ylabel("Training Time (Seconds)", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + max(tr_times)*0.015, f"{yval:.1f}s", ha="center", va="bottom", fontweight="bold", fontsize=10)
    plt.tight_layout()
    plt.savefig(graphs_dir / "training_time_comparison.png", dpi=300)
    plt.close()

    # Graph 6: Inference Time Comparison
    plt.figure(figsize=(9, 5))
    inf_times = [float(str(r["Inference Time"]).replace("s", "").strip()) for _, r in comp_df.iterrows()]
    bars = plt.bar(comp_df["Model"], inf_times, color=palette[:len(comp_df)], width=0.55, edgecolor="black")
    plt.title("Test Inference Latency (1,176,851 Test Flows)", fontsize=13, fontweight="bold")
    plt.ylabel("Inference Time (Seconds)", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + max(inf_times)*0.015, f"{yval:.2f}s", ha="center", va="bottom", fontweight="bold", fontsize=10)
    plt.tight_layout()
    plt.savefig(graphs_dir / "inference_time_comparison.png", dpi=300)
    plt.close()
    print(f"[SAVED] Performance comparison graphs -> {graphs_dir}")

    # 5. Class-wise Comparison Across Available Reports
    rep_files = {
        "Random Forest": metrics_dir / "random_forest_classification_report.csv",
        "XGBoost": metrics_dir / "xgboost_classification_report.csv",
        "1D-CNN": metrics_dir / "cnn_1d_classification_report.csv",
        "BiLSTM": metrics_dir / "bilstm_classification_report.csv",
        "CNN + BiLSTM": metrics_dir / "cnn_bilstm_classification_report.csv",
    }

    classwise_dfs = []
    for m_name, path in rep_files.items():
        if path.exists():
            df = pd.read_csv(path)
            # Normalize column names if needed
            if "Unnamed: 0" in df.columns:
                df = df.rename(columns={"Unnamed: 0": "Class_Name"})
            if "Class_Name" not in df.columns and "Class" in df.columns:
                df = df.rename(columns={"Class": "Class_Name"})

            # Filter only class rows (exclude macro/weighted avg)
            df = df[~df["Class_Name"].isin(["accuracy", "macro avg", "weighted avg"])]

            for _, row in df.iterrows():
                classwise_dfs.append({
                    "Model": m_name,
                    "Class": row["Class_Name"],
                    "Precision": round(float(row.get("precision", row.get("Precision", 0))), 4),
                    "Recall": round(float(row.get("recall", row.get("Recall", 0))), 4),
                    "F1-Score": round(float(row.get("f1-score", row.get("F1_Score", 0))), 4),
                    "Support": int(row.get("support", row.get("Support", 0))),
                })

    if classwise_dfs:
        cw_df = pd.DataFrame(classwise_dfs)
        cw_csv = metrics_dir / "classwise_model_comparison.csv"
        cw_df.to_csv(cw_csv, index=False)
        print(f"[SAVED] Class-wise model comparison -> {cw_csv}")

    # 6. Deep Learning Training Curves Multiplot
    hist_files = {
        "1D-CNN": metrics_dir / "cnn_1d_history.csv",
        "BiLSTM": metrics_dir / "bilstm_history.csv",
        "CNN + BiLSTM": metrics_dir / "cnn_bilstm_history.csv",
    }

    dl_hists = {k: pd.read_csv(p) for k, p in hist_files.items() if p.exists()}
    if dl_hists:
        # Loss Plot
        plt.figure(figsize=(10, 6))
        for m_name, h_df in dl_hists.items():
            epochs = h_df["epoch"] if "epoch" in h_df.columns else range(1, len(h_df) + 1)
            if "loss" in h_df.columns:
                plt.plot(epochs, h_df["loss"], label=f"{m_name} (Train)", linestyle="--", alpha=0.8)
            if "val_loss" in h_df.columns:
                plt.plot(epochs, h_df["val_loss"], label=f"{m_name} (Val)", lw=2)
        plt.title("Deep Learning Convergence: Training vs. Validation Loss", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Sparse Categorical Crossentropy Loss", fontsize=11)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(curves_dir / "dl_loss_curves.png", dpi=300)
        plt.close()

        # Accuracy Plot
        plt.figure(figsize=(10, 6))
        for m_name, h_df in dl_hists.items():
            epochs = h_df["epoch"] if "epoch" in h_df.columns else range(1, len(h_df) + 1)
            if "accuracy" in h_df.columns:
                plt.plot(epochs, h_df["accuracy"], label=f"{m_name} (Train)", linestyle="--", alpha=0.8)
            if "val_accuracy" in h_df.columns:
                plt.plot(epochs, h_df["val_accuracy"], label=f"{m_name} (Val)", lw=2)
        plt.title("Deep Learning Convergence: Training vs. Validation Accuracy", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Accuracy", fontsize=11)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(curves_dir / "dl_accuracy_curves.png", dpi=300)
        plt.close()
        print(f"[SAVED] Training curves multiplot -> {curves_dir}")

    print("\nPhase 11 evaluation and comparison generation completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
