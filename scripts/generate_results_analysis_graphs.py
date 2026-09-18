"""
Generate Results Analysis comparison graphs using ONLY existing metrics CSVs.
Runs instantly (<1 second) with zero model retraining or raw dataset access.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "graphs" / "results_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = PROJECT_ROOT / "results" / "metrics" / "final_results_table.csv"
if not CSV_PATH.exists():
    CSV_PATH = PROJECT_ROOT / "results" / "metrics" / "final_model_comparison.csv"


def clean_num(val):
    if pd.isna(val):
        return 0.0
    val_str = str(val).split()[0].replace(",", "")
    try:
        return float(val_str)
    except ValueError:
        return 0.0


def main():
    print(f"Loading existing metrics from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print("Columns found:", df.columns.tolist())

    models = df["Model"].tolist()
    palette = ["#2b5c8f", "#107c41", "#d83b01", "#5c2d91", "#008272"][: len(models)]

    # 1. Accuracy Comparison
    plt.figure(figsize=(9, 5), dpi=300)
    accuracies = [clean_num(v) * 100 for v in df["Accuracy"]]
    bars = plt.bar(models, accuracies, color=palette, width=0.55, edgecolor="black", linewidth=0.8)
    plt.title("Multi-Class Classification Accuracy Comparison (CICIoT2023 Test Set)", fontsize=13, pad=12, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    plt.xlabel("Model Architecture", fontsize=11, fontweight="bold")
    plt.ylim(0, 115)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    p1 = OUTPUT_DIR / "accuracy_comparison.png"
    plt.savefig(p1)
    plt.close()
    print(f"Saved: {p1}")

    # 2. Macro F1 Comparison
    plt.figure(figsize=(9, 5), dpi=300)
    macro_f1 = [clean_num(v) for v in df["F1 Macro"]]
    bars = plt.bar(models, macro_f1, color=palette, width=0.55, edgecolor="black", linewidth=0.8)
    plt.title("Macro F1-Score Comparison Across 34 Classes", fontsize=13, pad=12, fontweight="bold")
    plt.ylabel("Macro F1-Score", fontsize=11, fontweight="bold")
    plt.xlabel("Model Architecture", fontsize=11, fontweight="bold")
    plt.ylim(0, 1.1)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, macro_f1):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    p2 = OUTPUT_DIR / "macro_f1_comparison.png"
    plt.savefig(p2)
    plt.close()
    print(f"Saved: {p2}")

    # 3. Training Time Comparison
    plt.figure(figsize=(9, 5), dpi=300)
    train_times = [clean_num(v) for v in df["Training Time"]]
    bars = plt.bar(models, train_times, color=palette, width=0.55, edgecolor="black", linewidth=0.8)
    plt.title("Total Training Time Comparison (Log Scale)", fontsize=13, pad=12, fontweight="bold")
    plt.ylabel("Training Time in Seconds (Log Scale)", fontsize=11, fontweight="bold")
    plt.xlabel("Model Architecture", fontsize=11, fontweight="bold")
    plt.yscale("log")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, train_times):
        label_text = f"{val:.1f}s" if val < 3600 else f"{val/3600:.2f}h ({val:.0f}s)"
        plt.text(bar.get_x() + bar.get_width() / 2, val * 1.25, label_text, ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    p3 = OUTPUT_DIR / "training_time_comparison.png"
    plt.savefig(p3)
    plt.close()
    print(f"Saved: {p3}")

    # 4. Inference Time Comparison
    plt.figure(figsize=(9, 5), dpi=300)
    # Extract latency in ms/flow or total seconds
    inf_times = []
    for v in df["Inference Time"]:
        v_str = str(v)
        if "ms/flow" in v_str:
            # extract ms/flow inside parenthesis
            try:
                ms_val = float(v_str.split("(")[1].split("ms")[0].strip())
                inf_times.append(ms_val)
            except Exception:
                inf_times.append(clean_num(v))
        else:
            inf_times.append(clean_num(v))

    bars = plt.bar(models, inf_times, color=palette, width=0.55, edgecolor="black", linewidth=0.8)
    plt.title("Per-Flow Inference Latency (ms / Flow)", fontsize=13, pad=12, fontweight="bold")
    plt.ylabel("Latency (Milliseconds / Flow)", fontsize=11, fontweight="bold")
    plt.xlabel("Model Architecture", fontsize=11, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, inf_times):
        plt.text(bar.get_x() + bar.get_width() / 2, val + (max(inf_times) * 0.02), f"{val:.4f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    p4 = OUTPUT_DIR / "inference_time_comparison.png"
    plt.savefig(p4)
    plt.close()
    print(f"Saved: {p4}")

    print("All 4 required results analysis graphs successfully created from existing metrics!")


if __name__ == "__main__":
    main()
