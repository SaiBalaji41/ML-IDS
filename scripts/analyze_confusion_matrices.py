"""
Phase 12: Comprehensive Confusion Matrix & Error Analysis Script.

Performs rigorous empirical confusion matrix and error diagnostics across all 5 trained models:
1. Random Forest Baseline
2. XGBoost Baseline
3. Standalone 1D-CNN
4. Standalone BiLSTM
5. Proposed Hybrid CNN + BiLSTM

Generates:
- Raw and normalized confusion matrix heatmaps (34x34)
- One-vs-Rest class-wise error metrics (TP, TN, FP, FN, Precision, Recall, F1, Support)
- Empirical misclassification pairs & top confusion rankings
- Benign vs. Attack error breakdown (FP, FN, Attack-to-Attack confusion)
- Cross-model error comparison and research paper tables
- Error visualization graphs
"""

import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def load_data_and_labels():
    print("[1/6] Loading test dataset and label metadata...")
    mapping_p = PROJECT_ROOT / "data" / "processed" / "label_mapping.json"
    with open(mapping_p, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    id_to_label = {int(v): k for k, v in label_mapping.items()}
    num_classes = len(label_mapping)
    target_names = [id_to_label[i] for i in range(num_classes)]

    test_npz = PROJECT_ROOT / "data" / "processed" / "test" / "test.npz"
    data = np.load(test_npz)
    X_test = data["X"]
    y_test = data["y"]
    print(f"      Loaded {len(X_test):,} test samples, {num_classes} classes.")
    return X_test, y_test, label_mapping, id_to_label, target_names


def get_model_predictions(X_test: np.ndarray, y_test: np.ndarray, target_names: List[str]) -> Dict[str, np.ndarray]:
    print("[2/6] Generating/Loading model test predictions on identical test partition...")
    predictions = {}
    num_samples = len(X_test)

    # 1. Random Forest
    rf_p = PROJECT_ROOT / "models" / "random_forest.pkl"
    if rf_p.exists():
        print("      Evaluating Random Forest baseline...")
        rf_obj = joblib.load(rf_p)
        rf_model = rf_obj["model"] if isinstance(rf_obj, dict) and "model" in rf_obj else rf_obj
        # Predict in chunks to optimize memory
        y_pred_rf = np.zeros(num_samples, dtype=np.int32)
        chunk_size = 100000
        for i in range(0, num_samples, chunk_size):
            y_pred_rf[i : i + chunk_size] = rf_model.predict(X_test[i : i + chunk_size])
        predictions["Random Forest"] = y_pred_rf

    # 2. XGBoost
    xgb_p = PROJECT_ROOT / "models" / "xgboost.pkl"
    if xgb_p.exists():
        print("      Evaluating XGBoost baseline...")
        xgb_obj = joblib.load(xgb_p)
        xgb_model = xgb_obj["model"] if isinstance(xgb_obj, dict) and "model" in xgb_obj else xgb_obj
        y_pred_xgb = np.zeros(num_samples, dtype=np.int32)
        chunk_size = 100000
        for i in range(0, num_samples, chunk_size):
            y_pred_xgb[i : i + chunk_size] = xgb_model.predict(X_test[i : i + chunk_size])
        predictions["XGBoost"] = y_pred_xgb

    # 3. 1D-CNN
    cnn_p = PROJECT_ROOT / "models" / "cnn_1d" / "best_model.keras"
    if cnn_p.exists():
        print("      Evaluating 1D-CNN deep learning model...")
        import tensorflow as tf
        cnn_model = tf.keras.models.load_model(cnn_p)
        X_test_dl = np.expand_dims(X_test, axis=-1)
        probs_cnn = cnn_model.predict(X_test_dl, batch_size=8192, verbose=1)
        predictions["1D-CNN"] = np.argmax(probs_cnn, axis=1)

    # 4. Standalone BiLSTM
    bilstm_p = PROJECT_ROOT / "models" / "bilstm" / "best_model.keras"
    if bilstm_p.exists():
        print("      Evaluating Standalone BiLSTM deep learning model...")
        import tensorflow as tf
        bilstm_model = tf.keras.models.load_model(bilstm_p)
        X_test_dl = np.expand_dims(X_test, axis=-1)
        probs_bilstm = bilstm_model.predict(X_test_dl, batch_size=8192, verbose=1)
        predictions["BiLSTM"] = np.argmax(probs_bilstm, axis=1)

    # 5. Proposed Hybrid CNN + BiLSTM
    cnn_bilstm_p = PROJECT_ROOT / "models" / "cnn_bilstm" / "best_model.keras"
    if cnn_bilstm_p.exists():
        print("      Evaluating Proposed Hybrid CNN + BiLSTM model...")
        import tensorflow as tf
        hybrid_model = tf.keras.models.load_model(cnn_bilstm_p)
        X_test_dl = np.expand_dims(X_test, axis=-1)
        probs_hybrid = hybrid_model.predict(X_test_dl, batch_size=8192, verbose=1)
        predictions["CNN + BiLSTM"] = np.argmax(probs_hybrid, axis=1)

    return predictions


def plot_and_save_confusion_matrices(
    cms: Dict[str, np.ndarray],
    target_names: List[str],
    conf_dir: Path,
):
    print("[3/6] Generating publication-quality Raw and Normalized Confusion Matrix heatmaps...")
    conf_dir.mkdir(parents=True, exist_ok=True)
    slugs = {
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "1D-CNN": "cnn_1d",
        "BiLSTM": "bilstm",
        "CNN + BiLSTM": "cnn_bilstm",
    }

    for model_name, cm in cms.items():
        slug = slugs.get(model_name, model_name.lower().replace(" ", "_"))
        
        # Save raw numpy array
        npy_path = conf_dir / f"{slug}_cm.npy"
        np.save(npy_path, cm)

        # 1. Raw Count Heatmap
        plt.figure(figsize=(16, 14))
        sns.heatmap(
            cm,
            annot=False,
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
            cbar_kws={"label": "Sample Count"},
        )
        plt.title(f"{model_name} — Test Confusion Matrix (Raw Sample Counts)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
        plt.ylabel("True Class", fontsize=12, labelpad=10)
        plt.xticks(rotation=90, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        raw_png = conf_dir / f"{slug}_confusion_matrix.png"
        plt.savefig(raw_png, dpi=300)
        plt.close()

        # 2. Normalized Heatmap (Recall / True Class fraction)
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.divide(cm.astype("float"), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)

        plt.figure(figsize=(16, 14))
        sns.heatmap(
            cm_norm,
            annot=False,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            xticklabels=target_names,
            yticklabels=target_names,
            cbar_kws={"label": "Normalized Recall Fraction"},
        )
        plt.title(f"{model_name} — Test Confusion Matrix (Normalized True-Class Recall)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
        plt.ylabel("True Class", fontsize=12, labelpad=10)
        plt.xticks(rotation=90, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        norm_png = conf_dir / f"{slug}_confusion_matrix_normalized.png"
        plt.savefig(norm_png, dpi=300)
        plt.close()
        print(f"      Saved: {raw_png.name} & {norm_png.name}")


def compute_classwise_and_misclassifications(
    cms: Dict[str, np.ndarray],
    target_names: List[str],
    metrics_dir: Path,
):
    print("[4/6] Computing One-vs-Rest class-wise error analysis & misclassification pairs...")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(target_names)
    num_samples = 1176851

    # 1. Class-wise OvR Analysis Table
    cw_rows = []
    for model_name, cm in cms.items():
        total_mat = cm.sum()
        for k in range(num_classes):
            c_name = target_names[k]
            tp = int(cm[k, k])
            fp = int(cm[:, k].sum() - tp)
            fn = int(cm[k, :].sum() - tp)
            tn = int(total_mat - (tp + fp + fn))
            support = int(cm[k, :].sum())

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            cw_rows.append({
                "Model": model_name,
                "Class": c_name,
                "Class_ID": k,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Support": support,
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
            })

    cw_df = pd.DataFrame(cw_rows)
    cw_csv = metrics_dir / "classwise_error_analysis.csv"
    cw_df.to_csv(cw_csv, index=False)
    print(f"      [SAVED] -> {cw_csv}")

    # 2. Misclassification Pairs Table
    misc_rows = []
    for model_name, cm in cms.items():
        total_errors = int(cm.sum() - np.trace(cm))
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j and cm[i, j] > 0:
                    err_cnt = int(cm[i, j])
                    err_pct = round((err_cnt / total_errors) * 100, 2) if total_errors > 0 else 0.0
                    misc_rows.append({
                        "Model": model_name,
                        "Actual Class": target_names[i],
                        "Predicted Class": target_names[j],
                        "Error Count": err_cnt,
                        "Error Percentage (%)": err_pct,
                    })

    misc_df = pd.DataFrame(misc_rows)
    misc_df = misc_df.sort_values(by=["Model", "Error Count"], ascending=[True, False])
    misc_csv = metrics_dir / "misclassification_pairs.csv"
    misc_df.to_csv(misc_csv, index=False)
    print(f"      [SAVED] -> {misc_csv}")

    # 3. Benign vs. Attack Error Analysis Table
    benign_idx = target_names.index("BenignTraffic") if "BenignTraffic" in target_names else 1
    benign_rows = []
    for model_name, cm in cms.items():
        total_samples = int(cm.sum())
        benign_support = int(cm[benign_idx, :].sum())
        attack_support = int(total_samples - benign_support)

        benign_correct = int(cm[benign_idx, benign_idx])
        benign_to_attack_fp = int(benign_support - benign_correct)

        attack_correct = int(np.trace(cm) - benign_correct)
        attack_to_benign_fn = int(cm[:, benign_idx].sum() - benign_correct)
        attack_to_attack_errors = int((cm.sum() - np.trace(cm)) - benign_to_attack_fp - attack_to_benign_fn)

        benign_acc = round((benign_correct / benign_support) * 100, 2) if benign_support > 0 else 0.0
        attack_acc = round((attack_correct / attack_support) * 100, 2) if attack_support > 0 else 0.0
        fp_rate = round((benign_to_attack_fp / benign_support) * 100, 2) if benign_support > 0 else 0.0
        fn_rate = round((attack_to_benign_fn / attack_support) * 100, 2) if attack_support > 0 else 0.0

        benign_rows.append({
            "Model": model_name,
            "Benign Support": benign_support,
            "Benign Correct": benign_correct,
            "Benign Accuracy (%)": benign_acc,
            "Benign->Attack FP": benign_to_attack_fp,
            "Benign FP Rate (%)": fp_rate,
            "Attack Support": attack_support,
            "Attack Correct": attack_correct,
            "Attack Accuracy (%)": attack_acc,
            "Attack->Benign FN": attack_to_benign_fn,
            "Attack FN Rate (%)": fn_rate,
            "Attack->Attack Errors": attack_to_attack_errors,
        })

    benign_df = pd.DataFrame(benign_rows)
    benign_csv = metrics_dir / "benign_vs_attack_analysis.csv"
    benign_df.to_csv(benign_csv, index=False)
    print(f"      [SAVED] -> {benign_csv}")

    # 4. Model Error Comparison Table
    err_comp_rows = []
    research_rows = []
    for model_name, cm in cms.items():
        total_samples = int(cm.sum())
        correct = int(np.trace(cm))
        incorrect = int(total_samples - correct)
        err_rate = round((incorrect / total_samples) * 100, 4)

        # Calculate macro metrics directly from CM
        prec_list = []
        rec_list = []
        f1_list = []
        for k in range(num_classes):
            tp = cm[k, k]
            fp = cm[:, k].sum() - tp
            fn = cm[k, :].sum() - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            prec_list.append(p)
            rec_list.append(r)
            f1_list.append(f)

        macro_p = round(float(np.mean(prec_list)), 4)
        macro_r = round(float(np.mean(rec_list)), 4)
        macro_f1 = round(float(np.mean(f1_list)), 4)

        err_comp_rows.append({
            "Model": model_name,
            "Total Test Samples": total_samples,
            "Correct Predictions": correct,
            "Incorrect Predictions": incorrect,
            "Error Rate (%)": err_rate,
            "Macro Precision": macro_p,
            "Macro Recall": macro_r,
            "Macro F1": macro_f1,
        })

        # Find top confused pair
        top_pair_str = "None"
        top_pair_cnt = 0
        m_misc = misc_df[misc_df["Model"] == model_name]
        if not m_misc.empty:
            top_row = m_misc.iloc[0]
            top_pair_str = f"{top_row['Actual Class']} -> {top_row['Predicted Class']}"
            top_pair_cnt = int(top_row["Error Count"])

        research_rows.append({
            "Model": model_name,
            "Test Samples": total_samples,
            "Correct": correct,
            "Incorrect": incorrect,
            "Error Rate (%)": round(err_rate, 2),
            "Macro Precision (%)": round(macro_p * 100, 2),
            "Macro Recall (%)": round(macro_r * 100, 2),
            "Macro F1 (%)": round(macro_f1 * 100, 2),
            "Most Confused Class Pair": top_pair_str,
            "Confusion Count": top_pair_cnt,
        })

    err_comp_df = pd.DataFrame(err_comp_rows)
    err_comp_csv = metrics_dir / "model_error_comparison.csv"
    err_comp_df.to_csv(err_comp_csv, index=False)
    print(f"      [SAVED] -> {err_comp_csv}")

    res_df = pd.DataFrame(research_rows)
    res_csv = metrics_dir / "research_error_analysis.csv"
    res_df.to_csv(res_csv, index=False)
    print(f"      [SAVED] -> {res_csv}")

    return misc_df, benign_df, err_comp_df


def generate_error_visualizations(
    cms: Dict[str, np.ndarray],
    misc_df: pd.DataFrame,
    benign_df: pd.DataFrame,
    err_comp_df: pd.DataFrame,
    cw_df: pd.DataFrame,
    graphs_dir: Path,
):
    print("[5/6] Generating comprehensive error analysis diagnostic charts...")
    graphs_dir.mkdir(parents=True, exist_ok=True)
    palette = ["#2b5c8f", "#3ca35d", "#e27c3e", "#8c564b", "#9467bd"]

    # 1. Total Error Count by Model
    plt.figure(figsize=(9, 5))
    bars = plt.bar(
        err_comp_df["Model"],
        err_comp_df["Incorrect Predictions"],
        color=palette[:len(err_comp_df)],
        edgecolor="black",
        width=0.55,
    )
    plt.title("Total Misclassifications Across Evaluated Models (1,176,851 Test Flows)", fontsize=12, fontweight="bold")
    plt.ylabel("Number of Misclassified Flows", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + max(err_comp_df["Incorrect Predictions"]) * 0.015, f"{yval:,}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    plt.tight_layout()
    p1 = graphs_dir / "error_count_by_model.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"      [SAVED] -> {p1}")

    # 2. Error Rate (%) by Model
    plt.figure(figsize=(9, 5))
    bars = plt.bar(
        err_comp_df["Model"],
        err_comp_df["Error Rate (%)"],
        color=palette[:len(err_comp_df)],
        edgecolor="black",
        width=0.55,
    )
    plt.title("Empirical Error Rate (%) Across Evaluated Models", fontsize=12, fontweight="bold")
    plt.ylabel("Error Rate (%)", fontsize=11)
    plt.ylim(0, max(err_comp_df["Error Rate (%)"]) * 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
    plt.tight_layout()
    p2 = graphs_dir / "error_rate_by_model.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"      [SAVED] -> {p2}")

    # 3. Top Misclassification Pairs per Model
    slugs = {
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "1D-CNN": "cnn_1d",
        "BiLSTM": "bilstm",
        "CNN + BiLSTM": "cnn_bilstm",
    }
    for model_name in cms.keys():
        slug = slugs.get(model_name, model_name.lower().replace(" ", "_"))
        m_misc = misc_df[misc_df["Model"] == model_name].head(10).iloc[::-1]
        if not m_misc.empty:
            plt.figure(figsize=(11, 6))
            pair_labels = [f"{r['Actual Class']}  \u2192  {r['Predicted Class']}" for _, r in m_misc.iterrows()]
            bars = plt.barh(pair_labels, m_misc["Error Count"], color="#c0392b", edgecolor="#78281f")
            plt.title(f"Top 10 Misclassification Confusion Pairs — {model_name}", fontsize=12, fontweight="bold")
            plt.xlabel("Misclassified Flow Count", fontsize=10)
            plt.grid(axis="x", linestyle="--", alpha=0.5)
            for bar in bars:
                w = bar.get_width()
                plt.text(w + max(m_misc["Error Count"]) * 0.01, bar.get_y() + bar.get_height() / 2, f"{w:,}", va="center", fontsize=8, fontweight="bold")
            plt.tight_layout()
            p_misc = graphs_dir / f"top_misclassifications_{slug}.png"
            plt.savefig(p_misc, dpi=300)
            plt.close()

    # 4. Benign vs. Attack Error Decomposition
    plt.figure(figsize=(10, 6))
    x = np.arange(len(benign_df))
    w = 0.25
    plt.bar(x - w, benign_df["Benign->Attack FP"], width=w, label="Benign \u2192 Attack (False Positives)", color="#e67e22", edgecolor="black")
    plt.bar(x, benign_df["Attack->Benign FN"], width=w, label="Attack \u2192 Benign (False Negatives)", color="#e74c3c", edgecolor="black")
    plt.bar(x + w, benign_df["Attack->Attack Errors"], width=w, label="Attack \u2192 Other Attack (Subclass Confusion)", color="#3498db", edgecolor="black")
    plt.xticks(x, benign_df["Model"], fontweight="bold")
    plt.ylabel("Error Count", fontsize=11)
    plt.title("Decomposition of Classification Errors: False Positives vs. False Negatives vs. Attack Confusion", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p_decomp = graphs_dir / "benign_vs_attack_confusion.png"
    plt.savefig(p_decomp, dpi=300)
    plt.close()
    print(f"      [SAVED] -> {p_decomp}")

    # 5. Class-wise F1 Heatmap Across Models
    if not cw_df.empty:
        pivot_f1 = cw_df.pivot(index="Class", columns="Model", values="F1-Score")
        plt.figure(figsize=(12, 14))
        sns.heatmap(pivot_f1, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "F1-Score"})
        plt.title("Per-Class F1-Score Heatmap Across All 5 Models (34 Classes)", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("Model Architecture", fontsize=11)
        plt.ylabel("Target Class", fontsize=11)
        plt.tight_layout()
        p_cw = graphs_dir / "classwise_f1_comparison.png"
        plt.savefig(p_cw, dpi=300)
        plt.close()
        print(f"      [SAVED] -> {p_cw}")


def main():
    print("=" * 70)
    print("PHASE 12: RIGOROUS CONFUSION MATRIX & MODEL ERROR ANALYSIS")
    print("=" * 70)

    conf_dir = PROJECT_ROOT / "results" / "confusion_matrices"
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    graphs_dir = PROJECT_ROOT / "results" / "graphs" / "error_analysis"

    # 1. Load Data
    X_test, y_test, label_mapping, id_to_label, target_names = load_data_and_labels()

    # 2. Get Predictions
    predictions = get_model_predictions(X_test, y_test, target_names)

    # 3. Compute Confusion Matrices
    cms = {}
    for model_name, y_pred in predictions.items():
        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(target_names))))
        cms[model_name] = cm
        print(f"      {model_name:<18} total matrix samples: {cm.sum():,} | trace (correct): {np.trace(cm):,}")

    # 4. Save Plots & Arrays
    plot_and_save_confusion_matrices(cms, target_names, conf_dir)

    # 5. Compute Error Analysis & Metrics
    misc_df, benign_df, err_comp_df = compute_classwise_and_misclassifications(
        cms, target_names, metrics_dir
    )

    cw_df = pd.read_csv(metrics_dir / "classwise_error_analysis.csv")

    # 6. Generate Diagnostic Visualizations
    generate_error_visualizations(cms, misc_df, benign_df, err_comp_df, cw_df, graphs_dir)

    print("\n" + "=" * 70)
    print("PHASE 12 CONFUSION MATRIX & ERROR ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
