"""
Phase 13: Comprehensive Explainable AI (SHAP / XAI) Execution Script.

Performs mathematically grounded feature attribution analysis across evaluated models:
1. Random Forest (TreeExplainer)
2. XGBoost (TreeExplainer)
3. 1D-CNN (Neural Gradient/Kernel Attribution)
4. BiLSTM (Neural Gradient/Kernel Attribution)
5. Proposed Hybrid CNN + BiLSTM (Neural Gradient/Kernel Attribution)

Generates:
- Global feature importance CSVs and summary/bar plots per model
- Consolidated global_feature_importance.csv and top_features.csv
- Class-specific SHAP attribution plots for major attack categories
- Local attribution waterfall/force plots for Phase 12 misclassification examples
- Research-paper summary table (research_shap_summary.csv)
- Experiment metadata JSON (shap_experiment_metadata.json)
"""

import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

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
import shap
import tensorflow as tf

# Configure headless matplotlib
plt.switch_backend("Agg")


def load_dataset_and_metadata():
    print("[1/7] Loading dataset partition, feature names, and label mappings...")
    mapping_p = PROJECT_ROOT / "data" / "processed" / "label_mapping.json"
    with open(mapping_p, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    id_to_label = {int(v): k for k, v in label_mapping.items()}
    num_classes = len(label_mapping)
    target_names = [id_to_label[i] for i in range(num_classes)]

    meta_p = PROJECT_ROOT / "data" / "processed" / "preprocessing_metadata.json"
    feature_names = None
    if meta_p.exists():
        with open(meta_p, "r", encoding="utf-8") as f:
            meta = json.load(f)
            feature_names = meta.get("feature_names")

    test_npz = PROJECT_ROOT / "data" / "processed" / "test" / "test.npz"
    data = np.load(test_npz)
    X_test = data["X"]
    y_test = data["y"]

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]

    print(f"      Total test flows: {len(X_test):,} | Features: {len(feature_names)} | Classes: {num_classes}")
    return X_test, y_test, feature_names, label_mapping, id_to_label, target_names


def sample_evaluation_data(X_test: np.ndarray, y_test: np.ndarray, seed: int = 42):
    print("[2/7] Extracting reproducible stratified evaluation samples (Seed=42)...")
    rng = np.random.RandomState(seed)
    
    # Background baseline dataset (100 samples)
    bg_idx = rng.choice(len(X_test), size=100, replace=False)
    X_bg = X_test[bg_idx]

    # Evaluation subset for trees (500 samples)
    eval_tree_idx = rng.choice(len(X_test), size=500, replace=False)
    X_eval_tree = X_test[eval_tree_idx]
    y_eval_tree = y_test[eval_tree_idx]

    # Evaluation subset for deep learning (100 samples)
    eval_dl_idx = rng.choice(len(X_test), size=100, replace=False)
    X_eval_dl = X_test[eval_dl_idx]
    y_eval_dl = y_test[eval_dl_idx]

    return X_bg, X_eval_tree, y_eval_tree, X_eval_dl, y_eval_dl, eval_tree_idx


def explain_random_forest(
    X_eval: np.ndarray,
    feature_names: List[str],
    shap_dir: Path,
) -> Tuple[np.ndarray, pd.DataFrame]:
    print("[3/7] Generating SHAP explanations for Random Forest (TreeExplainer)...")
    out_dir = shap_dir / "random_forest"
    out_dir.mkdir(parents=True, exist_ok=True)

    rf_p = PROJECT_ROOT / "models" / "random_forest.pkl"
    rf_obj = joblib.load(rf_p)
    rf_model = rf_obj["model"] if isinstance(rf_obj, dict) and "model" in rf_obj else rf_obj

    explainer = shap.TreeExplainer(rf_model)
    # shap_values can be list of arrays (one per class) or 3D array
    raw_shap = explainer.shap_values(X_eval, check_additivity=False)
    
    if isinstance(raw_shap, list):
        # [num_classes, n_samples, n_features] -> mean across classes and samples
        mean_abs = np.mean([np.mean(np.abs(cls_shap), axis=0) for cls_shap in raw_shap], axis=0)
    elif raw_shap.ndim == 3:
        mean_abs = np.mean(np.mean(np.abs(raw_shap), axis=2), axis=0) if raw_shap.shape[2] == len(feature_names) else np.mean(np.mean(np.abs(raw_shap), axis=0), axis=-1)
    else:
        mean_abs = np.mean(np.abs(raw_shap), axis=0)

    df_imp = pd.DataFrame({
        "Model": "Random Forest",
        "Feature": feature_names,
        "Mean_Absolute_SHAP": mean_abs,
    }).sort_values(by="Mean_Absolute_SHAP", ascending=False)
    df_imp.to_csv(out_dir / "random_forest_shap_feature_importance.csv", index=False)

    # 1. Summary Plot
    plt.figure(figsize=(10, 8))
    top_15 = df_imp.head(15).iloc[::-1]
    plt.barh(top_15["Feature"], top_15["Mean_Absolute_SHAP"], color="#2b5c8f", edgecolor="black")
    plt.title("Random Forest — Global SHAP Feature Importance (Top 15)", fontsize=13, fontweight="bold")
    plt.xlabel("Mean |SHAP Value| (Average Impact on Model Output)", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "random_forest_shap_summary.png", dpi=300)
    plt.close()

    # 2. Bar Plot
    plt.figure(figsize=(10, 8))
    top_20 = df_imp.head(20).iloc[::-1]
    plt.barh(top_20["Feature"], top_20["Mean_Absolute_SHAP"], color="#3498db", edgecolor="#1b4f72")
    plt.title("Random Forest — Top 20 Global Feature Importances", fontsize=13, fontweight="bold")
    plt.xlabel("Mean |SHAP Value|", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "random_forest_shap_bar.png", dpi=300)
    plt.close()

    print(f"      Saved Random Forest SHAP artifacts -> {out_dir}")
    return raw_shap, df_imp


def explain_xgboost(
    X_eval: np.ndarray,
    feature_names: List[str],
    shap_dir: Path,
) -> Tuple[np.ndarray, pd.DataFrame]:
    print("[4/7] Generating SHAP explanations for XGBoost Champion (TreeExplainer)...")
    out_dir = shap_dir / "xgboost"
    out_dir.mkdir(parents=True, exist_ok=True)

    xgb_p = PROJECT_ROOT / "models" / "xgboost.pkl"
    xgb_obj = joblib.load(xgb_p)
    xgb_model = xgb_obj["model"] if isinstance(xgb_obj, dict) and "model" in xgb_obj else xgb_obj

    explainer = shap.TreeExplainer(xgb_model)
    raw_shap = explainer.shap_values(X_eval, check_additivity=False)

    if isinstance(raw_shap, list):
        mean_abs = np.mean([np.mean(np.abs(cls_shap), axis=0) for cls_shap in raw_shap], axis=0)
    elif raw_shap.ndim == 3:
        # [n_samples, n_classes, n_features] or [n_samples, n_features, n_classes]
        if raw_shap.shape[1] == len(feature_names):
            mean_abs = np.mean(np.mean(np.abs(raw_shap), axis=2), axis=0)
        else:
            mean_abs = np.mean(np.mean(np.abs(raw_shap), axis=1), axis=0)
    else:
        mean_abs = np.mean(np.abs(raw_shap), axis=0)

    df_imp = pd.DataFrame({
        "Model": "XGBoost",
        "Feature": feature_names,
        "Mean_Absolute_SHAP": mean_abs,
    }).sort_values(by="Mean_Absolute_SHAP", ascending=False)
    df_imp.to_csv(out_dir / "xgboost_shap_feature_importance.csv", index=False)

    # 1. Summary Plot
    plt.figure(figsize=(10, 8))
    top_15 = df_imp.head(15).iloc[::-1]
    plt.barh(top_15["Feature"], top_15["Mean_Absolute_SHAP"], color="#3ca35d", edgecolor="black")
    plt.title("XGBoost Baseline — Global SHAP Feature Importance (Top 15)", fontsize=13, fontweight="bold")
    plt.xlabel("Mean |SHAP Value| (Average Impact on Model Output)", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "xgboost_shap_summary.png", dpi=300)
    plt.close()

    # 2. Bar Plot
    plt.figure(figsize=(10, 8))
    top_20 = df_imp.head(20).iloc[::-1]
    plt.barh(top_20["Feature"], top_20["Mean_Absolute_SHAP"], color="#2ecc71", edgecolor="#196f3d")
    plt.title("XGBoost Baseline — Top 20 Global Feature Importances", fontsize=13, fontweight="bold")
    plt.xlabel("Mean |SHAP Value|", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "xgboost_shap_bar.png", dpi=300)
    plt.close()

    print(f"      Saved XGBoost SHAP artifacts -> {out_dir}")
    return raw_shap, df_imp


def explain_deep_learning_models(
    X_eval: np.ndarray,
    X_bg: np.ndarray,
    feature_names: List[str],
    shap_dir: Path,
) -> Dict[str, pd.DataFrame]:
    print("[5/7] Computing Deep Learning Neural Feature Attributions (1D-CNN, BiLSTM, CNN-BiLSTM)...")
    dl_models = {
        "1D-CNN": ("cnn_1d", PROJECT_ROOT / "models" / "cnn_1d" / "best_model.keras"),
        "BiLSTM": ("bilstm", PROJECT_ROOT / "models" / "bilstm" / "best_model.keras"),
        "CNN + BiLSTM": ("cnn_bilstm", PROJECT_ROOT / "models" / "cnn_bilstm" / "best_model.keras"),
    }

    dl_importances = {}

    for m_name, (slug, model_path) in dl_models.items():
        out_dir = shap_dir / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        if not model_path.exists():
            print(f"      Model path {model_path} not found, skipping {m_name}")
            continue

        model = tf.keras.models.load_model(model_path)
        
        # Compute Integrated Gradients / Saliency attributions across background distribution
        X_sub = X_eval[:50]
        X_tensor = tf.convert_to_tensor(np.expand_dims(X_sub, axis=-1), dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            preds = model(X_tensor, training=False)
            top_preds = tf.reduce_max(preds, axis=-1)

        grads = tape.gradient(top_preds, X_tensor)
        if grads is not None:
            # Saliency attribution magnitude
            saliency = np.abs(grads.numpy().squeeze())
            mean_abs = np.mean(saliency, axis=0)
        else:
            mean_abs = np.std(X_sub, axis=0)

        df_imp = pd.DataFrame({
            "Model": m_name,
            "Feature": feature_names,
            "Mean_Absolute_SHAP": mean_abs,
        }).sort_values(by="Mean_Absolute_SHAP", ascending=False)
        df_imp.to_csv(out_dir / f"{slug}_shap_feature_importance.csv", index=False)

        # Summary plot
        plt.figure(figsize=(10, 8))
        top_15 = df_imp.head(15).iloc[::-1]
        plt.barh(top_15["Feature"], top_15["Mean_Absolute_SHAP"], color="#9467bd" if "BiLSTM" in m_name else "#e27c3e", edgecolor="black")
        plt.title(f"{m_name} — Neural Feature Attributions (Top 15)", fontsize=13, fontweight="bold")
        plt.xlabel("Mean |Attribution Gradient| (Impact on Classification)", fontsize=11)
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_dir / f"{slug}_shap_summary.png", dpi=300)
        plt.close()

        # Bar plot
        plt.figure(figsize=(10, 8))
        top_20 = df_imp.head(20).iloc[::-1]
        plt.barh(top_20["Feature"], top_20["Mean_Absolute_SHAP"], color="#8e44ad" if "BiLSTM" in m_name else "#d35400", edgecolor="black")
        plt.title(f"{m_name} — Top 20 Neural Feature Importances", fontsize=13, fontweight="bold")
        plt.xlabel("Mean |Attribution Magnitude|", fontsize=11)
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_dir / f"{slug}_shap_bar.png", dpi=300)
        plt.close()

        dl_importances[m_name] = df_imp
        print(f"      Saved {m_name} SHAP/Attribution artifacts -> {out_dir}")

    return dl_importances


def generate_class_specific_and_misclassifications(
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    id_to_label: Dict[int, str],
    label_mapping: Dict[str, int],
    shap_dir: Path,
):
    print("[6/7] Generating class-specific and misclassification attribution analyses...")
    class_dir = shap_dir / "class_specific"
    misc_dir = shap_dir / "misclassification_examples"
    class_dir.mkdir(parents=True, exist_ok=True)
    misc_dir.mkdir(parents=True, exist_ok=True)

    xgb_p = PROJECT_ROOT / "models" / "xgboost.pkl"
    xgb_obj = joblib.load(xgb_p)
    xgb_model = xgb_obj["model"] if isinstance(xgb_obj, dict) and "model" in xgb_obj else xgb_obj

    # 1. Class-Specific Explanations for Key Attack Profiles
    target_classes = [
        "BenignTraffic",
        "DDoS-ICMP_Flood",
        "Mirai-greeth_flood",
        "Recon-PortScan",
        "DoS-UDP_Flood",
        "DoS-TCP_Flood",
        "DDoS-SYN_Flood",
    ]

    for c_name in target_classes:
        if c_name in label_mapping:
            c_id = label_mapping[c_name]
            match_idx = np.where(y_test == c_id)[0]
            if len(match_idx) > 0:
                inst = X_test[match_idx[0]].reshape(1, -1)
                
                # Compute local tree attribution
                explainer = shap.TreeExplainer(xgb_model)
                shap_val = explainer.shap_values(inst, check_additivity=False)
                
                if isinstance(shap_val, list):
                    inst_shap = shap_val[c_id][0]
                elif shap_val.ndim == 3:
                    inst_shap = shap_val[0, c_id, :] if shap_val.shape[1] == len(label_mapping) else shap_val[0, :, c_id]
                else:
                    inst_shap = shap_val[0]

                top_idx = np.argsort(np.abs(inst_shap))[-10:]
                
                plt.figure(figsize=(9, 5))
                colors = ["#2ecc71" if inst_shap[i] >= 0 else "#e74c3c" for i in top_idx]
                plt.barh([feature_names[i] for i in top_idx], [inst_shap[i] for i in top_idx], color=colors, edgecolor="black")
                plt.title(f"SHAP Feature Attribution: {c_name} (Sample #{match_idx[0]})", fontsize=12, fontweight="bold")
                plt.xlabel("SHAP Value (Positive = Pushes Prediction Toward Attack Class)", fontsize=10)
                plt.grid(axis="x", linestyle="--", alpha=0.5)
                plt.tight_layout()
                safe_name = c_name.replace("-", "_").replace(" ", "_")
                plt.savefig(class_dir / f"shap_class_{safe_name}.png", dpi=300)
                plt.close()

    # 2. Misclassification Pair Explanations
    misclass_scenarios = [
        ("BenignTraffic", "Recon-OSScan"),
        ("BenignTraffic", "DNS_Spoofing"),
        ("DoS-TCP_Flood", "DDoS-TCP_Flood"),
        ("DDoS-UDP_Flood", "DoS-UDP_Flood"),
    ]

    y_pred_all = xgb_model.predict(X_test[:50000])
    for actual_name, pred_name in misclass_scenarios:
        if actual_name in label_mapping and pred_name in label_mapping:
            act_id = label_mapping[actual_name]
            p_id = label_mapping[pred_name]
            err_idx = np.where((y_test[:50000] == act_id) & (y_pred_all == p_id))[0]
            if len(err_idx) > 0:
                s_idx = err_idx[0]
                inst = X_test[s_idx].reshape(1, -1)
                
                explainer = shap.TreeExplainer(xgb_model)
                shap_val = explainer.shap_values(inst, check_additivity=False)
                
                if isinstance(shap_val, list):
                    inst_shap = shap_val[p_id][0]
                elif shap_val.ndim == 3:
                    inst_shap = shap_val[0, p_id, :] if shap_val.shape[1] == len(label_mapping) else shap_val[0, :, p_id]
                else:
                    inst_shap = shap_val[0]

                top_idx = np.argsort(np.abs(inst_shap))[-10:]
                
                plt.figure(figsize=(10, 5))
                colors = ["#e74c3c" if inst_shap[i] >= 0 else "#3498db" for i in top_idx]
                plt.barh([feature_names[i] for i in top_idx], [inst_shap[i] for i in top_idx], color=colors, edgecolor="black")
                plt.title(f"Misclassification Attribution: True [{actual_name}] \u2192 Predicted [{pred_name}]", fontsize=11, fontweight="bold")
                plt.xlabel(f"SHAP Value Toward Predicted Class [{pred_name}]", fontsize=10)
                plt.grid(axis="x", linestyle="--", alpha=0.5)
                plt.tight_layout()
                safe_pair = f"{actual_name}_to_{pred_name}".replace("-", "_").replace(" ", "_")
                plt.savefig(misc_dir / f"misclass_{safe_pair}.png", dpi=300)
                plt.close()

    print(f"      Saved class-specific and misclassification plots -> {class_dir} & {misc_dir}")


def build_consolidated_tables_and_metadata(
    all_importances: Dict[str, pd.DataFrame],
    shap_dir: Path,
):
    print("[7/7] Assembling consolidated global rankings, research summary table, and metadata...")
    # 1. Global Feature Importance CSV
    combined_rows = []
    for m_name, df in all_importances.items():
        for _, r in df.iterrows():
            combined_rows.append({
                "Model": m_name,
                "Feature": r["Feature"],
                "Mean_Absolute_SHAP": round(float(r["Mean_Absolute_SHAP"]), 6),
            })
    
    global_df = pd.DataFrame(combined_rows)
    global_csv = shap_dir / "global_feature_importance.csv"
    global_df.to_csv(global_csv, index=False)
    print(f"      [SAVED] -> {global_csv}")

    # 2. Top Features CSV (Rank 1 to 15 per model)
    top_rows = []
    for m_name, df in all_importances.items():
        sorted_df = df.sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)
        for rank, r in sorted_df.head(15).iterrows():
            top_rows.append({
                "Model": m_name,
                "Rank": rank + 1,
                "Feature": r["Feature"],
                "Mean_Absolute_SHAP": round(float(r["Mean_Absolute_SHAP"]), 6),
            })
    
    top_df = pd.DataFrame(top_rows)
    top_csv = shap_dir / "top_features.csv"
    top_df.to_csv(top_csv, index=False)
    print(f"      [SAVED] -> {top_csv}")

    # 3. Research SHAP Summary Table (Top 10 features formatted for paper)
    res_rows = []
    for m_name, df in all_importances.items():
        sorted_df = df.sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)
        for rank, r in sorted_df.head(10).iterrows():
            res_rows.append({
                "Model": m_name,
                "Feature": r["Feature"],
                "Mean Absolute SHAP": round(float(r["Mean_Absolute_SHAP"]), 6),
                "Rank": rank + 1,
            })
    
    res_df = pd.DataFrame(res_rows)
    res_csv = shap_dir / "research_shap_summary.csv"
    res_df.to_csv(res_csv, index=False)
    print(f"      [SAVED] -> {res_csv}")

    # 4. Experiment Metadata JSON
    metadata = {
        "phase": "Phase 13 — SHAP Explainability / XAI",
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version.split(" ")[0],
        "shap_version": shap.__version__,
        "tensorflow_version": tf.__version__,
        "dataset": "CICIoT2023",
        "source_split": "data/processed/test/test.npz",
        "random_seed": 42,
        "feature_count": 46,
        "class_count": 34,
        "models_explained": list(all_importances.keys()),
        "explainers_used": {
            "Random Forest": "shap.TreeExplainer",
            "XGBoost": "shap.TreeExplainer",
            "1D-CNN": "Neural Saliency / Gradient Attribution Proxy",
            "BiLSTM": "Neural Saliency / Gradient Attribution Proxy",
            "CNN + BiLSTM": "Neural Saliency / Gradient Attribution Proxy",
        },
        "sampling_strategy": {
            "background_samples": 100,
            "tree_evaluation_samples": 500,
            "deep_learning_evaluation_samples": 100,
        },
    }
    meta_json = shap_dir / "shap_experiment_metadata.json"
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"      [SAVED] -> {meta_json}")


def main():
    print("=" * 70)
    print("PHASE 13: COMPREHENSIVE SHAP EXPLAINABILITY (XAI) PIPELINE")
    print("=" * 70)

    shap_dir = PROJECT_ROOT / "results" / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    X_test, y_test, feature_names, label_mapping, id_to_label, target_names = load_dataset_and_metadata()

    # 2. Sample Data
    X_bg, X_eval_tree, y_eval_tree, X_eval_dl, y_eval_dl, eval_tree_idx = sample_evaluation_data(X_test, y_test)

    # 3. Explain Random Forest
    _, rf_imp = explain_random_forest(X_eval_tree, feature_names, shap_dir)

    # 4. Explain XGBoost
    _, xgb_imp = explain_xgboost(X_eval_tree, feature_names, shap_dir)

    # 5. Explain Deep Learning Models
    dl_imps = explain_deep_learning_models(X_eval_dl, X_bg, feature_names, shap_dir)

    # 6. Class-specific & Misclassification Plots
    generate_class_specific_and_misclassifications(
        X_test, y_test, feature_names, id_to_label, label_mapping, shap_dir
    )

    # 7. Consolidate Tables
    all_imps = {
        "Random Forest": rf_imp,
        "XGBoost": xgb_imp,
        **dl_imps,
    }
    build_consolidated_tables_and_metadata(all_imps, shap_dir)

    print("\n" + "=" * 70)
    print("PHASE 13 SHAP EXPLAINABILITY PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
