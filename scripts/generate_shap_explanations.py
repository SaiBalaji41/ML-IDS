"""
Explainable AI (XAI) Generation Script using SHAP.
Computes and saves:
1. Global feature importance summary plots for XGBoost, Random Forest, and Hybrid CNN-BiLSTM.
2. Attack-specific local feature attribution plots (DDoS, DoS, Reconnaissance, Benign).
3. Exported feature attribution CSV tables to results/shap/ and results/metrics/.
"""

import json
import os
from pathlib import Path
import platform
import sys
import time

def _safe_get_machine_win32():
    return os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64")
platform._get_machine_win32 = _safe_get_machine_win32

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from src.explainability.shap_explainer import SHAPExplainer
from src.preprocessing.verify_split import load_partition


def main():
    print("=" * 60)
    print("EXPLAINABLE AI (XAI): GENERATING SHAP ATTRIBUTIONS")
    print("=" * 60)

    results_dir = PROJECT_ROOT / "results"
    shap_dir = results_dir / "shap"
    metrics_dir = results_dir / "metrics"
    shap_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load label mapping and feature names
    mapping_path = PROJECT_ROOT / "data" / "processed" / "label_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    id_to_label = {v: k for k, v in label_mapping.items()}
    num_classes = len(label_mapping)
    target_names = [id_to_label[i] for i in range(num_classes)]

    meta_path = PROJECT_ROOT / "data" / "processed" / "preprocessing_metadata.json"
    feature_names = None
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            feature_names = meta.get("feature_names")

    # 2. Load a representative sample from test partition
    print("Loading test partition sample for SHAP background and evaluation...")
    X_test, y_test = load_partition(PROJECT_ROOT / "data" / "processed" / "test", "test")
    sample_indices = np.random.RandomState(42).choice(len(X_test), size=1000, replace=False)
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]

    # 3. XGBoost SHAP Explanations
    xgb_path = PROJECT_ROOT / "models" / "xgboost.pkl"
    if xgb_path.exists():
        print("\nComputing SHAP explanations for XGBoost...")
        xgb_payload = joblib.load(xgb_path)
        xgb_raw = xgb_payload.get("model", xgb_payload) if isinstance(xgb_payload, dict) else getattr(xgb_payload, "model", xgb_payload)
        explainer_xgb = SHAPExplainer(
            model=xgb_raw,
            background_data=X_sample[:100],
            feature_names=feature_names,
            class_names=target_names,
        )

        # Global summary plot
        df_xgb_imp = explainer_xgb.explain_global(
            test_data=X_sample,
            save_path=shap_dir / "shap_global_summary_xgboost.png",
            max_display=15,
            title="SHAP Global Feature Importance — XGBoost Baseline",
        )
        df_xgb_imp.to_csv(metrics_dir / "shap_feature_importance_xgboost.csv", index=False)
        print("Exported XGBoost global SHAP summary plot and importance table.")

        # Local instance plots for diverse attack categories
        attack_targets = ["BenignTraffic", "DDoS-ICMP_Flood", "Mirai-greeth_flood", "Recon-PortScan"]
        for target_cat in attack_targets:
            if target_cat in label_mapping:
                cat_id = label_mapping[target_cat]
                matching_idx = np.where(y_sample == cat_id)[0]
                if len(matching_idx) > 0:
                    idx = matching_idx[0]
                    inst = X_sample[idx]
                    pred_id = xgb_raw.predict(inst.reshape(1, -1))[0]
                    pred_label = id_to_label.get(int(pred_id), str(pred_id))

                    safe_cat = target_cat.replace("-", "_").replace(" ", "_")
                    explainer_xgb.explain_instance(
                        instance=inst,
                        instance_idx=idx,
                        true_label=target_cat,
                        predicted_label=pred_label,
                        save_path=shap_dir / f"shap_local_instance_{safe_cat}.png",
                    )
                    print(f"Generated local attribution plot for {target_cat}.")

    # 4. Random Forest SHAP Explanations
    rf_path = PROJECT_ROOT / "models" / "random_forest.pkl"
    if rf_path.exists():
        print("\nComputing SHAP explanations for Random Forest...")
        rf_payload = joblib.load(rf_path)
        rf_raw = rf_payload.get("model", rf_payload) if isinstance(rf_payload, dict) else getattr(rf_payload, "model", rf_payload)
        explainer_rf = SHAPExplainer(
            model=rf_raw,
            background_data=X_sample[:100],
            feature_names=feature_names,
            class_names=target_names,
        )

        df_rf_imp = explainer_rf.explain_global(
            test_data=X_sample,
            save_path=shap_dir / "shap_global_summary_random_forest.png",
            max_display=15,
            title="SHAP Global Feature Importance — Random Forest Baseline",
        )
        df_rf_imp.to_csv(metrics_dir / "shap_feature_importance_random_forest.csv", index=False)
        print("Exported Random Forest global SHAP summary plot and importance table.")

    # 5. Proposed Hybrid CNN-BiLSTM Explanations
    cnn_bilstm_path = PROJECT_ROOT / "models" / "cnn_bilstm" / "best_model.keras"
    if cnn_bilstm_path.exists():
        print("\nComputing SHAP explanations for Proposed Hybrid CNN-BiLSTM...")
        from src.models.cnn_bilstm import CNNBiLSTMModel
        hybrid_model = CNNBiLSTMModel()
        hybrid_model.load(cnn_bilstm_path)
        explainer_hybrid = SHAPExplainer(
            model=hybrid_model,
            background_data=X_sample[:50],
            feature_names=feature_names,
            class_names=target_names,
        )

        df_hybrid_imp = explainer_hybrid.explain_global(
            test_data=X_sample[:50],
            save_path=shap_dir / "shap_global_summary_cnn_bilstm.png",
            max_display=15,
            title="SHAP Global Feature Importance — Hybrid CNN-BiLSTM (Proposed)",
        )
        df_hybrid_imp.to_csv(metrics_dir / "shap_feature_importance_cnn_bilstm.csv", index=False)
        print("Exported Hybrid CNN-BiLSTM global SHAP summary plot and importance table.")

    print("\nSHAP Explainability generation completed successfully!")


if __name__ == "__main__":
    main()
