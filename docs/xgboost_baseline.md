# XGBoost Baseline Model Report

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Phase:** Phase 7 — XGBoost Baseline Model  
**Primary Dataset:** CICIoT2023  
**Status:** Architecture, Training Pipeline & Evaluation Suite Implemented (Awaiting raw dataset placement in `data/raw/`)

---

## 1. Objective
Establish the second classical machine learning baseline using **XGBoost (Extreme Gradient Boosting)**. This baseline benchmarks gradient boosted decision trees against Random Forest before proceeding to deep learning architectures:
$$\text{Random Forest} \longrightarrow \text{XGBoost} \longrightarrow \text{1D-CNN} \longrightarrow \text{BiLSTM} \longrightarrow \text{Hybrid 1D-CNN + BiLSTM}$$

---

## 2. Dataset & Features
- **Source:** Processed partitions from `data/processed/` (`train.npz`, `val.npz`, `test.npz`).
- **Input Features:** Scaled flow and packet-level features ($Z$-score standardized via training set).
- **Target Isolation:** Target labels excluded from feature matrices.

---

## 3. Classification Strategy & Objective
- **Multiclass:** `objective='multi:softprob'`, `eval_metric='mlogloss'`.
- **Binary:** `objective='binary:logistic'`, `eval_metric='logloss'`.
- Configured dynamically based on unique class counts in `data/processed/label_mapping.json`.

---

## 4. Model Configuration & Hyperparameters
- **Estimators (`n_estimators`):** `300` boosting rounds
- **Max Depth (`max_depth`):** `6`
- **Learning Rate (`learning_rate`):** `0.1`
- **Subsample Ratio (`subsample`):** `0.8` (row subsampling per tree)
- **Column Sample Ratio (`colsample_bytree`):** `0.8` (feature subsampling per tree)
- **Min Child Weight (`min_child_weight`):** `1.0`
- **Regularization:** $\gamma = 0.0, \alpha = 0.0, \lambda = 1.0$
- **Random State:** `42`
- **Parallel Workers (`n_jobs`):** `-1`

---

## 5. Training Procedure
- Trained exclusively on `X_train, y_train` using `XGBoostBaselineModel.fit(X_train, y_train, eval_set=[(X_val, y_val)])`.
- Serialized to `models/xgboost.pkl`.
- Metadata saved to `models/xgboost_metadata.json`.

---

## 6. Evaluation Protocol
- **Validation Evaluation:** Evaluated on `X_val` to generate `results/metrics/xgboost_validation_metrics.json` and `results/confusion_matrices/xgboost_validation.png`.
- **Test Evaluation:** Evaluated once on `X_test` to generate `results/metrics/xgboost_test_metrics.json` and `results/confusion_matrices/xgboost_test.png`.
- **Per-Class Breakdown:** Saved to `results/metrics/xgboost_per_class.csv`.
- **Feature Importance:** Gain-based importance scores exported to `results/metrics/xgboost_feature_importance.csv` and plotted in `results/graphs/xgboost_feature_importance.png`.

---

## 7. Comparative Benchmark: Random Forest vs. XGBoost
Upon execution, `results/metrics/model_comparison.csv` updates with both traditional ML models:

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Macro F1 | Training Time (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest Baseline** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **XGBoost Baseline** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

---

## 8. Reproducibility & CLI Execution
```bash
python scripts/train_xgboost.py --data-dir data/processed --n-estimators 300 --max-depth 6
```
