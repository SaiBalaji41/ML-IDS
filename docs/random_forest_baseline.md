# Random Forest Baseline Model Report

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Phase:** Phase 6 — Random Forest Baseline Model  
**Primary Dataset:** CICIoT2023  
**Status:** Architecture, Training Script & Evaluation Suite Implemented (Awaiting raw dataset placement in `data/raw/`)

---

## 1. Objective
Establish the first classical machine learning baseline for intrusion detection. The Random Forest model provides a competitive tree-ensemble benchmark to compare against:
- Baseline 2: **XGBoost**
- Deep Learning: **Standalone 1D-CNN** & **Standalone BiLSTM**
- Proposed Architecture: **Hybrid 1D-CNN + BiLSTM**

---

## 2. Dataset & Partitioning
- **Input Source:** Verified processed partitions from Phase 5 (`data/processed/train/`, `data/processed/validation/`, `data/processed/test/`).
- **Isolation Protocol:**
  - `X_train, y_train`: Used strictly for fitting ensemble decision trees.
  - `X_val, y_val`: Used for validation tuning and depth configuration.
  - `X_test, y_test`: Held-out for final evaluation.

---

## 3. Input Features & Representation
- Standardized numerical flow and header features ($Z$-score standardized via training set parameters).
- Target column excluded from feature matrices.

---

## 4. Model Configuration & Hyperparameters
- **Estimator Count (`n_estimators`):** `200`
- **Criterion:** Gini impurity
- **Max Depth (`max_depth`):** `None` (unconstrained growth until pure leaves or min split threshold)
- **Min Samples Split:** `2`
- **Min Samples Leaf:** `1`
- **Max Features:** `"sqrt"`
- **Random State:** `42`
- **Parallel Workers (`n_jobs`):** `-1` (all available CPU threads)

---

## 5. Training Procedure
- Fitted exclusively on training samples using `RandomForestBaselineModel.fit(X_train, y_train)`.
- Runtime duration recorded to `results/metrics/random_forest_runtime.json`.
- Model artifact serialized to `models/random_forest.pkl`.
- Metadata saved to `models/random_forest_metadata.json`.

---

## 6. Evaluation Protocol
- **Validation Evaluation:** Evaluated on `X_val` to generate `results/metrics/random_forest_validation_metrics.json` and `results/confusion_matrices/random_forest_validation.png`.
- **Test Evaluation:** Evaluated on `X_test` once to generate `results/metrics/random_forest_test_metrics.json` and `results/confusion_matrices/random_forest_test.png`.
- **Metrics Computed:**
  - Accuracy
  - Weighted & Macro Precision
  - Weighted & Macro Recall
  - Weighted & Macro F1-Score
  - Per-class breakdown saved to `results/metrics/random_forest_per_class.csv`.

---

## 7. Feature Importance & Explainability
- Gini-based feature importances extracted from tree splits.
- Exported to `results/metrics/random_forest_feature_importance.csv`.
- Top 20 important features visualized in `results/graphs/random_forest_feature_importance.png`.

---

## 8. Comparative Benchmark Status
Upon execution, results populate `results/metrics/model_comparison.csv`:

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Macro F1 | Training Time (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest Baseline** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

---

## 9. Reproducibility & CLI Execution
```bash
python scripts/train_random_forest.py --data-dir data/processed --n-estimators 200
```
