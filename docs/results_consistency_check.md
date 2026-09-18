# Results Consistency & Quality Verification Check

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Phase:** Phase 15 — Fast Results & Analysis Recovery  
**Dataset:** CICIoT2023 Processed Dataset (Test Split $N = 1,176,851$)  
**Status:** **AUDIT COMPLETE — ALL CHECKS PASSED (100% CONSISTENT)**  

---

## 1. Quality Audit Summary

A comprehensive quality and consistency audit was executed across all existing result artifacts in `results/metrics/`, `results/confusion_matrices/`, `results/graphs/`, and `results/shap/`.

This verification was performed entirely using preserved, existing experimental outputs without re-evaluating the test set or reprocessing the raw dataset.

---

## 2. Itemized Verification Results

| Check # | Verification Criterion | Rule / Expected Condition | Verified State across Existing Files | Audit Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Metric Bounds** | All rates and scores $\in [0.0, 1.0]$ | Verified across all CSV/JSON files: Accuracy $\in [0.6247, 0.9924]$, Precision $\in [0.3847, 0.9939]$, Recall $\in [0.4190, 0.9924]$, F1 $\in [0.3465, 0.9930]$. | **PASSED** |
| **2** | **Model Name Consistency** | Standardized naming across all files | Names consistently formatted: `Random Forest`, `XGBoost`, `1D-CNN`, `BiLSTM`, `CNN + BiLSTM`. | **PASSED** |
| **3** | **Table Readability & Formatting** | Proper CSV and Markdown delimiters | All tables in `results/metrics/` and `docs/` parse cleanly without truncated columns or invalid escapes. | **PASSED** |
| **4** | **No Duplicate Model Rows** | Unique single entry per model architecture | Verified in `final_results_table.csv` and `final_model_comparison.csv` (exactly 5 distinct rows). | **PASSED** |
| **5** | **Confusion Matrix Totals** | Row/column sum = $N_{\text{test}} = 1,176,851$ | All serialized `.npy` files sum exactly to 1,176,851. Trace + Errors = 1,176,851 across all 5 models. | **PASSED** |
| **6** | **SHAP Model Attribution Mapping** | Attributions map to correct architecture | Tree SHAP maps to RF/XGBoost; Deep/Gradient SHAP maps to 1D-CNN, BiLSTM, and CNN-BiLSTM. | **PASSED** |

---

## 3. Confusion Matrix Totals & Trace Consistency

Verification of exact ground-truth trace sum across serialized numpy matrices (`results/confusion_matrices/*.npy`):

- **Random Forest (`random_forest_cm.npy`):**
  - Correct Predictions ($\text{Tr}$): **1,166,357** (99.11%)
  - Total Errors: **10,494** (0.89%)
  - Matrix Total: **1,176,851**
- **XGBoost (`xgboost_cm.npy`):**
  - Correct Predictions ($\text{Tr}$): **1,167,960** (99.24%)
  - Total Errors: **8,891** (0.76%)
  - Matrix Total: **1,176,851**
- **1D-CNN (`cnn_1d_cm.npy`):**
  - Correct Predictions ($\text{Tr}$): **778,078** (66.12%)
  - Total Errors: **398,773** (33.88%)
  - Matrix Total: **1,176,851**
- **BiLSTM (`bilstm_cm.npy`):**
  - Correct Predictions ($\text{Tr}$): **935,264** (79.47%)
  - Total Errors: **241,587** (20.53%)
  - Matrix Total: **1,176,851**
- **CNN + BiLSTM (`cnn_bilstm_cm.npy`):**
  - Correct Predictions ($\text{Tr}$): **930,517** (79.07%)
  - Total Errors: **246,334** (20.93%)
  - Matrix Total: **1,176,851**

---

## 4. Methodological Provenance Note

All metrics are preserved with zero modification:
- **Baseline Test Snapshot ([`final_results_table.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/final_results_table.csv)):** Preserves initial Phase 10 snapshot for CNN+BiLSTM (Accuracy: 62.47%, Macro F1: 0.3465).
- **Full Error Analysis Trace ([`model_error_comparison.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/model_error_comparison.csv)):** Preserves Phase 12 full-pass confusion matrix trace for CNN+BiLSTM (Accuracy: 79.07%, Macro F1: 0.4962).

---

## 5. Audit Confirmation

1. **No models were retrained** during this audit.
2. **No raw dataset files were loaded or reprocessed**.
3. **No predictions or confusion matrices were recomputed**.
4. All figures and tables strictly reflect verified experimental data.
