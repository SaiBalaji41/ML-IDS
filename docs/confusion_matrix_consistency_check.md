# Confusion Matrix Data Consistency & Integrity Check (Phase 12)

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Dataset:** CICIoT2023 Processed Test Partition  
**Evaluation Date:** September 2026  
**Status:** **PASSED & VERIFIED (Zero Inconsistencies)**

---

## 1. Executive Summary
This document provides formal verification of experimental consistency across all five evaluated candidate machine learning and deep learning architectures for Phase 12. To prevent experimental bias or data leakage, all confusion matrices, error tables, and diagnostic metrics were strictly computed under an identical, leak-free evaluation protocol.

---

## 2. Verification Checklist

| Consistency Dimension | Requirement | Observed Verification | Status |
| :--- | :--- | :--- | :---: |
| **Test Dataset File** | Exact identical test partition | `data/processed/test/test.npz` | **PASSED** |
| **Total Test Samples** | 1,176,851 instances | Exactly 1,176,851 samples evaluated | **PASSED** |
| **Feature Dimension** | 46 engineered features | `(1176851, 46)` across all models | **PASSED** |
| **Number of Classes** | 34 multiclass categories | 34 classes indexed `[0, 33]` | **PASSED** |
| **Label Mapping** | `data/processed/label_mapping.json` | Identical mapping across all models | **PASSED** |
| **Class Label Order** | Alphabetical/Index aligned | Identical target names vector | **PASSED** |
| **Feature Scaling** | Stateless transform using train scaler | `StandardScaler` fitted on train partition only | **PASSED** |
| **Stratification** | Class ratio preserved from raw corpus | Stratified test split verified | **PASSED** |
| **Evaluation Pass** | Single unbiased test pass | No retraining or test-set tuning | **PASSED** |

---

## 3. Confusion Matrix Totals & Trace Verification

For a mathematically valid confusion matrix $C \in \mathbb{R}^{34 \times 34}$:
- $\sum_{i=1}^{34} \sum_{j=1}^{34} C_{i, j} = N_{\text{test}} = 1,176,851$
- Trace $\text{Tr}(C) = \sum_{k=1}^{34} C_{k, k} = N_{\text{correct}}$
- Total Errors $= N_{\text{test}} - \text{Tr}(C)$

### Verified Model Matrix Dimensions:

```text
1. Random Forest Baseline:
   - Matrix Dimension: [34, 34]
   - Total Sum: 1,176,851 (100.0%)
   - Trace (Correct): 1,166,357 (99.11%)
   - Misclassifications: 10,494 (0.89%)
   - Non-negative elements: VERIFIED (min >= 0)

2. XGBoost Baseline (Champion):
   - Matrix Dimension: [34, 34]
   - Total Sum: 1,176,851 (100.0%)
   - Trace (Correct): 1,167,960 (99.24%)
   - Misclassifications: 8,891 (0.76%)
   - Non-negative elements: VERIFIED (min >= 0)

3. 1D-CNN Deep Learning:
   - Matrix Dimension: [34, 34]
   - Total Sum: 1,176,851 (100.0%)
   - Trace (Correct): 778,078 (66.12%)
   - Misclassifications: 398,773 (33.88%)
   - Non-negative elements: VERIFIED (min >= 0)

4. Standalone BiLSTM:
   - Matrix Dimension: [34, 34]
   - Total Sum: 1,176,851 (100.0%)
   - Trace (Correct): 935,264 (79.47%)
   - Misclassifications: 241,587 (20.53%)
   - Non-negative elements: VERIFIED (min >= 0)

5. Proposed Hybrid CNN + BiLSTM:
   - Matrix Dimension: [34, 34]
   - Total Sum: 1,176,851 (100.0%)
   - Trace (Correct): 930,517 (79.07%)
   - Misclassifications: 246,334 (20.93%)
   - Non-negative elements: VERIFIED (min >= 0)
```

---

## 4. Inconsistency and Data Leakage Audit Findings
- **Zero Inconsistencies Detected:** All confusion matrices match the ground-truth test partition sample count to the exact integer.
- **Zero Label Drift:** Target class indices strictly correspond with `data/processed/label_mapping.json`.
- **Reproducibility Guaranteed:** Fixed evaluation seed (42) and pre-saved deterministic model binaries.
