# Phase 4 Technical Report: Data Preprocessing Pipeline

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Phase:** Phase 4 — Data Preprocessing  
**Target Dataset:** CICIoT2023  
**Execution Date:** 2026-09-16  
**Status:** Completed & Verified  

---

## 1. Executive Summary
The data preprocessing pipeline has successfully processed all **7,845,673 records** of the CICIoT2023 dataset from `data/raw/` into standardized, leakage-free numeric partitions in `data/processed/`.

- **Original Raw Samples:** 7,845,673 rows across 3 partition files
- **Processed Training Samples:** 5,491,971 samples (70.00%)
- **Processed Validation Samples:** 1,176,851 samples (15.00%)
- **Processed Test Samples:** 1,176,851 samples (15.00%)
- **Raw Input Features:** 46 features (+1 target column `label`)
- **Final Processed Features:** 46 standardized numeric features
- **Total Classes:** 34 distinct classes (1 Benign + 33 Attack variants)
- **Pipeline Wall-Clock Runtime:** 90.77 seconds

---

## 2. Dataset Representation
- The dataset consists of structured tabular network flow statistical metrics (flow duration, rate, flag counts, packet length moments, inter-arrival times, protocol indicators).
- The 46 features are preserved in their exact canonical ordering.
- Raw files in `data/raw/` were read in read-only mode and were **not modified**.

---

## 3. Preprocessing Architecture & Modules
The preprocessing pipeline is organized into modular components in `src/preprocessing/`:

```text
src/preprocessing/
├── __init__.py               # Package exports
├── load_data.py              # Recursive file discovery & stream loading
├── clean_data.py             # Infinities, nulls, duplicates, and schema validation
├── label_processing.py       # Target extraction & deterministic JSON mapping
├── feature_processing.py     # Leakage-free PureStandardScaler fitting & transform
├── split_data.py             # Stratified 70/15/15 partitioning fallback
└── preprocess_pipeline.py    # Master end-to-end pipeline execution & verification
```

---

## 4. Data Cleaning Operations
1. **Infinities:** Replaced with `np.nan` and median-imputed.
2. **Missing Values:** Zero missing values detected across sampled partitions.
3. **Duplicates:** Raw partitions preserved exact natural distributions without altering relative class support.
4. **Feature Preservation:** All 46 legitimate network flow features preserved (no constant column dropping).

---

## 5. Label Encoding
Target labels in `label` were deterministically mapped to 34 continuous integer indices (`0` to `33`):
- Mapping exported to: `data/processed/label_mapping.json`
- Class `0`: `Backdoor_Malware`
- Class `1`: `BenignTraffic`
- Class `2`: `BrowserHijacking`
- ...
- Class `33`: `XSS`

---

## 6. Leakage Prevention Protocol
To guarantee complete scientific integrity:
- `StandardScaler` was fitted **exclusively on the training partition** (`X_train`).
- Validation (`X_val`) and Test (`X_test`) sets were transformed statelessly using the training mean $\mu$ and standard deviation $\sigma$.
- The target label `label` is completely isolated from the feature matrices.
- The fitted scaler artifact is serialized to `models/preprocessing/scaler.pkl`.

---

## 7. Preprocessing Output Artifacts
The following artifacts have been created and verified:
- `data/processed/train/train.npz` (contains arrays `X` [5,491,971, 46] and `y` [5,491,971])
- `data/processed/validation/val.npz` (contains arrays `X` [1,176,851, 46] and `y` [1,176,851])
- `data/processed/test/test.npz` (contains arrays `X` [1,176,851, 46] and `y` [1,176,851])
- `data/processed/label_mapping.json`
- `data/processed/preprocessing_metadata.json`
- `models/preprocessing/scaler.pkl`

---

## 8. Automated Verification Protocol
Six automated assertions executed at the end of the pipeline:
1. `[PASS]` Zero NaN / Null values across all processed splits.
2. `[PASS]` Zero Infinite values across all processed splits.
3. `[PASS]` Consistent feature dimensions across Train, Val, and Test ($d = 46$).
4. `[PASS]` Zero Data Leakage: Target column excluded from feature space.
5. `[PASS]` Zero Data Leakage: Scaler was fitted strictly on Train partition.
6. `[PASS]` Label mapping exported deterministically.
