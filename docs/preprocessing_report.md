# Phase 4 Technical Report: Data Preprocessing Pipeline

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Phase:** Phase 4 — Data Preprocessing  
**Target Dataset:** CICIoT2023  
**Status:** Pipeline Architecture & Modules Implemented (Awaiting raw dataset files in `data/raw/`)

---

## 1. Dataset Representation
- The offline CICIoT2023 dataset contains structured tabular flow and packet statistics (e.g., flow durations, packet counts, inter-arrival times, flag counts, protocol numbers).
- It is distinguished from raw payload byte streams, which are reserved for the future Phase 17 Scapy packet sniffer.

---

## 2. Preprocessing Architecture & Modules
The preprocessing pipeline is built with modular separation of concerns in `src/preprocessing/`:

```text
src/preprocessing/
├── __init__.py               # Package exports
├── load_data.py              # File discovery & chunked/sampled loading
├── clean_data.py             # Infinities, nulls, duplicates, and constant filtering
├── label_processing.py       # Target identification & deterministic JSON mapping
├── feature_processing.py     # Leakage-free StandardScaler/MinMaxScaler fitting
├── split_data.py             # Stratified 70/15/15 partitioning
└── preprocess_pipeline.py    # Master CLI pipeline entrypoint & automated validation
```

---

## 3. Data Cleaning Operations
1. **Infinities:** Replaced with `np.nan` and median-imputed using parameters derived exclusively from the training split.
2. **Missing Values:** Median imputation for numerical features; mode imputation for categorical attributes.
3. **Duplicates:** Exact duplicate records are detected and pruned.
4. **Zero-Variance Columns:** Constant columns with `nunique() <= 1` are removed.

---

## 4. Leakage Prevention Protocol
To guarantee strict scientific integrity:
- Splitting is performed **prior** to feature scaling.
- `StandardScaler` / `MinMaxScaler` is fit strictly on `X_train`.
- Validation and Test splits are transformed statelessly using the fitted parameters.
- Target column is strictly isolated from the feature matrix $X$.
- The fitted scaler is saved to `models/preprocessing/scaler.pkl` for inference reusability.

---

## 5. Train / Validation / Test Splitting
- **Training Set:** 70%
- **Validation Set:** 15%
- **Test Set:** 15%
- **Stratification:** Maintained across all splits to ensure equal representation of minority attack classes.
- **Random State:** 42

---

## 6. Preprocessing Artifacts
When executed on the dataset, the pipeline generates:
- `data/processed/train/train.npz`
- `data/processed/validation/val.npz`
- `data/processed/test/test.npz`
- `data/processed/label_mapping.json`
- `data/processed/preprocessing_metadata.json`
- `models/preprocessing/scaler.pkl`

---

## 7. Automated Verification Protocol
The pipeline enforces six automated validation assertions:
1. Zero NaN / Null values across all processed splits.
2. Zero Infinite values across all processed splits.
3. Identical feature column dimensions across Train, Val, and Test splits.
4. Zero Data Leakage: Target column excluded from feature space $X$.
5. Zero Data Leakage: Scaler was fitted strictly on the Training partition.
6. Deterministic label mapping saved to disk.
