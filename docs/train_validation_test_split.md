# Train / Validation / Test Split Specification & Verification Report

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Phase:** Phase 5 — Train/Validation/Test Split Verification  
**Primary Dataset:** CICIoT2023  
**Status:** Verification Framework & Module Established (Awaiting raw dataset placement in `data/raw/`)

---

## 1. Dataset Used
- **Dataset:** Canadian Institute for Cybersecurity IoT Dataset 2023 (**CICIoT2023**).
- **Representation:** Flow-based statistical metrics and packet-header attributes.
- **Source Directory:** `data/raw/`
- **Destination Partitions:** `data/processed/train/`, `data/processed/validation/`, `data/processed/test/`

---

## 2. Split Strategy
- **Partition Ratios:**
  - **Training Set:** 70% (Used exclusively for model training and fitting transformers).
  - **Validation Set:** 15% (Used for hyperparameter tuning, early stopping, and model selection).
  - **Test Set:** 15% (Held-out untouched benchmark for final unbiased comparative evaluation).
- **Split Mechanism:** Two-stage stratified split:
  1. Split total data into **Train (70%)** and **Temp (30%)**.
  2. Split Temp into **Validation (15%)** and **Test (15%)** ($50\% - 50\%$ of Temp).

---

## 3. Random Seed
- **Fixed Random State:** `42` across all random number generators (`numpy`, `scikit-learn`, `tensorflow`).
- **Reproducibility Guarantee:** Guarantees deterministic partition indices across repeated executions.

---

## 4. Stratification Protocol
- **Stratified Sampling:** `stratify=y` enabled at every split stage.
- **Class Balance Preservation:** Every attack category (DoS, DDoS, Reconnaissance, Spoofing, Mirai, etc.) and benign traffic maintains proportional representation in all three partitions.

---

## 5. Partition Specifications

### 5.1 Training Set (`data/processed/train/`)
- **Role:** Model parameter learning (weights, trees, embeddings).
- **Leakage Isolation:** Scalers and encoders fit strictly on this partition.
- **Expected Samples / Features / Classes:** *Populated upon dataset placement in `data/raw/`.*

### 5.2 Validation Set (`data/processed/validation/`)
- **Role:** Epoch-level loss monitoring, learning rate scheduling (`ReduceLROnPlateau`), early stopping (`EarlyStopping`).
- **Leakage Isolation:** Transformed statelessly using training-fitted scaler.
- **Expected Samples / Features / Classes:** *Populated upon dataset placement in `data/raw/`.*

### 5.3 Test Set (`data/processed/test/`)
- **Role:** Final comparative evaluation across Random Forest, XGBoost, Standalone 1D-CNN, Standalone BiLSTM, and Proposed Hybrid CNN+BiLSTM.
- **Leakage Isolation:** Completely isolated from model fitting, feature selection, and threshold tuning.
- **Expected Samples / Features / Classes:** *Populated upon dataset placement in `data/raw/`.*

---

## 6. Class Distribution & Imbalance
- Export Target: `results/dataset_analysis/class_distribution_split.csv`
- Graph Target: `results/graphs/dataset_analysis/class_distribution_split.png`
- Any future class balancing (e.g., class-weighted loss functions) will apply **strictly to the training set**; validation and test sets remain unmanipulated to represent natural distribution.

---

## 7. Duplicate Overlap & Data Leakage Checks
- **Methodology:** Exact sample MD5 hashing across partition feature arrays.
- **Assertion:**
  - $\text{Overlap}(\text{Train}, \text{Validation}) = 0$
  - $\text{Overlap}(\text{Train}, \text{Test}) = 0$
  - $\text{Overlap}(\text{Validation}, \text{Test}) = 0$

---

## 8. Feature & Label Consistency
- **Feature Consistency:** $X_{\text{train}}, X_{\text{val}}, X_{\text{test}}$ have identical column count, identical column ordering, and identical data types (`float32`).
- **Zero NaN / Inf:** Automated validation asserts zero NaN and zero infinite values in all partitions.
- **Label Mapping:** Uniform integer encoding mapped via `data/processed/label_mapping.json`.

---

## 9. Verification Summary & CLI Command
To verify the processed split integrity:
```bash
python src/preprocessing/verify_split.py --data-dir data/processed
```
