# Preprocessing Decisions & Leakage Prevention Architecture

## 1. Dataset Used
- **Target Dataset:** Canadian Institute for Cybersecurity IoT Dataset 2023 (**CICIoT2023**).
- **Location:** `data/raw/`
- **Representation:** Flow-based statistical and packet-header numerical feature vectors.

---

## 2. Target Column Identification
- The target label column is identified dynamically and validated against security class sets.
- Candidate labels: `label`, `Label`, `attack_type`, `Attack`.
- Supports both multi-class intrusion detection (e.g. DoS, DDoS, Recon, Spoofing, Mirai, Benign) and binary classification.

---

## 3. Removed Columns & Feature Selection Rationale

| Column Name | Action | Reason | Evidence & Rationale |
|:---|:---:|:---|:---|
| Constant / Zero-Variance Features | Drop | Zero information gain | Features with `nunique() <= 1` across all records provide zero discriminatory value to classifiers. |
| Non-network Identifiers (e.g. `Flow_ID`, `Timestamp`) | Drop / Omit | Target leakage prevention | Exact timestamps or sequential session IDs may allow models to memorize temporal partitions rather than attack patterns. |
| Secondary Duplicate Labels | Drop | Direct label leakage | Any auxiliary attack classification columns that directly encode or leak the ground-truth target. |

---

## 4. Missing Value Handling
- **Strategy:** Median imputation for numerical features; mode for categorical.
- **Leakage Prevention:** Imputation medians are computed **exclusively on the training split** and applied to validation/testing splits.

---

## 5. Infinite Value Handling
- Network flow rates and duration ratios can occasionally produce `+inf` or `-inf` (e.g. division by zero in packet rate calculation).
- All infinite values are converted to `np.nan` and subsequently imputed using feature medians computed from training partitions.

---

## 6. Duplicate Record Handling
- Exact duplicate rows are identified.
- Removal is evaluated to prevent identical packet flows from spanning both training and test partitions.

---

## 7. Categorical Feature Handling
- Any string/protocol features are dummy/one-hot encoded with alignment to training column signatures to ensure matching tensor dimensions.

---

## 8. Feature Scaling & Normalization
- **Scaler:** `StandardScaler` (Z-score standardization: $\mu = 0, \sigma = 1$) or `MinMaxScaler` ($[0, 1]$).
- **CRITICAL LEAKAGE RULE:** The scaler is fit strictly on `X_train`:
  $$X_{\text{train, scaled}} = \frac{X_{\text{train}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
  $$X_{\text{val, scaled}} = \frac{X_{\text{val}} - \mu_{\text{train}}}{\sigma_{\text{train}}}, \quad X_{\text{test, scaled}} = \frac{X_{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
- The fitted scaler is serialized to `models/preprocessing/scaler.pkl` for exact reproducibility during inference.

---

## 9. Label Encoding
- Target class strings are sorted and mapped deterministically to integers $[0, N-1]$, with `Benign` mapped to index `0` if present.
- The mapping dictionary is serialized to `data/processed/label_mapping.json`.

---

## 10. Data Splitting Strategy
- **Split Ratio:** 70% Train, 15% Validation, 15% Test.
- **Stratification:** Enabled (`stratify=y`) to maintain identical class ratios across all partitions, protecting rare attack classes.
- **Random Seed:** Fixed to `42` for strict experimental reproducibility.

---

## 11. Final Feature Set
- Exported as compressed numpy arrays:
  - `data/processed/train/train.npz`
  - `data/processed/validation/val.npz`
  - `data/processed/test/test.npz`
- Accompanied by `data/processed/preprocessing_metadata.json`.
