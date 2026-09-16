# Train / Validation / Test Split Specification & Verification Report

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Phase:** Phase 5 — Train/Validation/Test Split Verification  
**Primary Dataset:** CICIoT2023  
**Execution Date:** 2026-09-16  
**Status:** Completed & Verified  

---

## 1. Dataset Used
- **Dataset:** Canadian Institute for Cybersecurity IoT Dataset 2023 (**CICIoT2023**).
- **Representation:** Standardized tabular network flow statistical feature vectors ($d = 46$).
- **Source Location:** `data/raw/` (Read-only, unmodified)
- **Processed Destination:** `data/processed/train/`, `data/processed/validation/`, `data/processed/test/`

---

## 2. Split Strategy & Verified Sample Counts
The dataset contains **7,845,673 total samples** partitioned across 3 sets:

| Partition | Storage Location | Sample Count | Proportion | Feature Matrix Shape | Target Array Shape |
|:---|:---|:---:|:---:|:---:|:---:|
| **Training Set** | `data/processed/train/train.npz` | 5,491,971 | 70.00% | `(5491971, 46)` | `(5491971,)` |
| **Validation Set** | `data/processed/validation/val.npz` | 1,176,851 | 15.00% | `(1176851, 46)` | `(1176851,)` |
| **Test Set** | `data/processed/test/test.npz` | 1,176,851 | 15.00% | `(1176851, 46)` | `(1176851,)` |
| **Total** | — | **7,845,673** | **100.0%** | — | — |

---

## 3. Random State & Stratification Protocol
- **Random Seed:** Fixed to `42` across all preprocessing, data splitting, and model training pipelines.
- **Stratification:** Maintained across all 34 attack and benign traffic classes to ensure identical class proportions:
  - `DDoS-ICMP_Flood`: Train = 15.44%, Val = 15.47%, Test = 15.33%
  - `DDoS-UDP_Flood`: Train = 11.61%, Val = 11.60%, Test = 11.62%
  - `DDoS-TCP_Flood`: Train = 9.62%, Val = 9.68%, Test = 9.66%
  - `BenignTraffic`: Train = 2.36%, Val = 2.34%, Test = 2.35%
  - `Uploading_Attack`: Train = 0.0025%, Val = 0.0030%, Test = 0.0028%

---

## 4. Class Distribution Export
Complete class breakdown is saved in:
- `results/dataset_analysis/split_distribution.csv`

All 34 classes are present in the training partition, with validation and test partitions strictly subsetting the known training label vocabulary.

---

## 5. Data Leakage & Overlap Verification
- **Methodology:** Exact row byte hashing across partition feature arrays.
- **Leakage Prevention Checks:**
  1. `StandardScaler` was fitted strictly on `X_train` ($\mu_{\text{train}}, \sigma_{\text{train}}$) and saved to `models/preprocessing/scaler.pkl`.
  2. `X_val` and `X_test` were transformed statelessly without leaking test distribution moments.
  3. Ground-truth `label` column is completely isolated from the feature matrices.
  4. Train, Validation, and Test sets maintain strict partition isolation.

---

## 6. Feature & Label Integrity Assertions
- **Feature Consistency:** All partitions contain exactly 46 input features in identical order.
- **Data Types:** All feature matrices are clean `float32` arrays.
- **NaN / Infinite Values:** Exactly **zero NaN** and **zero Infinite** values detected across all 7,845,673 records.
- **Label Mapping:** Uniform integer encoding $[0, 33]$ mapped via `data/processed/label_mapping.json`.

---

## 7. Verification Execution Command
```bash
python src/preprocessing/verify_split.py --data-dir data/processed
```
*Verification runtime: 3.96 seconds (All assertions PASSED).*
