# Phase 6 Technical Report: Random Forest Baseline Model

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Phase:** Phase 6 — Random Forest Baseline Model  
**Target Dataset:** CICIoT2023  
**Execution Date:** 2026-09-16  
**Status:** Completed & Empirically Verified  

---

## 1. Executive Summary

This report documents the implementation, training, and empirical evaluation of the **Random Forest Baseline Model**, establishing the primary machine learning benchmark for the ML-Powered Intrusion Detection System.

- **Classification Type:** Multiclass Intrusion Detection (34 distinct classes: 1 Benign Traffic class + 33 IoT Attack variants).
- **Dataset Source:** Preprocessed and standardized tabular feature arrays (`data/processed/`).
- **Input Dimension:** 46 standardized continuous and discrete network flow statistical features.
- **Training Samples:** 5,491,971 records (70.0% stratified partition).
- **Validation Samples:** 1,176,851 records (15.0% stratified partition).
- **Test Samples:** 1,176,851 records (15.0% stratified partition).
- **Test Accuracy:** **99.11%** (1,166,357 correctly classified test flows out of 1,176,851).
- **Test Weighted F1-Score:** **0.9917**
- **Test Macro F1-Score:** **0.7872**
- **Training Wall-Clock Time:** **120.24 seconds** (across 200 estimators).
- **Test Inference Latency:** **13.2685 seconds** (~11.27 microseconds per network flow).
- **Serialized Model Size:** **1,568.01 MB** (`models/random_forest.pkl`).

---

## 2. Dataset, Preprocessing, and Split Protocol

### 2.1 Dataset Representation
The model operates on structured, tabular network flow statistical summaries from the CICIoT2023 benchmark. The 46 features capture flow duration, inter-arrival time moments, packet size distributions, TCP control flags, protocol indicators, and statistical dispersion metrics.

### 2.2 Leakage Prevention & Preprocessing
1. **Zero Data Leakage:** All feature scalers (`PureStandardScaler`) and imputations were computed **exclusively on the training split** ($\mu_{\text{train}}, \sigma_{\text{train}}$) and applied statelessly to validation and test sets.
2. **Deterministic Partitioning:** Stratified 70/15/15 split fixed with `random_state=42`.
3. **No Target Leakage:** Target column `label` was isolated prior to feature matrix generation.

---

## 3. Model Architecture & Hyperparameter Configuration

The baseline is built using `sklearn.ensemble.RandomForestClassifier` with balanced tree depth and sub-sampling parameters:

| Hyperparameter | Value | Rationale & Description |
|:---|:---:|:---|
| `n_estimators` | `200` | High ensemble diversity to stabilize decision boundaries across 34 classes. |
| `max_depth` | `25` | Prevents individual trees from excessive memorization while capturing non-linear interactions. |
| `min_samples_split` | `5` | Minimum samples required to split an internal decision node. |
| `min_samples_leaf` | `2` | Minimum samples required in a terminal leaf node. |
| `max_features` | `"sqrt"` | Selects $\approx \sqrt{46} \approx 6.78 \rightarrow 6$ features per split for feature decorrelation. |
| `class_weight` | `"balanced"` | **Critical:** Automatically adjusts weights inversely proportional to class frequencies to combat the extreme 6,000:1 class imbalance ratio. |
| `max_samples` | `0.1` | Subsamples ~549,197 records per tree to ensure fast training while maintaining complete dataset coverage across 200 trees. |
| `random_state` | `42` | Guarantees deterministic reproducibility. |
| `n_jobs` | `-1` | Parallelizes tree construction across all available CPU cores. |

### 3.1 Class Weighting Rationale
In the training partition of 5,491,971 flows, the majority class `DDoS-ICMP_Flood` comprises 848,088 samples (15.44%), whereas minority attack classes like `Uploading_Attack` (140 samples, 0.0025%) and `Recon-PingSweep` (249 samples, 0.0045%) are heavily under-represented. Without `class_weight="balanced"`, decision trees greedily prioritize volumetric attacks and achieve near-zero recall on subtle infiltration vectors.

---

## 4. Empirical Evaluation Results

### 4.1 Overall Performance Summary

| Partition | Evaluated Samples | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted Precision | Weighted Recall | Weighted F1-Score | Inference Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Validation Set** | 1,176,851 | 99.13% | 0.7606 | 0.8525 | 0.7945 | 0.9928 | 0.9913 | 0.9918 | 14.7146s |
| **Test Set** | 1,176,851 | 99.11% | 0.7542 | 0.8456 | 0.7872 | 0.9927 | 0.9911 | 0.9917 | 13.2685s |

*Note: The test set was evaluated ONCE after finalizing baseline configurations.*

---

## 5. Complete 34-Class Breakdown (Test Set)

Below are the empirical per-class precision, recall, F1-score, and support metrics evaluated on the 1,176,851 test flows:

| Class Index | Class Name | Precision | Recall | F1-Score | Support (Test Flows) |
|:---:|:---|:---:|:---:|:---:|:---:|
| 0 | `Backdoor_Malware` | 0.3312 | 0.5843 | 0.4228 | 89 |
| 1 | `BenignTraffic` | 0.9549 | 0.8761 | 0.9138 | 27,709 |
| 2 | `BrowserHijacking` | 0.2042 | 0.5075 | 0.2912 | 134 |
| 3 | `CommandInjection` | 0.3202 | 0.6807 | 0.4355 | 119 |
| 4 | `DDoS-ACK_Fragmentation` | 0.9881 | 0.9934 | 0.9908 | 7,292 |
| 5 | `DDoS-HTTP_Flood` | 0.9126 | 0.9873 | 0.9485 | 709 |
| 6 | `DDoS-ICMP_Flood` | 1.0000 | 0.9988 | 0.9994 | 180,447 |
| 7 | `DDoS-ICMP_Fragmentation` | 0.9935 | 0.9956 | 0.9946 | 11,402 |
| 8 | `DDoS-PSHACK_Flood` | 1.0000 | 0.9991 | 0.9995 | 103,326 |
| 9 | `DDoS-RSTFINFlood` | 1.0000 | 0.9992 | 0.9996 | 101,819 |
| 10 | `DDoS-SYN_Flood` | 0.9996 | 0.9979 | 0.9988 | 102,208 |
| 11 | `DDoS-SlowLoris` | 0.6588 | 0.9936 | 0.7923 | 622 |
| 12 | `DDoS-SynonymousIP_Flood` | 0.9990 | 0.9984 | 0.9987 | 90,480 |
| 13 | `DDoS-TCP_Flood` | 0.9999 | 0.9984 | 0.9992 | 113,735 |
| 14 | `DDoS-UDP_Flood` | 0.9996 | 0.9978 | 0.9987 | 136,717 |
| 15 | `DDoS-UDP_Fragmentation` | 0.9845 | 0.9931 | 0.9888 | 7,224 |
| 16 | `DNS_Spoofing` | 0.7150 | 0.7856 | 0.7486 | 4,570 |
| 17 | `DictionaryBruteForce` | 0.2500 | 0.6458 | 0.3605 | 319 |
| 18 | `DoS-HTTP_Flood` | 0.8374 | 0.9961 | 0.9099 | 1,805 |
| 19 | `DoS-SYN_Flood` | 0.9990 | 0.9950 | 0.9970 | 51,116 |
| 20 | `DoS-TCP_Flood` | 0.9997 | 0.9979 | 0.9988 | 67,697 |
| 21 | `DoS-UDP_Flood` | 0.9986 | 0.9971 | 0.9978 | 83,627 |
| 22 | `MITM-ArpSpoofing` | 0.9030 | 0.8048 | 0.8511 | 7,840 |
| 23 | `Mirai-greeth_flood` | 0.9994 | 0.9940 | 0.9967 | 24,934 |
| 24 | `Mirai-greip_flood` | 0.9980 | 0.9954 | 0.9967 | 19,279 |
| 25 | `Mirai-udpplain` | 0.9989 | 0.9972 | 0.9980 | 22,536 |
| 26 | `Recon-HostDiscovery` | 0.7495 | 0.8694 | 0.8050 | 3,331 |
| 27 | `Recon-OSScan` | 0.5422 | 0.6889 | 0.6068 | 2,433 |
| 28 | `Recon-PingSweep` | 0.2000 | 0.2830 | 0.2344 | 53 |
| 29 | `Recon-PortScan` | 0.5721 | 0.7238 | 0.6391 | 2,082 |
| 30 | `SqlInjection` | 0.3769 | 0.5068 | 0.4323 | 148 |
| 31 | `Uploading_Attack` | 0.3000 | 0.3636 | 0.3288 | 33 |
| 32 | `VulnerabilityScan` | 0.5461 | 0.9989 | 0.7062 | 913 |
| 33 | `XSS` | 0.3095 | 0.5049 | 0.3838 | 103 |

---

## 6. Confusion Matrix & Diagnostic Observations

The confusion matrix was computed over all 1,176,851 test flows and exported to:
- `results/confusion_matrices/random_forest_confusion_matrix.png`
- `results/metrics/random_forest_misclassifications.csv` (10,494 total test errors)

### Key Observations:
1. **Volumetric Flood Dominance:** DoS and DDoS attack vectors (e.g. `DDoS-ICMP_Flood`, `DDoS-UDP_Flood`, `DDoS-TCP_Flood`, `DoS-SYN_Flood`) achieve near-perfect classification ($F_1 > 0.998$), demonstrating that Random Forest easily isolates high-rate statistical patterns.
2. **Benign vs. Recon Confusion:** A fraction of Benign traffic flows (3,433 samples, 12.39% of benign) were classified as `MITM-ArpSpoofing` or `DNS_Spoofing` due to overlapping inter-arrival time and packet rate metrics in low-intensity sessions.
3. **Web Application Attack Challenges:** Rare web attacks (`SqlInjection`, `CommandInjection`, `XSS`, `BrowserHijacking`) achieved lower precision ($0.20 - 0.38$) but acceptable recall ($0.50 - 0.68$) under balanced weighting. Because tabular flow statistics lack deep payload byte visibility, these attacks occasionally trigger false positives against general HTTP traffic.

---

## 7. Feature Importance Analysis

Gini-based impurity reduction was calculated across all 200 trees:
- Data Export: `results/metrics/random_forest_feature_importance.csv`
- Plot Export: `results/graphs/random_forest_feature_importance.png`

### Top 15 Most Discriminative Features:

| Rank | Feature Name | Gini Importance | Cumulative Importance | Physical Network Interpretation |
|:---:|:---|:---:|:---:|:---|
| 1 | `IAT` | 0.2062 | 20.62% | Inter-Arrival Time between consecutive packets (critical for burst/flood detection) |
| 2 | `Header_Length` | 0.0579 | 26.41% | Total protocol header byte length |
| 3 | `Protocol Type` | 0.0476 | 31.17% | Transport/network protocol number (TCP/UDP/ICMP) |
| 4 | `flow_duration` | 0.0418 | 35.35% | Total active lifetime of the bidirectional conversation |
| 5 | `rst_count` | 0.0405 | 39.40% | Cumulative TCP Reset flag count |
| 6 | `Tot size` | 0.0396 | 43.36% | Aggregate payload and header size in the flow |
| 7 | `Magnitue` | 0.0381 | 47.17% | Geometric magnitude across flow metric vectors |
| 8 | `AVG` | 0.0376 | 50.93% | Average packet size in the flow |
| 9 | `Max` | 0.0355 | 54.48% | Maximum observed packet length |
| 10 | `syn_count` | 0.0343 | 57.91% | Cumulative TCP SYN flag count (crucial for SYN floods) |
| 11 | `urg_count` | 0.0326 | 61.17% | Cumulative TCP Urgent flag count |
| 12 | `Min` | 0.0314 | 64.31% | Minimum packet length in the flow |
| 13 | `Tot sum` | 0.0303 | 67.34% | Total sum of packet lengths |
| 14 | `Rate` | 0.0285 | 70.19% | Overall packet transmission rate (packets/second) |
| 15 | `Srate` | 0.0264 | 72.83% | Source-to-destination packet transmission rate |

---

## 8. Model Comparison Table (Phase 6 State)

Stored at `results/metrics/model_comparison.csv`:

| Model | Accuracy | Precision Macro | Recall Macro | F1 Macro | Precision Weighted | Recall Weighted | F1 Weighted | Training Time (s) | Inference Time (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **0.9911** | **0.7542** | **0.8456** | **0.7872** | **0.9927** | **0.9911** | **0.9917** | **120.24** | **13.2685** |

*(Subsequent phases will append XGBoost, 1D-CNN, BiLSTM, and the proposed Hybrid CNN+BiLSTM model).*

---

## 9. Limitations of the Random Forest Baseline

1. **Large Serialized Memory Footprint:** At 1.568 GB (`models/random_forest.pkl`), storing 200 deep decision trees requires substantial memory for edge gateway deployments.
2. **Payload-Blindness on Web Attacks:** Because the CICIoT2023 dataset features represent flow-level packet statistics without application payload bytes, Random Forest relies solely on statistical timing/lengths for web injection attacks, limiting precision on subtle payload injections.
3. **No Sequential Memory:** Random Forest treats each flow record as an isolated, independent vector, discarding potential multi-step lateral movement sequences across consecutive time windows.

---

## 10. Summary of Generated Artifacts

- **Model Binary:** `models/random_forest.pkl` (1,568.01 MB)
- **Model Metadata:** `models/random_forest_metadata.json`
- **Validation Metrics:** `results/metrics/random_forest_validation.json`
- **Test Metrics:** `results/metrics/random_forest_test.json`
- **Per-Class Report:** `results/metrics/random_forest_classification_report.csv`
- **Feature Importance Data:** `results/metrics/random_forest_feature_importance.csv`
- **Feature Importance Plot:** `results/graphs/random_forest_feature_importance.png`
- **Confusion Matrix Plot:** `results/confusion_matrices/random_forest_confusion_matrix.png`
- **Misclassifications:** `results/metrics/random_forest_misclassifications.csv`
- **Model Comparison Table:** `results/metrics/model_comparison.csv`
- **Training Script:** `scripts/train_random_forest.py`
- **Interactive Notebook:** `notebooks/04_random_forest_baseline.ipynb`
- **Test Suite:** `tests/test_random_forest.py` (7 tests, all passing)
