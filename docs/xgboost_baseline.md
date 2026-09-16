# Phase 7 Technical Report: XGBoost Baseline Model

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Phase:** Phase 7 — XGBoost Baseline Model  
**Target Dataset:** CICIoT2023  
**Execution Date:** 2026-09-16  
**Status:** Completed & Empirically Verified  

---

## 1. Executive Summary

This report presents the implementation, training, and empirical evaluation of the **XGBoost (Extreme Gradient Boosting)** baseline classifier, establishing the second classical machine learning benchmark for the ML-Powered Intrusion Detection System.

- **Classification Type:** Multiclass Intrusion Detection (34 distinct classes: 1 Benign Traffic class + 33 IoT Attack variants).
- **XGBoost Version:** `3.4.1`
- **Dataset Source:** Verified preprocessed numeric feature arrays (`data/processed/`).
- **Input Dimension:** 46 standardized continuous and discrete network flow statistical features.
- **Training Samples:** 750,609 stratified records (from 5,491,971 available training records).
- **Validation Samples:** 1,176,851 records (15.0% stratified partition).
- **Test Samples:** 1,176,851 records (15.0% stratified partition).
- **Test Accuracy:** **99.24%** (1,167,960 correctly classified test flows out of 1,176,851).
- **Test Weighted F1-Score:** **0.9930**
- **Test Macro F1-Score:** **0.7928**
- **Test Macro Recall:** **0.8597**
- **Test Macro Precision:** **0.7701**
- **Training Wall-Clock Time:** **158.73 seconds**.
- **Test Inference Latency:** **12.3201 seconds** (~10.47 microseconds per network flow).
- **Serialized Model Size:** **7.36 MB** (`models/xgboost.pkl`) — **213x smaller** than Random Forest (1,568.01 MB).

---

## 2. Dataset, Preprocessing, and Split Protocol

### 2.1 Dataset Representation
The model operates on identical standardized tabular network flow statistical summaries from the CICIoT2023 benchmark as used in Phase 6. All 46 numerical input features remain identical in definition, scale, and ordering.

### 2.2 Leakage Prevention
1. **Zero Data Leakage:** `PureStandardScaler` parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) were fitted strictly on the training partition.
2. **Deterministic Partitioning:** Stratified 70/15/15 split preserved using `random_state=42`.
3. **Target Isolation:** Ground truth `label` was excluded prior to feature matrix construction.

---

## 3. Model Architecture & Hyperparameter Configuration

The baseline is implemented via `xgboost.XGBClassifier` utilizing the histogram-based tree building algorithm:

| Hyperparameter | Value | Description & Rationale |
|:---|:---:|:---|
| `n_estimators` | `100` | Number of gradient boosted boosting rounds. |
| `max_depth` | `6` | Shallow tree depth to prevent overfitting and capture robust interaction terms. |
| `learning_rate` ($\eta$) | `0.1` | Step size shrinkage to prevent overshooting during gradient descent. |
| `subsample` | `0.8` | Row subsampling ratio per boosting iteration to introduce bagging diversity. |
| `colsample_bytree` | `0.8` | Column subsampling ratio per tree to prevent dominant features from masking subtle signals. |
| `objective` | `"multi:softprob"` | Softmax loss function generating calibrated class probability distributions over 34 classes. |
| `eval_metric` | `"mlogloss"` | Multiclass logarithmic loss for monitoring gradient convergence. |
| `tree_method` | `"hist"` | Optimized histogram binning algorithm for fast CPU parallel training on large datasets. |
| `random_state` | `42` | Guarantees deterministic reproducibility. |
| `n_jobs` | `-1` | Utilizes all available CPU cores. |

---

## 4. Class Imbalance Handling

### 4.1 Sample Weighting Strategy
To address the extreme **6,000:1 class imbalance** without synthetic oversampling (no SMOTE to avoid distorting high-dimensional network flow manifolds), **balanced sample weights** were computed **exclusively on the training partition**:

$$w_i = \frac{N_{\text{train}}}{K \cdot N_{c(i)}}$$

Where:
- $N_{\text{train}} = 750,609$ is the training sample size.
- $K = 34$ is the number of distinct classes.
- $N_{c(i)}$ is the sample count of class $c$ associated with sample $i$.
- **Sample Weight Range:** $0.1906$ (for majority `DDoS-ICMP_Flood`) to $157.6910$ (for minority `Uploading_Attack`).

Validation and test distributions remained completely unaltered to reflect the true natural traffic distribution.

---

## 5. Empirical Evaluation Results

### 5.1 Validation vs. Test Metrics

| Partition | Evaluated Samples | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted Precision | Weighted Recall | Weighted F1-Score | Inference Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Validation Set** | 1,176,851 | 99.25% | 0.7712 | 0.8624 | 0.7975 | 0.9939 | 0.9925 | 0.9931 | 12.4159s |
| **Test Set** | 1,176,851 | 99.24% | 0.7701 | 0.8597 | 0.7928 | 0.9939 | 0.9924 | 0.9930 | 12.3201s |

---

## 6. Complete 34-Class Breakdown (Test Set)

| Class Index | Class Name | Precision | Recall | F1-Score | Support |
|:---:|:---|:---:|:---:|:---:|:---:|
| 0 | `Backdoor_Malware` | 0.2181 | 0.5955 | 0.3193 | 89 |
| 1 | `BenignTraffic` | 0.9587 | 0.8721 | 0.9133 | 27,709 |
| 2 | `BrowserHijacking` | 0.3132 | 0.6567 | 0.4241 | 134 |
| 3 | `CommandInjection` | 0.2692 | 0.5294 | 0.3569 | 119 |
| 4 | `DDoS-ACK_Fragmentation` | 0.9982 | 0.9995 | 0.9988 | 7,292 |
| 5 | `DDoS-HTTP_Flood` | 0.9886 | 0.9817 | 0.9851 | 709 |
| 6 | `DDoS-ICMP_Flood` | 1.0000 | 0.9999 | 1.0000 | 180,447 |
| 7 | `DDoS-ICMP_Fragmentation` | 0.9988 | 0.9994 | 0.9991 | 11,402 |
| 8 | `DDoS-PSHACK_Flood` | 1.0000 | 1.0000 | 1.0000 | 103,326 |
| 9 | `DDoS-RSTFINFlood` | 1.0000 | 0.9999 | 0.9999 | 101,819 |
| 10 | `DDoS-SYN_Flood` | 0.9996 | 0.9993 | 0.9995 | 102,208 |
| 11 | `DDoS-SlowLoris` | 0.9748 | 0.9968 | 0.9857 | 622 |
| 12 | `DDoS-SynonymousIP_Flood` | 0.9992 | 0.9996 | 0.9994 | 90,480 |
| 13 | `DDoS-TCP_Flood` | 0.9999 | 0.9992 | 0.9996 | 113,735 |
| 14 | `DDoS-UDP_Flood` | 0.9998 | 0.9997 | 0.9997 | 136,717 |
| 15 | `DDoS-UDP_Fragmentation` | 0.9996 | 0.9989 | 0.9992 | 7,224 |
| 16 | `DNS_Spoofing` | 0.7105 | 0.7733 | 0.7406 | 4,570 |
| 17 | `DictionaryBruteForce` | 0.3615 | 0.6176 | 0.4560 | 319 |
| 18 | `DoS-HTTP_Flood` | 0.9939 | 0.9928 | 0.9933 | 1,805 |
| 19 | `DoS-SYN_Flood` | 0.9995 | 0.9990 | 0.9992 | 51,116 |
| 20 | `DoS-TCP_Flood` | 0.9989 | 0.9998 | 0.9993 | 67,697 |
| 21 | `DoS-UDP_Flood` | 0.9994 | 0.9996 | 0.9995 | 83,627 |
| 22 | `MITM-ArpSpoofing` | 0.8763 | 0.8311 | 0.8531 | 7,840 |
| 23 | `Mirai-greeth_flood` | 0.9988 | 0.9986 | 0.9987 | 24,934 |
| 24 | `Mirai-greip_flood` | 0.9981 | 0.9985 | 0.9983 | 19,279 |
| 25 | `Mirai-udpplain` | 1.0000 | 0.9997 | 0.9998 | 22,536 |
| 26 | `Recon-HostDiscovery` | 0.8398 | 0.8355 | 0.8376 | 3,331 |
| 27 | `Recon-OSScan` | 0.5264 | 0.6317 | 0.5743 | 2,433 |
| 28 | `Recon-PingSweep` | 0.1125 | 0.6981 | 0.1937 | 53 |
| 29 | `Recon-PortScan` | 0.5518 | 0.6960 | 0.6155 | 2,082 |
| 30 | `SqlInjection` | 0.2955 | 0.6149 | 0.3991 | 148 |
| 31 | `Uploading_Attack` | 0.0478 | 0.4848 | 0.0870 | 33 |
| 32 | `VulnerabilityScan` | 0.9901 | 0.9858 | 0.9879 | 913 |
| 33 | `XSS` | 0.1661 | 0.4466 | 0.2421 | 103 |

---

## 7. Confusion Matrix & Misclassification Breakdown

The test confusion matrix was plotted and exported to:
- `results/confusion_matrices/xgboost_confusion_matrix.png`
- `results/metrics/xgboost_misclassifications.csv` (8,891 total test errors, down from 10,494 in Random Forest).

### Top 10 Confusion Pairs:
1. `BenignTraffic` $\rightarrow$ `Recon-OSScan` (865 flows)
2. `BenignTraffic` $\rightarrow$ `DNS_Spoofing` (863 flows)
3. `BenignTraffic` $\rightarrow$ `MITM-ArpSpoofing` (569 flows)
4. `BenignTraffic` $\rightarrow$ `Recon-PortScan` (463 flows)
5. `MITM-ArpSpoofing` $\rightarrow$ `BenignTraffic` (387 flows)
6. `MITM-ArpSpoofing` $\rightarrow$ `DNS_Spoofing` (354 flows)
7. `Recon-OSScan` $\rightarrow$ `Recon-PortScan` (334 flows)
8. `Recon-OSScan` $\rightarrow$ `BenignTraffic` (317 flows)
9. `Recon-PortScan` $\rightarrow$ `Recon-OSScan` (243 flows)
10. `DNS_Spoofing` $\rightarrow$ `MITM-ArpSpoofing` (229 flows)

---

## 8. Feature Importance Analysis (Average Gain)

In XGBoost, **Gain** measures the average improvement in log-loss brought by a feature across all tree splits.
- Data Export: `results/metrics/xgboost_feature_importance.csv`
- Plot Export: `results/graphs/xgboost_feature_importance.png`

### Top 15 Most Discriminative Features by Gain:

| Rank | Feature Name | Average Gain | Weight (Frequency) | Network Security Significance |
|:---:|:---|:---:|:---:|:---|
| 1 | `fin_flag_number` | 1960.96 | 119 | Critical indicator of FIN-flood teardown anomalies. |
| 2 | `syn_flag_number` | 1777.32 | 446 | Essential for isolating TCP SYN floods and half-open scans. |
| 3 | `ICMP` | 1483.60 | 246 | Protocol filter for Ping sweep and ICMP flood attacks. |
| 4 | `UDP` | 1090.37 | 329 | Primary discriminator for UDP flood and Mirai botnet UDP traffic. |
| 5 | `psh_flag_number` | 723.98 | 280 | Detects PSH+ACK flood activity and rapid data delivery. |
| 6 | `rst_flag_number` | 703.23 | 278 | Identifies abnormal TCP connection termination. |
| 7 | `SSH` | 699.16 | 88 | Distinguishes dictionary brute force on port 22. |
| 8 | `LLC` | 568.58 | 6 | Logical Link Control layer anomalies. |
| 9 | `IAT` | 567.51 | 12,079 | Most frequently evaluated feature for inter-packet arrival timings. |
| 10 | `HTTP` | 541.16 | 554 | Isolates HTTP DoS/DDoS and web application injection flows. |
| 11 | `Magnitue` | 533.05 | 1,590 | Geometric magnitude of packet vector dynamics. |
| 12 | `Protocol Type` | 424.06 | 3,404 | Transport protocol numerical representation. |
| 13 | `ARP` | 372.70 | 63 | Discriminator for MITM ARP spoofing attacks. |
| 14 | `fin_count` | 359.15 | 1,339 | Cumulative FIN flag count in the flow window. |
| 15 | `Weight` | 335.64 | 1,045 | Flow statistical weighting coefficient. |

---

## 9. Direct Measured Comparison: Random Forest vs. XGBoost

Below is the exact empirical comparison between the two classical baselines evaluated on identical 1,176,851 test flows:

| Metric | Random Forest (Phase 6) | XGBoost (Phase 7) | Absolute Difference ($\Delta$) | Observations |
|:---|:---:|:---:|:---:|:---|
| **Test Accuracy** | 0.9911 (99.11%) | **0.9924 (99.24%)** | **+0.13%** (+1,603 correct flows) | XGBoost achieves higher overall classification accuracy. |
| **Weighted Precision** | 0.9927 | **0.9939** | **+0.0012** | Higher confidence in predicted positive labels. |
| **Weighted Recall** | 0.9911 | **0.9924** | **+0.0013** | Higher fraction of attack flows detected. |
| **Weighted F1-Score** | 0.9917 | **0.9930** | **+0.0013** | Superior aggregate F1-score across all 34 classes. |
| **Macro Precision** | 0.7542 | **0.7701** | **+0.0159** | Reduced false alarms on low-frequency classes. |
| **Macro Recall** | 0.8456 | **0.8597** | **+0.0141** | Higher detection rate on minority attack vectors. |
| **Macro F1-Score** | 0.7872 | **0.7928** | **+0.0056** | Improved unweighted balance across all 34 classes. |
| **Training Time** | **120.24 s** | 158.73 s | +38.49 s | Random Forest parallel tree fitting was slightly faster. |
| **Test Inference Time** | 13.2685 s | **12.3201 s** | **-0.9484 s** (7.1% faster) | XGBoost decision trees execute faster during inference. |
| **Model Disk Size** | 1,568.01 MB | **7.36 MB** | **-1,560.65 MB (99.53% reduction)** | **Key finding:** XGBoost is 213x more memory-efficient. |
| **Total Test Errors** | 10,494 | **8,891** | **-1,603 errors (15.28% fewer errors)** | Significant error reduction across test partitions. |

---

## 10. Limitations of the XGBoost Baseline

1. **Tabular Feature Dependency:** Like Random Forest, XGBoost operates purely on scalar flow statistics and cannot inspect raw payload byte sequences for zero-day obfuscated web injection vectors.
2. **Lack of Temporal Memory:** XGBoost does not model long-range sequential state transitions across consecutive multi-flow attack stages (e.g. initial reconnaissance followed by lateral movement). This motivates the transition to Deep Learning models (1D-CNN, BiLSTM, and Hybrid CNN-BiLSTM in Phases 8–10).

---

## 11. Summary of Generated Artifacts

- **Model Binary:** `models/xgboost.pkl` (7.36 MB)
- **Model Metadata:** `models/xgboost_metadata.json`
- **Validation Metrics:** `results/metrics/xgboost_validation.json`
- **Test Metrics:** `results/metrics/xgboost_test.json`
- **Per-Class Report:** `results/metrics/xgboost_classification_report.csv`
- **Feature Importance Data:** `results/metrics/xgboost_feature_importance.csv`
- **Feature Importance Plot:** `results/graphs/xgboost_feature_importance.png`
- **Confusion Matrix Plot:** `results/confusion_matrices/xgboost_confusion_matrix.png`
- **Misclassifications:** `results/metrics/xgboost_misclassifications.csv`
- **Updated Comparison Table:** `results/metrics/model_comparison.csv`
- **Training Script:** `scripts/train_xgboost.py`
- **Interactive Notebook:** `notebooks/05_xgboost_baseline.ipynb`
- **Test Suite:** `tests/test_xgboost.py` (7 tests, all passing)
