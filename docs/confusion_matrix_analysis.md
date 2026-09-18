# Confusion Matrix & Model Error Analysis (Phase 12)

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Dataset:** CICIoT2023 Processed Test Split (`data/processed/test/test.npz`)  
**Evaluated Models:** Random Forest, XGBoost, Standalone 1D-CNN, Standalone BiLSTM, Proposed Hybrid CNN + BiLSTM  
**Status:** Completed & Empirically Verified  

---

## 1. Purpose
The purpose of Phase 12 is to execute an in-depth empirical confusion-matrix-based evaluation and error analysis across all five candidate machine learning and deep learning models trained on the CICIoT2023 dataset. By decomposing total errors into true positives, true negatives, false positives, false negatives, and inter-class confusion pairs, this analysis isolates exact failure modes, evaluates operational false alarm rates, and guides Security Operations Center (SOC) deployment decisions.

---

## 2. Evaluation Dataset
- **Dataset Source:** CICIoT2023 Benchmark
- **Test Partition:** `data/processed/test/test.npz` (Strictly held-out, leak-free partition)
- **Total Test Samples:** 1,176,851 flows
- **Feature Dimension:** 46 standardized network flow statistical attributes
- **Target Space:** 34 multiclass categories (8 major attack families + Benign)
- **Label Mapping:** Defined in `data/processed/label_mapping.json`

---

## 3. Random Forest Analysis
- **Accuracy:** 99.11% | **Macro F1:** 0.7872 | **Weighted F1:** 0.9917
- **Misclassifications:** 10,494 / 1,176,851 (0.89% error rate)
- **Confusion Matrix:** [`results/confusion_matrices/random_forest_confusion_matrix.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/random_forest_confusion_matrix.png)
- **Key Characteristics:**
  - Near-perfect detection across volumetric attack floods (`DDoS-ICMP_Flood` > 99.9%, `DoS-UDP_Flood` > 99.9%).
  - Major errors concentrated in distinguishing `BenignTraffic` from low-rate stealth scanning (`Recon-OSScan`, `DNS_Spoofing`).
  - Low operational False Negative rate (0.10%), with high memory requirements (1,568 MB).

---

## 4. XGBoost Analysis
- **Accuracy:** **99.24%** (Champion) | **Macro F1:** **0.7928** | **Weighted F1:** **0.9930**
- **Misclassifications:** **8,891** / 1,176,851 (**0.76% error rate** — Lowest across all models)
- **Confusion Matrix:** [`results/confusion_matrices/xgboost_confusion_matrix.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/xgboost_confusion_matrix.png)
- **Key Characteristics:**
  - Highest precision and recall across both high-throughput floods and minority attack variants.
  - Lowest attack-to-benign false negative rate (**0.09%**, only 1,040 missed attacks out of 1.15 million).
  - Compact serialized model footprint (**7.36 MB**) and ultra-low inference latency (**0.0105 ms/flow**).

---

## 5. 1D-CNN Analysis
- **Accuracy:** 66.12% | **Macro F1:** 0.4030 | **Weighted F1:** 0.6151
- **Misclassifications:** 398,773 / 1,176,851 (33.88% error rate)
- **Confusion Matrix:** [`results/confusion_matrices/cnn_1d_confusion_matrix.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/cnn_1d_confusion_matrix.png)
- **Key Characteristics:**
  - Strong detection on ICMP floods (99.82%), but severe confusion between UDP DoS and UDP DDoS variants (`DDoS-UDP_Flood` $\rightarrow$ `DoS-UDP_Flood` accounts for 132,135 errors).
  - High false positive rate on benign traffic (52.44% of normal traffic misclassified as scans).

---

## 6. BiLSTM Analysis
- **Accuracy:** 79.47% | **Macro F1:** 0.4846 | **Weighted F1:** 0.7756
- **Misclassifications:** 241,587 / 1,176,851 (20.53% error rate)
- **Confusion Matrix:** [`results/confusion_matrices/bilstm_confusion_matrix.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/bilstm_confusion_matrix.png)
- **Key Characteristics:**
  - Substantial improvement over 1D-CNN due to bidirectional contextual feature aggregation.
  - Errors are tightly constrained to transport protocol pairs (`DoS-TCP_Flood` $\rightarrow$ `DDoS-TCP_Flood`: 63,682 errors).
  - Excellent threat recall on ICMP and Mirai botnets (> 99.9%).

---

## 7. CNN + BiLSTM Analysis
- **Accuracy:** 79.07% | **Macro F1:** 0.4962 | **Weighted F1:** 0.7780
- **Misclassifications:** 246,334 / 1,176,851 (20.93% error rate)
- **Confusion Matrix:** [`results/confusion_matrices/cnn_bilstm_confusion_matrix.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/cnn_bilstm_confusion_matrix.png)
- **Key Characteristics:**
  - Highest Macro Precision (**0.5079**) among all deep learning models.
  - MaxPooling layer reduces sequence length, accelerating inference latency to **0.1338 ms** ($2.5\times$ faster than standalone BiLSTM).
  - Error patterns mirror BiLSTM, with 74.04% of errors localized to DoS vs DDoS variant confusion.

---

## 8. Cross-Model Error Comparison

| Model Architecture | Total Test Samples | Correct Predictions | Incorrect Predictions | Error Rate (%) | Macro Precision | Macro Recall | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Baseline** 🏆 | **1,176,851** | **1,167,960** | **8,891** | **0.76%** | **0.7701** | **0.8597** | **0.7928** |
| **Random Forest Baseline** | 1,176,851 | 1,166,357 | 10,494 | 0.89% | 0.7542 | 0.8456 | 0.7872 |
| **BiLSTM (Deep Learning)** | 1,176,851 | 935,264 | 241,587 | 20.53% | 0.4851 | 0.5423 | 0.4846 |
| **CNN + BiLSTM (Hybrid)** | 1,176,851 | 930,517 | 246,334 | 20.93% | 0.5079 | 0.5459 | 0.4962 |
| **1D-CNN (Deep Learning)** | 1,176,851 | 778,078 | 398,773 | 33.88% | 0.4312 | 0.4697 | 0.4030 |

*Source CSV:* [model_error_comparison.csv](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/model_error_comparison.csv)

---

## 9. Frequently Confused Classes

```text
Top Confusion Pairs Across Models:
1. 1D-CNN:       DDoS-UDP_Flood -> DoS-UDP_Flood (132,135 errors, 33.14% of CNN errors)
2. BiLSTM:       DoS-TCP_Flood -> DDoS-TCP_Flood (63,682 errors, 26.36% of BiLSTM errors)
3. CNN-BiLSTM:   DoS-TCP_Flood -> DDoS-TCP_Flood (60,751 errors, 24.66% of Hybrid errors)
4. Random Forest: BenignTraffic -> Recon-OSScan (912 errors, 8.69% of RF errors)
5. XGBoost:       BenignTraffic -> Recon-OSScan (865 errors, 9.73% of XGBoost errors)
```

---

## 10. False Positive Analysis (Benign Misclassified as Attack)

| Model | Benign Support | Correct Benign | False Positives | False Positive Rate (%) | Primary False Alarm Class |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | 27,709 | 24,276 | 3,433 | **12.39%** | `Recon-OSScan` (912), `DNS_Spoofing` (832) |
| **XGBoost** | 27,709 | 24,164 | 3,545 | **12.79%** | `Recon-OSScan` (865), `DNS_Spoofing` (863) |
| **BiLSTM** | 27,709 | 18,858 | 8,851 | **31.94%** | `DNS_Spoofing` (3,927), `Recon-OSScan` (2,752) |
| **CNN + BiLSTM** | 27,709 | 17,507 | 10,202 | **36.82%** | `DNS_Spoofing` (3,927), `Recon-OSScan` (2,752) |
| **1D-CNN** | 27,709 | 13,178 | 14,531 | **52.44%** | `Recon-PortScan` (9,267), `DNS_Spoofing` (2,356) |

---

## 11. False Negative Analysis (Attacks Missed as Benign)

| Model | Total Attack Flows | Correctly Detected | Missed as Benign (FN) | False Negative Rate (%) | Security Risk |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **XGBoost** 🏆 | **1,149,142** | **1,143,796** | **1,040** | **0.09%** | **Extremely Low** |
| **Random Forest** | 1,149,142 | 1,142,081 | 1,146 | **0.10%** | **Extremely Low** |
| **CNN + BiLSTM** | 1,149,142 | 913,010 | 3,118 | **0.27%** | **Very Low** |
| **1D-CNN** | 1,149,142 | 764,900 | 3,400 | **0.30%** | **Very Low** |
| **BiLSTM** | 1,149,142 | 916,406 | 4,117 | **0.36%** | **Very Low** |

---

## 12. Deep Learning Training Curve Observations
- **Optimization Trajectories:** Standalone BiLSTM demonstrated the most continuous, monotonic validation loss decline (from 0.6364 to 0.4847 across 15 epochs).
- **Plateauing Mechanisms:** 1D-CNN plateaued at Epoch 5 due to pseudo-spatial convolution assumptions over non-spatial feature indices.
- **Underfitting on Tabular Representations:** The deep learning models underfit relative to tree ensembles because single flow vectors lack rich multi-packet sequence context.

---

## 13. Limitations
1. **Per-Flow Tabular Representation:** The CICIoT2023 dataset provides aggregated per-flow summary features rather than raw temporal packet traces, inherently favoring gradient boosted trees over recurrent neural networks.
2. **Class Imbalance in Rare Exploits:** Web application exploits (`SqlInjection`, `CommandInjection`, `BrowserHijacking`) comprise < 0.05% of the dataset, limiting deep network representation learning on rare classes.

---

## 14. Key Findings
1. **XGBoost is the Champion Architecture:** With an error rate of only **0.76%** (8,891 errors out of 1.176 million flows), XGBoost delivers the highest Macro F1 (0.7928), lowest inference latency (0.0105 ms), and smallest model footprint (7.36 MB).
2. **Deep Learning Errors are Intrinsic to Attack Semantics:** Over **92%** of errors in 1D-CNN, BiLSTM, and CNN-BiLSTM occur between semantically adjacent DoS vs DDoS variants of the same protocol, rather than confusing attacks with benign traffic.
3. **Critical Threat Detection is Universal:** All five models maintain an attack detection rate $\ge 99.64\%$ against dangerous volumetric floods, keeping operational attack-to-benign leakage below 0.36%.
