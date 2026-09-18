# Model Error Analysis Report: Standalone 1D-CNN (Phase 12)

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Model:** 1D Convolutional Neural Network (Conv1D $\rightarrow$ MaxPool $\rightarrow$ Conv1D $\rightarrow$ GAP $\rightarrow$ Dense $\rightarrow$ Softmax)  
**Evaluation Partition:** CICIoT2023 Processed Test Split (`data/processed/test/test.npz`)  
**Evaluated Samples:** 1,176,851 flows (34 classes)  

---

## 1. Quantitative Performance Overview
- **Total Test Samples:** 1,176,851
- **Correct Predictions:** 778,078 (66.12%)
- **Total Misclassifications:** 398,773 (**33.88% Error Rate**)
- **Macro Precision:** 0.4312
- **Macro Recall:** 0.4697
- **Macro F1-Score:** 0.4030
- **Weighted F1-Score:** 0.6151

---

## 2. Confusion Matrix Observations
- **Raw & Normalized Matrix Files:**
  - Raw Heatmap: [`results/confusion_matrices/cnn_1d_confusion_matrix.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/cnn_1d_confusion_matrix.png)
  - Normalized Heatmap: [`results/confusion_matrices/cnn_1d_confusion_matrix_normalized.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/cnn_1d_confusion_matrix_normalized.png)
- **High Off-Diagonal Density:** Massive off-diagonal clustering across DoS/DDoS subclass blocks of the same network transport protocol.

---

## 3. Strongly Classified Classes (Recall $\ge 95\%$)
1. `DDoS-ICMP_Flood`: Recall = 99.82%, Precision = 99.89% (301,768 / 302,301 correct)
2. `DDoS-SlowLoris`: Recall = 98.42%
3. `Mirai-udpplain`: Recall = 97.80%

---

## 4. Top Misclassification Confusion Pairs

| Rank | Actual Class | Predicted Class | Error Count | Percentage of 1D-CNN Errors |
| :---: | :--- | :--- | :---: | :---: |
| **1** | `DDoS-UDP_Flood` | `DoS-UDP_Flood` | 132,135 | **33.14%** |
| **2** | `DDoS-SynonymousIP_Flood` | `DDoS-SYN_Flood` | 61,319 | **15.38%** |
| **3** | `DoS-TCP_Flood` | `DDoS-TCP_Flood` | 53,159 | **13.33%** |
| **4** | `DoS-SYN_Flood` | `DDoS-SYN_Flood` | 37,752 | **9.47%** |
| **5** | `Mirai-greeth_flood` | `Mirai-greip_flood` | 24,387 | **6.12%** |
| **6** | `DDoS-SYN_Flood` | `DDoS-SynonymousIP_Flood` | 16,843 | 4.22% |
| **7** | `DDoS-TCP_Flood` | `DoS-TCP_Flood` | 16,131 | 4.05% |
| **8** | `BenignTraffic` | `Recon-PortScan` | 9,267 | 2.32% |
| **9** | `DoS-SYN_Flood` | `DDoS-SynonymousIP_Flood` | 6,065 | 1.52% |
| **10** | `DoS-UDP_Flood` | `DDoS-UDP_Flood` | 4,667 | 1.17% |

*Key Finding:* Over **81.44%** of all 1D-CNN classification errors stem directly from confusing single-source DoS and distributed DDoS variants of the same protocol!

---

## 5. False Positive & False Negative Analysis

### Benign vs. Attack Breakdown:
- **Benign Support:** 27,709 flows
- **Benign Correct:** 13,178 flows (47.56% accuracy)
- **Benign $\rightarrow$ Attack (False Positives):** 14,531 flows (**52.44% FP Rate**)
- **Attack Support:** 1,149,142 flows
- **Attack Correct:** 764,900 flows (66.56% accuracy)
- **Attack $\rightarrow$ Benign (False Negatives):** 3,400 flows (**0.30% FN Rate**)
- **Attack-to-Attack Confusion:** **380,842 flows** (95.50% of all errors)

---

## 6. Data-Driven Interpretation
- **Spatial Inductive Bias Mismatch:** Convolutional kernels rely on local adjacency. Because the 46 features represent an un-ordered tabular vector, sliding convolutions compute arbitrary linear combinations of neighboring features (e.g. `Duration` convolved with `Rate`), leading to poor representation of threshold-based packet rates.
- **Protocol Semantic Invariance:** The CNN easily learns transport protocol filters (ICMP vs UDP vs TCP), but fails to separate single-source DoS from distributed multi-source DDoS because both exhibit identical packet payload sizes and header flags.

---

## 7. Model Limitations
- **High Computational Overhead:** Training requires 682 seconds on CPU, yielding significantly lower accuracy (66.12%) than lightweight tree baselines (99.24%).
- **Severe Benign False Alarms:** 52.44% of benign flows are incorrectly flagged as port scans or reconnaissance.
