# Model Error Analysis Report: Proposed Hybrid CNN + BiLSTM (Phase 12)

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Model:** Proposed Hybrid 1D-CNN + BiLSTM (`Conv1D(64)` $\rightarrow$ `MaxPool(2)` $\rightarrow$ `BiLSTM(64)` $\rightarrow$ `Dense(64)` $\rightarrow$ `Softmax(34)`)  
**Evaluation Partition:** CICIoT2023 Processed Test Split (`data/processed/test/test.npz`)  
**Evaluated Samples:** 1,176,851 flows (34 classes)  

---

## 1. Quantitative Performance Overview
- **Total Test Samples:** 1,176,851
- **Correct Predictions:** 930,517 (79.07%)
- **Total Misclassifications:** 246,334 (**20.93% Error Rate**)
- **Macro Precision:** 0.5079
- **Macro Recall:** 0.5459
- **Macro F1-Score:** 0.4962
- **Weighted F1-Score:** 0.7780
- **Inference Latency:** 0.1338 ms / sample

---

## 2. Confusion Matrix Observations
- **Raw & Normalized Matrix Files:**
  - Raw Heatmap: [`results/confusion_matrices/cnn_bilstm_confusion_matrix.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/cnn_bilstm_confusion_matrix.png)
  - Normalized Heatmap: [`results/confusion_matrices/cnn_bilstm_confusion_matrix_normalized.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/cnn_bilstm_confusion_matrix_normalized.png)
- **High Macro Precision on Deep Classes:** CNN + BiLSTM achieves the highest Macro Precision ($0.5079$) among all three neural architectures (compared to $0.4312$ for 1D-CNN and $0.4851$ for BiLSTM).

---

## 3. Strongly Classified Classes (Recall $\ge 95\%$)
1. `DDoS-ICMP_Flood`: Recall = 99.98%, Precision = 99.99% (302,238 / 302,301 correct)
2. `Mirai-greeth_flood`: Recall = 99.96%
3. `Mirai-udpplain`: Recall = 99.93%
4. `DDoS-SlowLoris`: Recall = 98.44%
5. `DDoS-UDP_Flood`: Recall = 85.13%

---

## 4. Top Misclassification Confusion Pairs

| Rank | Actual Class | Predicted Class | Error Count | Percentage of Hybrid Errors |
| :---: | :--- | :--- | :---: | :---: |
| **1** | `DoS-TCP_Flood` | `DDoS-TCP_Flood` | 60,751 | **24.66%** |
| **2** | `DDoS-UDP_Flood` | `DoS-UDP_Flood` | 32,239 | **13.09%** |
| **3** | `DDoS-SynonymousIP_Flood` | `DDoS-SYN_Flood` | 26,236 | **10.65%** |
| **4** | `DoS-UDP_Flood` | `DDoS-UDP_Flood` | 25,655 | **10.41%** |
| **5** | `DoS-SYN_Flood` | `DDoS-SYN_Flood` | 24,483 | **9.94%** |
| **6** | `DDoS-SYN_Flood` | `DoS-SYN_Flood` | 13,019 | 5.29% |
| **7** | `Mirai-greeth_flood` | `Mirai-greip_flood` | 12,059 | 4.90% |
| **8** | `BenignTraffic` | `DNS_Spoofing` | 3,927 | 1.59% |
| **9** | `DDoS-SynonymousIP_Flood` | `DoS-SYN_Flood` | 3,845 | 1.56% |
| **10** | `BenignTraffic` | `Recon-OSScan` | 2,752 | 1.12% |

*Key Finding:* **74.04%** of hybrid CNN-BiLSTM errors are concentrated across DoS $\leftrightarrow$ DDoS variant confusion.

---

## 5. False Positive & False Negative Analysis

### Benign vs. Attack Breakdown:
- **Benign Support:** 27,709 flows
- **Benign Correct:** 17,507 flows (63.18% accuracy)
- **Benign $\rightarrow$ Attack (False Positives):** 10,202 flows (**36.82% FP Rate**)
- **Attack Support:** 1,149,142 flows
- **Attack Correct:** 913,010 flows (79.45% accuracy)
- **Attack $\rightarrow$ Benign (False Negatives):** 3,118 flows (**0.27% FN Rate**)
- **Attack-to-Attack Confusion:** **233,014 flows** (94.59% of errors)

---

## 6. Data-Driven Interpretation
- **Hybrid Spatial-Sequential Representation:** The initial Conv1D layer extracts local feature combinations across adjacent dimensions, while the subsequent BiLSTM layer aggregates contextual dependencies.
- **Improved Inference Efficiency:** By pooling the sequence length by half via `MaxPooling1D`, the BiLSTM operates on 23 time steps instead of 46, reducing inference latency from 0.3316 ms (standalone BiLSTM) down to **0.1338 ms** ($2.5\times$ speedup).

---

## 7. Model Limitations
- **Subclass Invariance:** Like standalone BiLSTM, the hybrid model struggles to differentiate volumetric single-source DoS from distributed DDoS without IP address cardinality features.
- **Ensemble Gap:** While leading the deep learning models in Macro Precision, it remains outperformed in overall accuracy and speed by gradient-boosted decision trees (XGBoost: 99.24%).
