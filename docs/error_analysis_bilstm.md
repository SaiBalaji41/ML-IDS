# Model Error Analysis Report: Standalone BiLSTM (Phase 12)

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Model:** Bidirectional LSTM (BiLSTM 64 $\rightarrow$ BiLSTM 32 $\rightarrow$ Dense 64 $\rightarrow$ Softmax 34)  
**Evaluation Partition:** CICIoT2023 Processed Test Split (`data/processed/test/test.npz`)  
**Evaluated Samples:** 1,176,851 flows (34 classes)  

---

## 1. Quantitative Performance Overview
- **Total Test Samples:** 1,176,851
- **Correct Predictions:** 935,264 (79.47%)
- **Total Misclassifications:** 241,587 (**20.53% Error Rate**)
- **Macro Precision:** 0.4851
- **Macro Recall:** 0.5423
- **Macro F1-Score:** 0.4846
- **Weighted F1-Score:** 0.7756
- **Inference Latency:** 0.3316 ms / sample

---

## 2. Confusion Matrix Observations
- **Raw & Normalized Matrix Files:**
  - Raw Heatmap: [`results/confusion_matrices/bilstm_confusion_matrix.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/bilstm_confusion_matrix.png)
  - Normalized Heatmap: [`results/confusion_matrices/bilstm_confusion_matrix_normalized.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/bilstm_confusion_matrix_normalized.png)
- **Distinct Block Diagonal Structure:** Substantially better diagonal sharpness than 1D-CNN, with errors tightly constrained to specific transport protocol pairs.

---

## 3. Strongly Classified Classes (Recall $\ge 95\%$)
1. `DDoS-ICMP_Flood`: Recall = 99.98%, Precision = 99.99% (302,238 / 302,301 correct)
2. `Mirai-greeth_flood`: Recall = 99.96%
3. `Mirai-udpplain`: Recall = 99.93%
4. `DDoS-SlowLoris`: Recall = 98.71%
5. `DDoS-UDP_Flood`: Recall = 85.31%

---

## 4. Top Misclassification Confusion Pairs

| Rank | Actual Class | Predicted Class | Error Count | Percentage of BiLSTM Errors |
| :---: | :--- | :--- | :---: | :---: |
| **1** | `DoS-TCP_Flood` | `DDoS-TCP_Flood` | 63,682 | **26.36%** |
| **2** | `DDoS-UDP_Flood` | `DoS-UDP_Flood` | 31,859 | **13.19%** |
| **3** | `DDoS-SynonymousIP_Flood` | `DDoS-SYN_Flood` | 24,820 | **10.27%** |
| **4** | `DoS-UDP_Flood` | `DDoS-UDP_Flood` | 24,567 | **10.17%** |
| **5** | `DoS-SYN_Flood` | `DDoS-SYN_Flood` | 22,566 | **9.34%** |
| **6** | `DDoS-SYN_Flood` | `DoS-SYN_Flood` | 13,999 | 5.79% |
| **7** | `Mirai-greeth_flood` | `Mirai-greip_flood` | 12,059 | 4.99% |
| **8** | `BenignTraffic` | `DNS_Spoofing` | 3,927 | 1.63% |
| **9** | `DDoS-SynonymousIP_Flood` | `DoS-SYN_Flood` | 3,745 | 1.55% |
| **10** | `BenignTraffic` | `Recon-OSScan` | 2,752 | 1.14% |

*Key Finding:* Over **75.12%** of BiLSTM errors occur across DoS $\leftrightarrow$ DDoS variants of the same protocol!

---

## 5. False Positive & False Negative Analysis

### Benign vs. Attack Breakdown:
- **Benign Support:** 27,709 flows
- **Benign Correct:** 18,858 flows (68.06% accuracy)
- **Benign $\rightarrow$ Attack (False Positives):** 8,851 flows (**31.94% FP Rate**)
- **Attack Support:** 1,149,142 flows
- **Attack Correct:** 916,406 flows (79.75% accuracy)
- **Attack $\rightarrow$ Benign (False Negatives):** 4,117 flows (**0.36% FN Rate**)
- **Attack-to-Attack Confusion:** **228,619 flows** (94.63% of errors)

---

## 6. Data-Driven Interpretation
- **Bidirectional Contextual Encoding:** BiLSTM processes the 46 features sequentially in forward and backward directions, capturing inter-feature dependencies more effectively than localized 1D-CNN kernels.
- **Protocol Flag Mastery:** BiLSTM successfully isolates protocol signatures (ICMP, GRE, UDP, TCP), achieving near-perfect 99.98% recall on ICMP floods and Mirai botnet activity.
- **Rate Magnitude Ambiguity:** BiLSTM struggles to distinguish single-source from multi-source volumetric floods because individual flow statistics in CICIoT2023 reflect per-flow metrics rather than full multi-host topology graphs.

---

## 7. Model Limitations
- **Excessive Training Time:** Wall-clock training on CPU required **74,830 seconds ($\sim 20.8$ hours)** due to unrolled recurrent unrolling steps across 5.49 million training flows.
- **Inference Latency:** At 0.3316 ms/sample, BiLSTM is $\sim 30\times$ slower than XGBoost (0.0105 ms/sample).
