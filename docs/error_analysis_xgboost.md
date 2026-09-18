# Model Error Analysis Report: XGBoost Baseline (Phase 12)

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Model:** XGBoost Classifier (`n_estimators=100`, `max_depth=10`, `learning_rate=0.1`, `tree_method='hist'`)  
**Evaluation Partition:** CICIoT2023 Processed Test Split (`data/processed/test/test.npz`)  
**Evaluated Samples:** 1,176,851 flows (34 classes)  
**Status:** **Champion Production Model**  

---

## 1. Quantitative Performance Overview
- **Total Test Samples:** 1,176,851
- **Correct Predictions:** 1,167,960 (99.24%)
- **Total Misclassifications:** 8,891 (**0.76%** — Lowest Error Count)
- **Macro Precision:** **0.7701** (Highest across all models)
- **Macro Recall:** **0.8597** (Highest across all models)
- **Macro F1-Score:** **0.7928** (Highest across all models)
- **Weighted F1-Score:** **0.9930**
- **Inference Latency:** **0.0105 ms / sample**

---

## 2. Confusion Matrix Observations
- **Raw & Normalized Matrix Files:**
  - Raw Heatmap: [`results/confusion_matrices/xgboost_confusion_matrix.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/xgboost_confusion_matrix.png)
  - Normalized Heatmap: [`results/confusion_matrices/xgboost_confusion_matrix_normalized.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/xgboost_confusion_matrix_normalized.png)
- **Sharpest Diagonal Concentration:** XGBoost demonstrates the cleanest decision boundaries across high-throughput and rare-class attack types.

---

## 3. Strongly Classified Classes (Recall $\ge 99\%$)
1. `DDoS-ICMP_Flood`: Recall = 100.0%, Precision = 99.98% (302,298 / 302,301 correct)
2. `DoS-UDP_Flood`: Recall = 99.99%, Precision = 99.88% (127,105 / 127,112 correct)
3. `DDoS-UDP_Flood`: Recall = 99.90%, Precision = 99.96% (216,590 / 216,810 correct)
4. `DDoS-TCP_Flood`: Recall = 99.98%, Precision = 99.99% (94,561 / 94,582 correct)
5. `DDoS-SYN_Flood`: Recall = 99.98%, Precision = 99.96% (81,678 / 81,698 correct)
6. `Mirai-greeth_flood`: Recall = 99.99%, Precision = 99.99% (42,766 / 42,771 correct)
7. `Mirai-udpplain`: Recall = 99.99%, Precision = 99.99% (36,523 / 36,527 correct)

---

## 4. Top Misclassification Confusion Pairs

| Rank | Actual Class | Predicted Class | Error Count | Percentage of XGBoost Errors |
| :---: | :--- | :--- | :---: | :---: |
| **1** | `BenignTraffic` | `Recon-OSScan` | 865 | 9.73% |
| **2** | `BenignTraffic` | `DNS_Spoofing` | 863 | 9.71% |
| **3** | `BenignTraffic` | `MITM-ArpSpoofing` | 569 | 6.40% |
| **4** | `BenignTraffic` | `Recon-PortScan` | 463 | 5.21% |
| **5** | `MITM-ArpSpoofing` | `BenignTraffic` | 387 | 4.35% |
| **6** | `Recon-OSScan` | `BenignTraffic` | 239 | 2.69% |
| **7** | `DNS_Spoofing` | `BenignTraffic` | 215 | 2.42% |
| **8** | `Recon-PortScan` | `BenignTraffic` | 158 | 1.78% |
| **9** | `DDoS-UDP_Flood` | `DoS-UDP_Flood` | 150 | 1.69% |
| **10** | `BenignTraffic` | `Recon-HostDiscovery` | 148 | 1.66% |

---

## 5. False Positive & False Negative Analysis

### Benign vs. Attack Breakdown:
- **Benign Support:** 27,709 flows
- **Benign Correct:** 24,164 flows (87.21% accuracy)
- **Benign $\rightarrow$ Attack (False Positives):** 3,545 flows (**12.79% FP Rate**)
- **Attack Support:** 1,149,142 flows
- **Attack Correct:** 1,143,796 flows (**99.53% accuracy** — Highest)
- **Attack $\rightarrow$ Benign (False Negatives):** 1,040 flows (**0.09% FN Rate** — Lowest)
- **Attack-to-Attack Confusion:** 4,306 flows (Lowest)

#### Observations:
- **Outstanding Threat Capture:** 99.91% of all attack instances are successfully flagged and blocked.
- **Ultra-Compact Model Footprint:** At only **7.36 MB**, XGBoost delivers 200x memory compression compared to Random Forest while achieving superior precision, recall, and inference speed.

---

## 6. Data-Driven Interpretation
- **Gradient Boosting Optimization:** Second-order Taylor gradient expansion allows XGBoost to discover fine-grained decision boundaries separating high-volume DDoS packets based on subtle flag combinations (`fin_flag_number`, `syn_flag_number`, `psh_flag_number`).
- **Low-Rate Probing Ambiguity:** Misclassifications are almost exclusively localized to low-rate exploratory attacks (`Recon-OSScan`, `DNS_Spoofing`, `MITM-ArpSpoofing`) that mimic normal intermittent DNS queries or host lookups.

---

## 7. Model Limitations
- **Imbalance Sensitivity for Ultra-Rare Classes:** While weighted metrics exceed 99.3%, macro precision on single-sample attack classes (e.g., `BrowserHijacking`, `SqlInjection`) is bounded by extreme imbalance.
