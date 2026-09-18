# Model Error Analysis Report: Random Forest Baseline (Phase 12)

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Model:** Random Forest Classifier (`n_estimators=200`, `max_depth=25`, `max_samples=0.1`, `class_weight='balanced'`)  
**Evaluation Partition:** CICIoT2023 Processed Test Split (`data/processed/test/test.npz`)  
**Evaluated Samples:** 1,176,851 flows (34 classes)  

---

## 1. Quantitative Performance Overview
- **Total Test Samples:** 1,176,851
- **Correct Predictions:** 1,166,357 (99.11%)
- **Total Misclassifications:** 10,494 (0.89%)
- **Macro Precision:** 0.7542
- **Macro Recall:** 0.8456
- **Macro F1-Score:** 0.7872
- **Weighted F1-Score:** 0.9917

---

## 2. Confusion Matrix Observations
- **Raw & Normalized Matrix Files:**
  - Raw Heatmap: [`results/confusion_matrices/random_forest_confusion_matrix.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/random_forest_confusion_matrix.png)
  - Normalized Heatmap: [`results/confusion_matrices/random_forest_confusion_matrix_normalized.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/random_forest_confusion_matrix_normalized.png)
- **Diagonal Concentration:** Strong diagonal dominance across massive volumetric classes (`DDoS-ICMP_Flood`, `DoS-UDP_Flood`, `DDoS-TCP_Flood`, `DDoS-SYN_Flood`).

---

## 3. Strongly Classified Classes (Recall $\ge 98\%$)
The Random Forest model excels at identifying high-rate, structured protocol floods:
1. `DDoS-ICMP_Flood`: Recall = 100.0%, Precision = 99.98% (302,298 / 302,301 correct)
2. `DoS-UDP_Flood`: Recall = 99.99%, Precision = 99.78% (127,103 / 127,112 correct)
3. `DDoS-UDP_Flood`: Recall = 99.82%, Precision = 99.95% (216,420 / 216,810 correct)
4. `DDoS-TCP_Flood`: Recall = 99.96%, Precision = 99.98% (94,545 / 94,582 correct)
5. `DDoS-SYN_Flood`: Recall = 99.96%, Precision = 99.95% (81,667 / 81,698 correct)
6. `Mirai-greeth_flood`: Recall = 99.98%, Precision = 99.98% (42,763 / 42,771 correct)

---

## 4. Top Misclassification Confusion Pairs

| Rank | Actual Class | Predicted Class | Error Count | Percentage of RF Errors |
| :---: | :--- | :--- | :---: | :---: |
| **1** | `BenignTraffic` | `Recon-OSScan` | 912 | 8.69% |
| **2** | `BenignTraffic` | `DNS_Spoofing` | 832 | 7.93% |
| **3** | `MITM-ArpSpoofing` | `BenignTraffic` | 486 | 4.63% |
| **4** | `BenignTraffic` | `Recon-PortScan` | 481 | 4.58% |
| **5** | `MITM-ArpSpoofing` | `DNS_Spoofing` | 320 | 3.05% |
| **6** | `DDoS-UDP_Flood` | `DoS-UDP_Flood` | 277 | 2.64% |
| **7** | `BenignTraffic` | `DictionaryBruteForce` | 239 | 2.28% |
| **8** | `BenignTraffic` | `MITM-ArpSpoofing` | 219 | 2.09% |
| **9** | `Recon-OSScan` | `BenignTraffic` | 217 | 2.07% |
| **10** | `Recon-OSScan` | `Recon-PortScan` | 189 | 1.80% |

---

## 5. False Positive & False Negative Analysis

### Benign vs. Attack Breakdown:
- **Benign Support:** 27,709 flows
- **Benign Correct:** 24,276 flows (87.61% accuracy)
- **Benign $\rightarrow$ Attack (False Positives):** 3,433 flows (**12.39% FP Rate**)
- **Attack Support:** 1,149,142 flows
- **Attack Correct:** 1,142,081 flows (99.39% accuracy)
- **Attack $\rightarrow$ Benign (False Negatives):** 1,146 flows (**0.10% FN Rate**)
- **Attack-to-Attack Confusion:** 5,915 flows

#### Observations:
- **Low Operational False Negative Rate:** Only 0.10% of malicious flows escape detection as benign.
- **Moderate False Positive Rate:** 12.39% of normal benign flows are flagged as low-rate reconnaissance or spoofing due to shared flow duration and packet size profiles.

---

## 6. Data-Driven Interpretation
1. **Stealth Scan Inseparability:** OS scanning and port scanning send sparse probe packets that mimic standard network handshakes (e.g., occasional SYN without full sessions), causing overlap with legitimate background traffic.
2. **ARP Spoofing Subtlety:** ARP cache poisoning does not significantly inflate traffic volume or packet rates, making it difficult for tree splits based purely on aggregated flow statistics to separate from normal local subnet communications.

---

## 7. Model Limitations
- **Memory Footprint:** Random Forest requires **1,568 MB** of memory on disk and in RAM to store 200 deep decision trees.
- **Benign Precision Trade-off:** While overall accuracy is 99.11%, macro precision is pulled down to 75.42% due to false alarms on ultra-minority attack classes.
