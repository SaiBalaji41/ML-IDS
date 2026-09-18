# Error Analysis Results

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Data Source:** [`results/metrics/model_error_comparison.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/model_error_comparison.csv), [`results/metrics/research_error_analysis.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/research_error_analysis.csv)  
**Evaluated Split:** CICIoT2023 Held-Out Test Partition ($N = 1,176,851$)

---

## 1. Quantitative Error Breakdown

Table 1 summarizes the total misclassifications, error rates, and macro diagnostic metrics across the five candidate models evaluated on the held-out test set.

### Table 1: Model Error Comparison on Held-Out Test Partition

| Model | Total Test Samples | Correct Predictions | Total Misclassifications | Error Rate (%) | Macro Precision | Macro Recall | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 1,176,851 | 1,166,357 | 10,494 | **0.8917%** | 0.7542 | 0.8456 | 0.7872 |
| **XGBoost** | 1,176,851 | 1,167,960 | 8,891 | **0.7555%** | 0.7701 | 0.8597 | 0.7928 |
| **1D-CNN** | 1,176,851 | 778,078 | 398,773 | **33.8847%** | 0.4312 | 0.4697 | 0.4030 |
| **BiLSTM** | 1,176,851 | 935,264 | 241,587 | **20.5283%** | 0.4851 | 0.5423 | 0.4846 |
| **CNN + BiLSTM** | 1,176,851 | 930,517 | 246,334 | **20.9316%** | 0.5079 | 0.5459 | 0.4962 |

---

## 2. False-Positive and False-Negative Analysis

In the operational security context of Network Intrusion Detection:
- **False Positives (Benign Classified as Attack):** Induces alert fatigue for security analysts and threatens availability if automated blocking rules are triggered.
- **False Negatives (Attack Classified as Benign):** Represents severe security exposure where active intrusion bypasses detection mechanisms.

### Table 2: Binary Security Error Distribution (Benign vs. Attack)

| Model | Benign Test Samples | False Positives (Benign $\rightarrow$ Attack) | FP Rate on Benign (%) | Attack Test Samples | False Negatives (Attack $\rightarrow$ Benign) | FN Rate on Attacks (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 27,709 | 3,433 | 12.39% | 1,149,142 | 398 | **0.0346%** |
| **XGBoost** | 27,709 | 3,545 | 12.79% | 1,149,142 | 322 | **0.0280%** |
| **1D-CNN** | 27,709 | 16,029 | 57.85% | 1,149,142 | 4,812 | 0.4187% |
| **BiLSTM** | 27,709 | 8,784 | 31.70% | 1,149,142 | 3,115 | 0.2711% |
| **CNN + BiLSTM** | 27,709 | 14,630 | 52.80% | 1,149,142 | 3,890 | 0.3385% |

---

## 3. Recurring Error Modes and Interpretations

### 3.1 Flood Symmetry Confusion
The most frequent error across 1D-CNN, BiLSTM, and CNN + BiLSTM occurred between `DDoS-UDP_Flood` and `DoS-UDP_Flood` (accounting for over 130,000 errors per deep learning model).
- *Observed evidence:* Both attack categories utilize identical UDP packet headers, static packet lengths, and maximum packet transmission rates.
- *Possible explanation:* Without explicit source IP cardinality or multi-flow temporal graphs, tabular feature vectors for a single UDP flood flow do not contain distinct statistical variance between single-source and distributed originations.

### 3.2 Spoofing and Layer-2 / Layer-7 Ambiguity
For tree-based models (Random Forest and XGBoost), the leading confusion pairs were `MITM-ArpSpoofing` $\leftrightarrow$ `DNS_Spoofing` (~1,500 errors).
- *Possible explanation:* Both attack simulations in CICIoT2023 were conducted in local subnet environments where ARP poisoning was utilized to facilitate DNS redirection, creating correlated flow characteristics.

### 3.3 Extreme Minority Sample Collapse
For attacks with $N \le 150$ samples (`Backdoor_Malware`, `BrowserHijacking`, `CommandInjection`, `SqlInjection`, `Uploading_Attack`, `XSS`):
- Deep neural networks predicted 0 true positives (100% false negative rate on these specific classes).
- Tree ensembles retained recall between 36.36% and 68.07%.
- *Possible explanation:* In high-dimensional spaces optimized via mini-batch stochastic gradient descent, the loss contribution of extremely rare classes (0.001% of batches) is negligible compared to massive gradient signals from millions of flood samples, causing gradient updates to collapse into majority class priors.
