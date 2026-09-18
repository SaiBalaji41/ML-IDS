# Confusion Matrix Results Analysis

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Data Sources:** [`results/confusion_matrices/`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/), [`results/metrics/misclassification_pairs.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/misclassification_pairs.csv)  
**Evaluated Split:** CICIoT2023 Held-Out Test Partition ($N = 1,176,851$)

---

## 1. Overview and Matrix Structural Verification

The confusion matrix for each candidate model is a $34 \times 34$ integer matrix $C \in \mathbb{N}^{34 \times 34}$, where element $C_{i, j}$ represents the count of test flows whose true ground-truth label was class $i$ and which the model predicted as class $j$.

- **Sum Constraint:** $\sum_{i=1}^{34}\sum_{j=1}^{34} C_{i, j} = 1,176,851$ (100% of test partition).
- **Diagonal Sum (Trace):** $\text{Tr}(C) = \sum_{k=1}^{34} C_{k, k} = N_{\text{correct}}$.
- **Off-Diagonal Sum:** $\sum_{i \ne j} C_{i, j} = N_{\text{errors}}$.

---

## 2. Model-by-Model Confusion Matrix Diagnostics

### 2.1 Random Forest
- **Total Correct Classifications:** 1,166,357 (99.11% of test set).
- **Total Misclassifications:** 10,494 (0.89% of test set).
- **Benign $\rightarrow$ Attack Errors:** 3,433 benign flows misclassified as attack classes (primarily `MITM-ArpSpoofing` and `DNS_Spoofing`).
- **Attack $\rightarrow$ Benign Errors:** 398 attack flows misclassified as benign traffic.
- **Dominant Attack $\rightarrow$ Attack Confusion Pairs:**
  1. `MITM-ArpSpoofing` $\rightarrow$ `DNS_Spoofing`: 1,481 errors.
  2. `DNS_Spoofing` $\rightarrow$ `MITM-ArpSpoofing`: 897 errors.
  3. `Recon-OSScan` $\rightarrow$ `Recon-HostDiscovery`: 625 errors.
  4. `DDoS-SynonymousIP_Flood` $\rightarrow$ `DDoS-SYN_Flood`: 127 errors.

### 2.2 XGBoost
- **Total Correct Classifications:** 1,167,960 (99.24% of test set).
- **Total Misclassifications:** 8,891 (0.76% of test set).
- **Benign $\rightarrow$ Attack Errors:** 3,545 benign flows misclassified as attacks.
- **Attack $\rightarrow$ Benign Errors:** 322 attack flows misclassified as benign traffic.
- **Dominant Attack $\rightarrow$ Attack Confusion Pairs:**
  1. `MITM-ArpSpoofing` $\rightarrow$ `DNS_Spoofing`: 1,492 errors.
  2. `DNS_Spoofing` $\rightarrow$ `MITM-ArpSpoofing`: 721 errors.
  3. `Recon-OSScan` $\rightarrow$ `Recon-HostDiscovery`: 412 errors.
  4. `DDoS-SynonymousIP_Flood` $\rightarrow$ `DDoS-SYN_Flood`: 118 errors.

### 2.3 1D-CNN
- **Total Correct Classifications:** 778,078 (66.12% of test set).
- **Total Misclassifications:** 398,773 (33.88% of test set).
- **Benign $\rightarrow$ Attack Errors:** 16,029 benign flows misclassified as attacks (including 9,267 misclassified as `Recon-PortScan`).
- **Attack $\rightarrow$ Benign Errors:** 4,812 attack flows misclassified as benign traffic.
- **Dominant Attack $\rightarrow$ Attack Confusion Pairs:**
  1. `DDoS-UDP_Flood` $\rightarrow$ `DoS-UDP_Flood`: 132,135 errors (33.14% of all 1D-CNN errors).
  2. `DDoS-SynonymousIP_Flood` $\rightarrow$ `DDoS-SYN_Flood`: 61,319 errors (15.38% of all 1D-CNN errors).
  3. `DoS-TCP_Flood` $\rightarrow$ `DDoS-TCP_Flood`: 53,159 errors (13.33% of all 1D-CNN errors).
  4. `DoS-SYN_Flood` $\rightarrow$ `DDoS-SYN_Flood`: 37,752 errors (9.47% of all 1D-CNN errors).
  5. `Mirai-greeth_flood` $\rightarrow$ `Mirai-greip_flood`: 24,387 errors (6.12% of all 1D-CNN errors).

### 2.4 BiLSTM
- **Total Correct Classifications:** 935,264 (79.47% of test set).
- **Total Misclassifications:** 241,587 (20.53% of test set).
- **Benign $\rightarrow$ Attack Errors:** 8,784 benign flows misclassified as attacks.
- **Attack $\rightarrow$ Benign Errors:** 3,115 attack flows misclassified as benign traffic.
- **Dominant Attack $\rightarrow$ Attack Confusion Pairs:**
  1. `DDoS-UDP_Flood` $\rightarrow$ `DoS-UDP_Flood`: 134,812 errors.
  2. `DDoS-SynonymousIP_Flood` $\rightarrow$ `DDoS-SYN_Flood`: 88,214 errors.
  3. `Mirai-greeth_flood` $\rightarrow$ `Mirai-greip_flood`: 23,892 errors.
  4. `DoS-SYN_Flood` $\rightarrow$ `DDoS-SYN_Flood`: 48,110 errors.

### 2.5 CNN + BiLSTM
- **Total Correct Classifications:** 930,517 (79.07% of test set).
- **Total Misclassifications:** 246,334 (20.93% of test set).
- **Benign $\rightarrow$ Attack Errors:** 14,630 benign flows misclassified as attacks.
- **Attack $\rightarrow$ Benign Errors:** 3,890 attack flows misclassified as benign traffic.
- **Dominant Attack $\rightarrow$ Attack Confusion Pairs:**
  1. `DDoS-UDP_Flood` $\rightarrow$ `DoS-UDP_Flood`: 135,110 errors.
  2. `DDoS-SynonymousIP_Flood` $\rightarrow$ `DDoS-SYN_Flood`: 89,450 errors.
  3. `Mirai-greeth_flood` $\rightarrow$ `Mirai-greip_flood`: 24,115 errors.
  4. `DoS-SYN_Flood` $\rightarrow$ `DDoS-SYN_Flood`: 48,920 errors.

---

## 3. Structural Confusion Patterns

### 3.1 Single-Source (DoS) vs. Distributed (DDoS) Flood Symmetries
Across deep learning architectures (1D-CNN, BiLSTM, CNN + BiLSTM), the largest source of misclassification stems from confusion between single-source DoS and distributed DDoS attacks sharing identical protocol flags and packet sizes:
- `DDoS-UDP_Flood` $\leftrightarrow$ `DoS-UDP_Flood`
- `DDoS-SYN_Flood` $\leftrightarrow$ `DoS-SYN_Flood`
- `DDoS-TCP_Flood` $\leftrightarrow$ `DoS-TCP_Flood`

*Possible Explanation:* In tabular aggregated flow records, per-packet statistical attributes (e.g., header length, protocol type, flags) are nearly identical between single-source and distributed floods; distributed patterns are typically distinguished by distinct source IP diversity or inter-arrival entropy across aggregated time windows, which individual flow records may not fully capture.

### 3.2 Protocol Tunneling Confusion
`Mirai-greeth_flood` and `Mirai-greip_flood` exhibited consistent mutual confusion across neural models due to overlapping GRE encapsulation header characteristics.
