# Class-Wise Performance Results Analysis

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Data Source:** [`results/metrics/classwise_model_comparison.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/classwise_model_comparison.csv)  
**Evaluated Split:** CICIoT2023 Held-Out Test Partition ($N = 1,176,851$)

---

## 1. Overview of Multi-Class Evaluation

The CICIoT2023 dataset presents 34 heterogeneous traffic classes with support ranging from 180,447 test samples (`DDoS-ICMP_Flood`) down to 33 test samples (`Uploading_Attack`). Performance varied substantially across individual attack vectors.

---

## 2. Model-Specific Class-Wise Observations

### 2.1 Random Forest
- **High-Recall Classes ($R > 0.99$):** `DDoS-ICMP_Flood` (0.9988), `DDoS-PSHACK_Flood` (0.9991), `DDoS-RSTFINFlood` (0.9992), `DDoS-TCP_Flood` (0.9984), `DDoS-UDP_Flood` (0.9978), `DoS-HTTP_Flood` (0.9961), `DoS-UDP_Flood` (0.9971), `Mirai-udpplain` (0.9972), `VulnerabilityScan` (0.9989).
- **Lower-Recall Classes ($R < 0.60$):** `Backdoor_Malware` (0.5843), `BrowserHijacking` (0.5075), `Recon-PingSweep` (0.2830), `SqlInjection` (0.5068), `Uploading_Attack` (0.3636), `XSS` (0.5049).
- **High-Precision Classes ($P > 0.99$):** `DDoS-ICMP_Flood` (1.0000), `DDoS-PSHACK_Flood` (1.0000), `DDoS-RSTFINFlood` (1.0000), `DDoS-TCP_Flood` (0.9999), `DoS-TCP_Flood` (0.9997), `Mirai-greeth_flood` (0.9994).
- **Lower-Precision Classes ($P < 0.40$):** `Backdoor_Malware` (0.3312), `BrowserHijacking` (0.2042), `CommandInjection` (0.3202), `DictionaryBruteForce` (0.2500), `Recon-PingSweep` (0.2000), `SqlInjection` (0.3769), `Uploading_Attack` (0.3000), `XSS` (0.3095).
- **Lower F1-Score Classes ($F_1 < 0.40$):** `BrowserHijacking` (0.2912), `DictionaryBruteForce` (0.3605), `Recon-PingSweep` (0.2344), `Uploading_Attack` (0.3288), `XSS` (0.3838).

### 2.2 XGBoost
- **High-Recall Classes ($R > 0.99$):** `DDoS-ACK_Fragmentation` (0.9995), `DDoS-ICMP_Flood` (0.9999), `DDoS-ICMP_Fragmentation` (0.9994), `DDoS-PSHACK_Flood` (1.0000), `DDoS-RSTFINFlood` (0.9999), `DDoS-TCP_Flood` (0.9989), `DDoS-UDP_Flood` (0.9986), `DoS-UDP_Flood` (0.9983), `Mirai-greeth_flood` (0.9984), `Mirai-udpplain` (0.9989).
- **Lower-Recall Classes ($R < 0.60$):** `Backdoor_Malware` (0.5955), `CommandInjection` (0.5294), `Recon-PingSweep` (0.3019), `SqlInjection` (0.4797), `Uploading_Attack` (0.3939), `XSS` (0.4854).
- **High-Precision Classes ($P > 0.99$):** `DDoS-ICMP_Flood` (1.0000), `DDoS-PSHACK_Flood` (1.0000), `DDoS-RSTFINFlood` (1.0000), `DDoS-TCP_Flood` (0.9999), `DoS-SYN_Flood` (0.9992), `Mirai-greip_flood` (0.9992).
- **Lower-Precision Classes ($P < 0.40$):** `Backdoor_Malware` (0.2181), `BrowserHijacking` (0.3132), `CommandInjection` (0.2692), `DictionaryBruteForce` (0.2959), `Recon-PingSweep` (0.2319), `SqlInjection` (0.3859), `Uploading_Attack` (0.3514), `XSS` (0.3378).
- **Lower F1-Score Classes ($F_1 < 0.40$):** `Backdoor_Malware` (0.3193), `CommandInjection` (0.3569), `DictionaryBruteForce` (0.3946), `Recon-PingSweep` (0.2623), `Uploading_Attack` (0.3714), `XSS` (0.3984).

### 2.3 1D-CNN
- **High-Recall Classes ($R > 0.90$):** `DDoS-ICMP_Flood` (0.9859), `DDoS-RSTFINFlood` (0.9950), `DDoS-SYN_Flood` (0.9701), `Mirai-udpplain` (0.9912).
- **Lower-Recall Classes ($R < 0.10$):** `Backdoor_Malware` (0.0000), `BrowserHijacking` (0.0000), `CommandInjection` (0.0000), `DDoS-SynonymousIP_Flood` (0.0000), `DDoS-UDP_Flood` (0.0000), `DictionaryBruteForce` (0.0000), `DoS-SYN_Flood` (0.0019), `Mirai-greeth_flood` (0.0000), `Recon-PingSweep` (0.0000), `SqlInjection` (0.0000), `Uploading_Attack` (0.0000), `XSS` (0.0000).
- **Classes with Zero Precision and Recall ($F_1 = 0.0000$):** 12 classes achieved an F1-score of 0.0000 due to output collapse on minority and overlapping flood signatures.

### 2.4 BiLSTM
- **High-Recall Classes ($R > 0.90$):** `DDoS-ICMP_Flood` (0.9881), `DDoS-RSTFINFlood` (0.9972), `DDoS-SYN_Flood` (0.9664), `DoS-UDP_Flood` (0.9942), `Mirai-greip_flood` (0.9712), `Mirai-udpplain` (0.9935), `VulnerabilityScan` (0.9168).
- **Lower-Recall Classes ($R < 0.10$):** `Backdoor_Malware` (0.0000), `BrowserHijacking` (0.0000), `CommandInjection` (0.0000), `DDoS-SynonymousIP_Flood` (0.0002), `DDoS-UDP_Flood` (0.0008), `DictionaryBruteForce` (0.0157), `DoS-SYN_Flood` (0.0284), `Mirai-greeth_flood` (0.0112), `Recon-PingSweep` (0.0943), `SqlInjection` (0.0000), `Uploading_Attack` (0.0000), `XSS` (0.0000).
- **Zero Recall Classes:** 7 minority classes recorded zero true positives.

### 2.5 CNN + BiLSTM
- **High-Recall Classes ($R > 0.90$):** `DDoS-ICMP_Flood` (0.9695), `DDoS-RSTFINFlood` (0.9968), `DDoS-SYN_Flood` (0.9616), `DDoS-UDP_Fragmentation` (0.9819), `DoS-UDP_Flood` (0.9939), `Mirai-greip_flood` (0.9686), `Mirai-udpplain` (0.9928).
- **Lower-Recall Classes ($R < 0.10$):** `Backdoor_Malware` (0.0000), `BrowserHijacking` (0.0000), `CommandInjection` (0.0000), `DDoS-SynonymousIP_Flood` (0.0002), `DDoS-UDP_Flood` (0.0007), `DictionaryBruteForce` (0.0251), `DoS-SYN_Flood` (0.0343), `Mirai-greeth_flood` (0.0146), `Recon-PingSweep` (0.0000), `Recon-PortScan` (0.0735), `SqlInjection` (0.0000), `Uploading_Attack` (0.0000), `XSS` (0.0000).

---

## 3. Comparative Observations Across Classes

### 3.1 Flood and Volumetric Attacks
High-volume volumetric attacks (`DDoS-ICMP_Flood`, `DDoS-RSTFINFlood`, `Mirai-udpplain`) yielded F1-scores exceeding 0.98 across all five models.

### 3.2 Web Application and Infiltration Attacks
Web-based attacks (`SqlInjection`, `CommandInjection`, `BrowserHijacking`, `XSS`, `Uploading_Attack`) exhibited marked divergence between architectures:
- Tree ensembles achieved F1-scores in the range of 0.2912 to 0.4355.
- Deep learning models obtained F1-scores of 0.0000 on these classes.
- *Possible explanation:* The extremely small sample counts ($N \le 148$ per class) and the tabular nature of aggregated statistical flow features may have limited the gradient descent optimization of deep neural networks, while decision tree split criteria could isolate discrete rule thresholds for small sample partitions.

### 3.3 Benign Traffic Classification
- **Random Forest:** Precision = 0.9549, Recall = 0.8761, F1-Score = 0.9138 (Support = 27,709).
- **XGBoost:** Precision = 0.9587, Recall = 0.8721, F1-Score = 0.9133 (Support = 27,709).
- **BiLSTM:** Precision = 0.7241, Recall = 0.6830, F1-Score = 0.7029 (Support = 27,709).
- **CNN + BiLSTM:** Precision = 0.6832, Recall = 0.4720, F1-Score = 0.5583 (Support = 27,709).
- **1D-CNN:** Precision = 0.6124, Recall = 0.4215, F1-Score = 0.4992 (Support = 27,709).
