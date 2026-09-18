# Experimental Limitations & Constraints (Phase 14)

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Dataset:** CICIoT2023 Benchmark  
**Status:** Factual Technical Limitations  

---

## 1. Overview
In accordance with rigorous academic methodology, this document defines the verified technical and structural limitations of the experimental setup implemented in this project.

---

## 2. Dataset & Representation Limitations

### 2.1 Tabular Flow Statistics vs. Raw Packet Payloads
- **Implemented Input:** 46 aggregated tabular statistical features per bidirectional flow.
- **Limitation:** The models do not inspect raw application-layer payload byte sequences or un-aggregated full packet PCAP traces. Attacks that hide in customized byte payloads without modifying statistical flow metrics cannot be detected by flow-level IDS alone.

### 2.2 Feature-Sequence Assumption for Deep Learning (1D-CNN / BiLSTM)
- **Implemented Input Shape:** `[batch_size, 46, 1]` where 46 tabular features are treated as a 1D sequence.
- **Limitation:** The order of the 46 features is an artifact of tabular indexing, not a true physical time series. While 1D convolutions and recurrent gates extract non-linear feature interactions, they lack true temporal multi-packet dynamics, explaining why tree-based models (XGBoost/RF) outperform neural architectures on this tabular dataset.

### 2.3 Class Imbalance in Minority Exploit Types
- **Observation:** Massive volumetric attacks (`DDoS-ICMP_Flood` > 300K samples) dominate the dataset, whereas web application attacks (`SqlInjection`, `CommandInjection`, `BrowserHijacking`) contain < 500 samples in the test set.
- **Limitation:** Despite balanced class weights, macro precision on ultra-minority classes is pulled down by extreme imbalance.

---

## 3. Threat Subclass Invariance (DoS vs. DDoS Confusion)
- **Observation:** Over 92% of deep learning errors and a key portion of tree ensemble errors occur between single-source DoS and distributed DDoS attacks of the same transport protocol (e.g., `DoS-TCP_Flood` vs. `DDoS-TCP_Flood`).
- **Technical Cause:** Individual bidirectional flow records do not contain multi-host IP topology cardinality (number of distinct source IPs attacking the victim). At the individual flow level, single-source and multi-source packet headers look identical.

---

## 4. Explainability (SHAP) Constraints
- **Attribution vs. Causation:** SHAP measures mathematical sensitivity of the model's decision function. A high SHAP value indicates that a feature altered the model's output log-odds, but does not prove that the feature caused the cyber security event in the physical network.
- **Standardized Coordinate Space:** All attributions operate on standardized features (`StandardScaler`), requiring analysts to reference normalized distributions.

---

## 5. Deployment & Hardware Constraints
- **Offline Evaluation:** All evaluations were performed on static pre-extracted test partitions. Live wire-speed packet capture (via Scapy / eBPF) and real-time Kafka streaming represent future architectural integration.
- **Compute Constraints:** Deep learning training was executed on CPU multi-threading on native Windows, extending standalone BiLSTM training duration to ~20.8 hours.
