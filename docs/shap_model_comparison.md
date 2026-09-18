# Multi-Model SHAP Feature Attribution Comparison (Phase 13)

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Dataset:** CICIoT2023 Benchmark (46 Tabular Flow Features)  
**Evaluated Architectures:** Random Forest, XGBoost, 1D-CNN, Standalone BiLSTM, Proposed Hybrid CNN + BiLSTM  

---

## 1. Cross-Model Feature Attribution Summary

| Feature Category | Features Present | Models Prioritizing This Category | Cyber Security Relevance |
| :--- | :--- | :--- | :--- |
| **Packet Timing & Inter-Arrival** | `IAT`, `flow_duration`, `Duration`, `Weight` | **XGBoost, Random Forest, BiLSTM** | Captures volumetric flood packet bursting vs. human-paced browsing |
| **Protocol & Header Structure** | `Protocol Type`, `Header_Length`, `IPv`, `LLC` | **XGBoost, Random Forest, 1D-CNN** | Separates ICMP, UDP, TCP, and GRE protocol traffic channels |
| **TCP State & Flag Numbers** | `syn_flag_number`, `syn_count`, `rst_count`, `urg_count`, `fin_flag_number` | **XGBoost, Random Forest, 1D-CNN** | Detects SYN flooding, stealth port scans, and teardown anomalies |
| **Traffic Rates & Throughput** | `Rate`, `Srate`, `Drate` | **CNN + BiLSTM, BiLSTM** | Measures asymmetric volumetric flooding rates |
| **Packet Size Distributions** | `Tot size`, `Magnitue`, `AVG`, `Std`, `Variance`, `Number` | **Random Forest, BiLSTM, 1D-CNN** | Differentiates small probe packets from large payload floods |

---

## 2. Shared Attributions Across Architectures
Across all five models, three features consistently appear within the top 10 global rankings:
1. **`IAT` (Inter-Arrival Time):** Evaluated as the single most discriminative feature in both XGBoost and Random Forest, and #5 in BiLSTM. Low IAT with high packet counts strongly indicates automated flood scripts.
2. **`Protocol Type` / `Header_Length`:** Fundamental layer-3/4 structural indicators that distinguish lightweight ICMP/UDP floods from full TCP three-way handshake flows.
3. **`syn_flag_number` / `syn_count`:** Dominant across tree models and 1D-CNN for detecting TCP connection exhaustion attacks (`DDoS-SYN_Flood`, `DoS-SYN_Flood`).

---

## 3. Architecture-Specific Differences

```text
Decision Trees (XGBoost / Random Forest)
├── Focus on exact threshold cuts on temporal dynamics (IAT) and discrete protocol flags.
└── Superior at isolating non-linear interactions between flag numbers and packet rates.

Recurrent Neural Networks (Standalone BiLSTM)
├── Attributions are dominated by continuous macro-level flow aggregates (flow_duration, Tot size, Header_Length).
└── Accumulates context sequentially across the 46 features.

Hybrid CNN + BiLSTM
├── Convolutional pooling compresses features, causing the network to rely on directional rate ratios (Srate, Drate, Rate).
└── Directional throughput asymmetries become the primary signal for threat classification.
```

---

## 4. Class-Specific Attribution Contrasts
- **Volumetric Floods (`DDoS-ICMP_Flood`, `DoS-UDP_Flood`):**
  - Trees assign ~80% of attribution to `ICMP`, `UDP`, and `Header_Length`.
  - Neural models also highlight packet length statistics (`Tot size`, `AVG`).
- **Stealth Scans (`Recon-PortScan`, `Recon-OSScan`):**
  - Attributions shift toward `syn_flag_number = 1`, `ack_flag_number = 0`, and small `flow_duration`.
- **Benign Traffic (`BenignTraffic`):**
  - Attributions reflect presence of `HTTPS = 1`, balanced `Srate`/`Drate`, and moderate `IAT` variance.

---

## 5. Research Summary Table (Top 5 Features by Architecture)

| Rank | Random Forest | XGBoost (Champion) | 1D-CNN | BiLSTM | CNN + BiLSTM |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | `IAT` (0.0122) | `IAT` (1.2548) | `HTTPS` (0.1081) | `flow_duration` (0.6916) | `Srate` (0.0753) |
| **2** | `Magnitue` (0.0060) | `Protocol Type` (0.3422) | `syn_flag_number` (0.0957) | `Header_Length` (0.5842) | `Drate` (0.0748) |
| **3** | `syn_count` (0.0056) | `Header_Length` (0.2338) | `ICMP` (0.0924) | `Tot size` (0.4245) | `Rate` (0.0661) |
| **4** | `Protocol Type` (0.0054) | `Tot size` (0.2231) | `syn_count` (0.0892) | `Std` (0.4080) | `ece_flag_number` (0.0472) |
| **5** | `Header_Length` (0.0053) | `syn_flag_number` (0.2005) | `Tot size` (0.0883) | `IAT` (0.4066) | `Magnitue` (0.0384) |

*Source CSV:* [research_shap_summary.csv](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/research_shap_summary.csv)
