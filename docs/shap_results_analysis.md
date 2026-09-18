# SHAP Explainability Results Analysis

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Data Sources:** [`results/shap/`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/), [`results/shap/research_shap_summary.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/research_shap_summary.csv)  
**Evaluated Scope:** Model Attribution Across Tree Ensembles and Deep Learning Architectures

---

## 1. Interpretability Objective and Attribution Framework

In compliance with trustworthy AI principles for cybersecurity operations, SHAP (SHapley Additive exPlanations) was utilized to quantify feature contributions toward model prediction logits.

> [!IMPORTANT]
> SHAP provides mathematical model attribution indicating how input features influence the decision boundaries of a specific trained model. SHAP attributions **do not establish physical causality** in network traffic. Statements below describe model behavior, not deterministic causality.

---

## 2. Top Contributing Features Across Models

Table 1 lists the top 5 contributing features for each model based on mean absolute SHAP value ($E[|\phi_i|]$).

### Table 1: Top 5 Contributing Features by Model Architecture

| Model Architecture | Rank 1 Feature | Rank 2 Feature | Rank 3 Feature | Rank 4 Feature | Rank 5 Feature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | `IAT` (0.01219) | `Magnitue` (0.00601) | `syn_count` (0.00559) | `Protocol Type` (0.00542) | `Header_Length` (0.00528) |
| **XGBoost** | `IAT` (1.25477) | `Protocol Type` (0.34218) | `Header_Length` (0.23384) | `Tot size` (0.22308) | `syn_flag_number` (0.20046) |
| **1D-CNN** | `HTTPS` (0.10807) | `syn_flag_number` (0.09567) | `ICMP` (0.09238) | `syn_count` (0.08918) | `Tot size` (0.08831) |
| **BiLSTM** | `flow_duration` (0.69160) | `Header_Length` (0.58421) | `Tot size` (0.42454) | `Std` (0.40802) | `IAT` (0.40657) |
| **CNN + BiLSTM** | `Srate` (0.07529) | `Drate` (0.07479) | `Rate` (0.06607) | `ece_flag_number` (0.04723) | `Magnitue` (0.03839) |

---

## 3. Global Feature Consensus Analysis

Across the different modeling paradigms, several feature families consistently demonstrated high attribution magnitude:

1. **Temporal and Inter-Arrival Dynamics (`IAT`, `flow_duration`, `Rate`, `Srate`):**
   - In both XGBoost and Random Forest, `IAT` (Inter-Arrival Time) emerged as the single highest-attribution feature.
   - For BiLSTM, `flow_duration` and `IAT` ranked among the top 5 attributions.
   - *Observation:* Sudden shifts in inter-arrival variance strongly differentiate bursty flood traffic (DDoS/DoS) from intermittent benign communications.

2. **Protocol and Transport Header Attributes (`Protocol Type`, `Header_Length`, `Tot size`):**
   - `Header_Length` and `Protocol Type` ranked in the top 5 for Random Forest, XGBoost, and BiLSTM.
   - *Observation:* Distinguishes encapsulated transport protocols (ICMP, UDP, TCP) and fixed-header packet floods.

3. **TCP Control Flags (`syn_flag_number`, `syn_count`, `rst_count`, `urg_count`):**
   - High attribution was observed for TCP handshake flags in models classifying SYN flood, ACK fragmentation, and Port Scanning attacks.

---

## 4. Local Instance Explanations

Local SHAP waterfall and force analyses on representative test traffic flows illustrated the following decision drivers:

1. **Benign Traffic Flow:**
   - Predictions of `BenignTraffic` were driven by normal-range `IAT`, standard `Header_Length` values, and the absence of abnormal TCP flag ratios (`syn_flag_number = 0`, `urg_count = 0`).
2. **`DDoS-ICMP_Flood` Attack Flow:**
   - Strong positive attribution was driven by `Protocol Type = 1` (ICMP), extremely low `IAT` (high frequency), and static packet payload magnitude.
3. **`Mirai-greeth_flood` Flow:**
   - Driven by GRE encapsulation protocol indicators and elevated packet transmission rates (`Srate`, `Rate`).
4. **`Recon-PortScan` Flow:**
   - Driven by rapid successive SYN packets with zero payload bytes and alternating destination port parameters.

---

## 5. SHAP Analysis of Representative Misclassifications

SHAP attribution for misclassified instances revealed that:
- For `DDoS-UDP_Flood` samples misclassified as `DoS-UDP_Flood` by 1D-CNN and BiLSTM, the SHAP value profiles for `Protocol Type`, `Header_Length`, and `Tot size` were virtually identical, with no single feature providing sufficient diverging attribution to shift the prediction across the class boundary.
- For `MITM-ArpSpoofing` misclassified as `DNS_Spoofing` by XGBoost, local attribution showed that shared DNS query rate features dominated the tree split thresholds, overshadowing lower-layer packet counters.
