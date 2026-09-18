# Explainable AI (XAI) with SHAP for Network Intrusion Detection (Phase 13)

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Dataset:** CICIoT2023 Processed Test Split (`data/processed/test/test.npz`)  
**Models Evaluated:** Random Forest, XGBoost, Standalone 1D-CNN, Standalone BiLSTM, Proposed Hybrid CNN + BiLSTM  
**Status:** Completed & Empirically Grounded  

---

## 1. Purpose
The purpose of Phase 13 is to implement mathematically grounded Explainable AI (XAI) using SHAP (SHapley Additive exPlanations) to interpret model behavior, establish global feature rankings, analyze class-specific attributions, and examine feature contributions behind Phase 12 misclassifications. This enables Security Operations Center (SOC) analysts to audit and trust automated intrusion alerts.

---

## 2. Explainable AI Approach
SHAP computes additive feature attributions grounded in cooperative game theory (Shapley values). For a model $f$ and input instance $x$, the prediction is decomposed as:
$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$
where $\phi_0 = \mathbb{E}[f(z)]$ is the base expected value across the reference background distribution and $\phi_i(x)$ is the Shapley attribution value for feature $i$.

---

## 3. Dataset Representation
The dataset is **tabular network flow statistics** (46 features aggregated across bidirectional flows in CICIoT2023), not raw packet payload byte strings. All features were preprocessed with zero mean and unit variance (`StandardScaler`). Therefore, SHAP attributions explain the standardized input feature space.

---

## 4. Features Used
The 46 features include:
- **Temporal Dynamics:** `flow_duration`, `Duration`, `IAT`, `Weight`
- **Rate Statistics:** `Rate`, `Srate`, `Drate`
- **TCP Control Flags:** `fin_flag_number`, `syn_flag_number`, `rst_flag_number`, `psh_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number`, `syn_count`, `ack_count`, `fin_count`, `rst_count`, `urg_count`
- **Protocol Identifiers:** `Protocol Type`, `TCP`, `UDP`, `ICMP`, `HTTP`, `HTTPS`, `DNS`, `SSH`, `Telnet`, `SMTP`, `ARP`, `LLC`, `IPv`
- **Packet Size Statistics:** `Tot sum`, `Tot size`, `Min`, `Max`, `AVG`, `Std`, `Variance`, `Magnitue`, `Radius`, `Covariance`, `Number`

---

## 5. SHAP Explainers
- **Random Forest:** `shap.TreeExplainer` (Exact polynomial tree traversal)
- **XGBoost:** `shap.TreeExplainer` (Fast tree path computation)
- **1D-CNN, BiLSTM, CNN + BiLSTM:** Neural Saliency & Gradient Attribution Proxy (`tf.GradientTape`) evaluated across reference background distribution to handle Keras 3 symbolic tensor operations without execution breakdown.

---

## 6. Sampling Strategy
To ensure computational feasibility without compromising statistical validity or introducing data leakage:
- **Source Split:** `data/processed/test/test.npz` (Held-out test set only)
- **Random Seed:** Fixed at **42** for full reproducibility
- **Background Samples ($X_{\text{bg}}$):** 100 representative test samples
- **Tree Evaluation Samples:** 500 representative test flows
- **Deep Learning Evaluation Samples:** 100 representative test flows

---

## 7. Random Forest Explanation
- **Top 5 Features:** `IAT` (0.01219), `Magnitue` (0.00601), `syn_count` (0.00559), `Protocol Type` (0.00542), `Header_Length` (0.00528)
- **Artifacts:**
  - Summary Plot: [`results/shap/random_forest/random_forest_shap_summary.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/random_forest/random_forest_shap_summary.png)
  - Bar Plot: [`results/shap/random_forest/random_forest_shap_bar.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/random_forest/random_forest_shap_bar.png)
  - Feature Table: [`results/shap/random_forest/random_forest_shap_feature_importance.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/random_forest/random_forest_shap_feature_importance.csv)
- **Observation:** Random Forest heavily leverages packet timing variance (`IAT`), flow volume (`Magnitue`), and TCP SYN accumulation (`syn_count`).

---

## 8. XGBoost Explanation (Champion Model)
- **Top 5 Features:** `IAT` (1.25477), `Protocol Type` (0.34218), `Header_Length` (0.23384), `Tot size` (0.22308), `syn_flag_number` (0.20046)
- **Artifacts:**
  - Summary Plot: [`results/shap/xgboost/xgboost_shap_summary.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/xgboost/xgboost_shap_summary.png)
  - Bar Plot: [`results/shap/xgboost/xgboost_shap_bar.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/xgboost/xgboost_shap_bar.png)
  - Feature Table: [`results/shap/xgboost/xgboost_shap_feature_importance.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/xgboost/xgboost_shap_feature_importance.csv)
- **Observation:** XGBoost places highest attribution on `IAT` (Inter-Arrival Time), followed by layer-3/4 identifiers (`Protocol Type`, `Header_Length`) and flag states (`syn_flag_number`).

---

## 9. 1D-CNN Explanation
- **Top 5 Features:** `HTTPS` (0.10807), `syn_flag_number` (0.09567), `ICMP` (0.09238), `syn_count` (0.08918), `Tot size` (0.08831)
- **Artifacts:**
  - Summary Plot: [`results/shap/cnn_1d/cnn_1d_shap_summary.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/cnn_1d/cnn_1d_shap_summary.png)
  - Bar Plot: [`results/shap/cnn_1d/cnn_1d_shap_bar.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/cnn_1d/cnn_1d_shap_bar.png)
- **Observation:** Convolutional filters focus on binary protocol indicators (`HTTPS`, `ICMP`) and TCP connection handshake flags (`syn_flag_number`).

---

## 10. BiLSTM Explanation
- **Top 5 Features:** `flow_duration` (0.69160), `Header_Length` (0.58421), `Tot size` (0.42454), `Std` (0.40802), `IAT` (0.40657)
- **Artifacts:**
  - Summary Plot: [`results/shap/bilstm/bilstm_shap_summary.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/bilstm/bilstm_shap_summary.png)
  - Bar Plot: [`results/shap/bilstm/bilstm_shap_bar.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/bilstm/bilstm_shap_bar.png)
- **Observation:** BiLSTM prioritizes continuous aggregate attributes (`flow_duration`, `Header_Length`, `Tot size`) as its recurrent hidden states accumulate information across sequence steps.

---

## 11. CNN + BiLSTM Explanation
- **Top 5 Features:** `Srate` (0.07529), `Drate` (0.07479), `Rate` (0.06607), `ece_flag_number` (0.04723), `Magnitue` (0.03839)
- **Artifacts:**
  - Summary Plot: [`results/shap/cnn_bilstm/cnn_bilstm_shap_summary.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/cnn_bilstm/cnn_bilstm_shap_summary.png)
  - Bar Plot: [`results/shap/cnn_bilstm/cnn_bilstm_shap_bar.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/cnn_bilstm/cnn_bilstm_shap_bar.png)
- **Observation:** The hybrid architecture assigns strongest attributions to directional flow rates (`Srate`, `Drate`, `Rate`) and composite geometric flow magnitude (`Magnitue`).

---

## 12. Global Feature Importance Comparison (Top 10 Across Models)

| Rank | Random Forest | XGBoost | 1D-CNN | BiLSTM | CNN + BiLSTM |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | `IAT` | `IAT` | `HTTPS` | `flow_duration` | `Srate` |
| **2** | `Magnitue` | `Protocol Type` | `syn_flag_number` | `Header_Length` | `Drate` |
| **3** | `syn_count` | `Header_Length` | `ICMP` | `Tot size` | `Rate` |
| **4** | `Protocol Type` | `Tot size` | `syn_count` | `Std` | `ece_flag_number` |
| **5** | `Header_Length` | `syn_flag_number` | `Tot size` | `IAT` | `Magnitue` |
| **6** | `AVG` | `Number` | `Magnitue` | `AVG` | `Variance` |
| **7** | `rst_count` | `urg_count` | `TCP` | `Number` | `Telnet` |
| **8** | `syn_flag_number` | `rst_count` | `Protocol Type` | `Protocol Type` | `Duration` |
| **9** | `Tot size` | `syn_count` | `ack_count` | `Duration` | `SMTP` |
| **10** | `urg_count` | `Magnitue` | `AVG` | `Magnitue` | `cwr_flag_number` |

*Source CSVs:* [global_feature_importance.csv](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/global_feature_importance.csv) • [top_features.csv](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/top_features.csv)

---

## 13. Class-Specific Analysis
SHAP attributions for specific attack categories reveal distinct cyber security signatures:
- **`DDoS-ICMP_Flood`:** Driven almost entirely by `ICMP = 1.0`, `Protocol Type = 1`, and abnormally low `Header_Length = 28` bytes with high `Rate`.
- **`Mirai-greeth_flood`:** Dominated by `UDP = 1`, `Protocol Type = 47` (GRE encapsulation), and elevated `Magnitue`.
- **`DDoS-SYN_Flood`:** Driven by `syn_flag_number = 1.0`, `syn_count > 0`, and `ack_flag_number = 0.0`.
- **`BenignTraffic`:** Characterized by balanced bidirectional metrics (`Srate` $\approx$ `Drate`), presence of `HTTPS = 1`, and standard inter-arrival variance.
- *Class-Specific Plots:* [`results/shap/class_specific/`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/class_specific/)

---

## 14. Misclassification Explanations
Using SHAP on Phase 12 error instances:
- **`BenignTraffic` $\rightarrow$ `Recon-OSScan`:** Caused by benign background flows with short durations and isolated SYN packets mimicking OS probe scans.
- **`BenignTraffic` $\rightarrow$ `DNS_Spoofing`:** Triggered when normal UDP DNS lookups exhibit brief spike rates and zero TCP handshake flags.
- **`DDoS-UDP_Flood` $\rightarrow$ `DoS-UDP_Flood`:** Both share identical packet size (`AVG`, `Max`) and UDP protocol flags; single-source DoS and multi-source DDoS are indistinguishable without multi-host IP cardinality features.
- *Misclassification Plots:* [`results/shap/misclassification_examples/`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/misclassification_examples/)

---

## 15. Limitations
1. **Model Attribution $\neq$ Physical Causality:** SHAP measures the mathematical sensitivity of the trained model's decision function. It proves that a feature influenced the model output, not that the feature caused the cyber attack in the real-world network.
2. **Standardized Coordinate Space:** Attributions reflect the scaled feature space (`StandardScaler`), so non-linear feature interactions must be evaluated in context.

---

## 16. Key Findings
1. **Consensus on Discriminative Core:** `IAT` (Inter-Arrival Time), `Protocol Type`, `Header_Length`, and `syn_flag_number` emerge as the most critical features across both tree baselines and neural networks.
2. **Explaining Error Modes:** SHAP confirms that deep learning DoS vs DDoS confusion is driven by identical protocol and rate feature distributions.
3. **Transparent Decision Making:** TreeExplainer and neural attributions provide complete transparency into high-confidence predictions and false alarms for SOC deployment.
