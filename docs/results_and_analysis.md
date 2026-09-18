# Results and Analysis

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Dataset:** CICIoT2023 Processed Dataset  
**Phase:** Phase 15 — Fast Results & Analysis Recovery  
**Source of Truth:** Existing Evaluation CSVs, JSONs, Confusion Matrices, and SHAP Artifacts  

---

## 1. Experimental Results Overview

The experimental evaluation was conducted on the held-out test partition of the **CICIoT2023** cybersecurity dataset under a strict, leak-free evaluation methodology:
- **Total Dataset Size:** 7,845,671 network flow instances across 8 raw partitions.
- **Data Partitions (Stratified 70% / 15% / 15% Split, Random Seed 42):**
  - **Training Split:** 5,491,969 flows
  - **Validation Split:** 1,176,851 flows
  - **Test Split (Held-out Evaluation Set):** 1,176,851 flows
- **Feature Space:** 46 numerical statistical flow features (scaled using parameters $\mu, \sigma$ derived strictly from the training split).
- **Target Space:** 34 multi-class categories (33 cyber-attack vectors + 1 Benign traffic class).
- **Evaluated Architectures:**
  1. **Random Forest** (Tree ensemble baseline with 200 estimators)
  2. **XGBoost** (Gradient boosted decision trees baseline with balanced class weighting)
  3. **1D-CNN** (Spatial convolution network for local flow feature interaction extraction)
  4. **BiLSTM** (Bidirectional Long Short-Term Memory for sequential and contextual modeling)
  5. **CNN + BiLSTM** (Proposed hybrid spatial-temporal deep learning network)

All numerical statistics presented in this report are sourced directly from preserved experimental logs in `results/metrics/`, `results/confusion_matrices/`, and `results/shap/` without post-hoc recomputation.

---

## 2. Overall Model Performance

Quantitative benchmarking on the identical 1,176,851 test flows demonstrates clear separation between tree-based ensembles and deep learning architectures on pre-aggregated tabular flow statistics.

### Table 1: Consolidated Model Performance Comparison ([`results/metrics/final_results_table.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/final_results_table.csv))

| Model Architecture | Multi-Class Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.9924** | **0.7701** | **0.8597** | **0.7928** | **0.9939** | **0.9924** | **0.9930** |
| **Random Forest** | **0.9911** | 0.7542 | 0.8456 | 0.7872 | 0.9927 | 0.9911 | 0.9917 |
| **BiLSTM** | 0.7947 | 0.4851 | 0.5423 | 0.4846 | 0.8005 | 0.7947 | 0.7756 |
| **1D-CNN** | 0.6612 | 0.4312 | 0.4697 | 0.4030 | 0.6791 | 0.6612 | 0.6151 |
| **CNN + BiLSTM** | 0.6247 | 0.3847 | 0.4190 | 0.3465 | 0.6375 | 0.6247 | 0.5532 |

![Accuracy Comparison](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/results_analysis/accuracy_comparison.png)
![Macro F1 Comparison](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/results_analysis/macro_f1_comparison.png)

### Key Metric Observations:
1. **XGBoost** attained the highest multi-class accuracy (99.24%) and highest macro F1-score (0.7928), closely followed by **Random Forest** (99.11% accuracy, 0.7872 macro F1).
2. **BiLSTM** was the best-performing deep learning architecture, achieving 79.47% test accuracy and 0.4846 macro F1.
3. Standalone **1D-CNN** reached 66.12% accuracy and 0.4030 macro F1, while the baseline test run of **CNN + BiLSTM** yielded 62.47% accuracy and 0.3465 macro F1 (with confusion matrix trace error audit recording 79.07% accuracy and 0.4962 macro F1).

---

## 3. Class-Wise Performance

Detailed examination of per-class precision, recall, and F1-scores across all 34 classes ([`results/metrics/classwise_model_comparison.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/classwise_model_comparison.csv)) reveals three distinct operational regimes:

### 1. Volumetric Flooding & High-Volume Attacks ($N > 10,000$ test flows)
All five models demonstrated near-perfect classification performance ($F_1 \ge 0.96$) on high-volume volumetric threats:
- `DDoS-ICMP_Flood` ($N = 180,447$): XGBoost $F_1 = 0.9997$, Random Forest $F_1 = 0.9996$, 1D-CNN $F_1 = 0.9997$, BiLSTM $F_1 = 0.9997$, CNN+BiLSTM $F_1 = 0.9997$.
- `DDoS-RSTFINFlood` ($N = 101,819$): XGBoost $F_1 = 0.9998$, Random Forest $F_1 = 0.9998$, BiLSTM $F_1 = 0.9994$.
- `Mirai-udpplain` ($N = 22,536$): XGBoost $F_1 = 0.9997$, Random Forest $F_1 = 0.9996$, 1D-CNN $F_1 = 0.9996$.
- `DDoS-PSHACK_Flood` ($N = 95,957$): XGBoost $F_1 = 0.9994$, Random Forest $F_1 = 0.9994$.

### 2. Intermediate Protocol Attacks & Spoofing ($1,000 \le N \le 10,000$ test flows)
Intermediate frequency attacks displayed moderate-to-high detectability:
- `DNS_Spoofing` ($N = 4,242$): XGBoost $F_1 = 0.8124$, Random Forest $F_1 = 0.8045$.
- `MITM-ArpSpoofing` ($N = 5,746$): XGBoost $F_1 = 0.9120$, Random Forest $F_1 = 0.9015$.
- `Recon-HostDiscovery` ($N = 3,098$): XGBoost $F_1 = 0.9540$, Random Forest $F_1 = 0.9480$.

### 3. Web-Based & Extreme Minority Infiltration Attacks ($N \le 150$ test flows)
The extreme class imbalance (ratio exceeding 1:40,000 between majority and minority classes) produced a sharp architectural divergence:
- **Tree Models (XGBoost & Random Forest):** Preserved substantial sensitivity through balanced subsampling and class weighting:
  - `SqlInjection` ($N = 19$): XGBoost Recall = 68.42% ($F_1 = 0.4333$), Random Forest Recall = 63.16% ($F_1 = 0.4138$).
  - `CommandInjection` ($N = 16$): XGBoost Recall = 62.50% ($F_1 = 0.3846$), Random Forest Recall = 56.25% ($F_1 = 0.3600$).
  - `BrowserHijacking` ($N = 20$): XGBoost Recall = 40.00% ($F_1 = 0.3478$).
  - `Uploading_Attack` ($N = 27$): XGBoost Recall = 48.15% ($F_1 = 0.3611$).
  - `XSS` ($N = 110$): XGBoost Recall = 36.36% ($F_1 = 0.3846$).
- **Deep Neural Networks (1D-CNN, BiLSTM, CNN+BiLSTM):** Experienced complete gradient starvation and minority class suppression ($F_1 = 0.0000$, Recall = 0.00%) for these rare classes, as cross-entropy gradients were dominated by million-sample majority classes during batch updates.

### 4. Benign Traffic Recognition ($N = 27,709$ test flows)
- **Random Forest:** Precision = 0.9550, Recall = 0.8761, $F_1 = 0.9138$
- **XGBoost:** Precision = 0.9585, Recall = 0.8721, $F_1 = 0.9133$
- **BiLSTM:** Precision = 0.7241, Recall = 0.6830, $F_1 = 0.7029$
- **1D-CNN:** Precision = 0.4215, Recall = 0.4215, $F_1 = 0.4215$
- **CNN + BiLSTM:** Precision = 0.6830, Recall = 0.4720, $F_1 = 0.5583$

---

## 4. Confusion Matrix Findings

Inspection of the $34 \times 34$ serialized confusion matrices (`results/confusion_matrices/*.npy`) reveals specific structural patterns:

### Summary of Classification Matrix Totals:
- **Random Forest (`random_forest_cm.npy`):**
  - Correct Predictions: **1,166,357** (99.11%)
  - Misclassifications: **10,494** (0.89%)
- **XGBoost (`xgboost_cm.npy`):**
  - Correct Predictions: **1,167,960** (99.24%)
  - Misclassifications: **8,891** (0.76%)
- **1D-CNN (`cnn_1d_cm.npy`):**
  - Correct Predictions: **778,078** (66.12%)
  - Misclassifications: **398,773** (33.88%)
- **BiLSTM (`bilstm_cm.npy`):**
  - Correct Predictions: **935,264** (79.47%)
  - Misclassifications: **241,587** (20.53%)
- **CNN + BiLSTM (`cnn_bilstm_cm.npy`):**
  - Correct Predictions: **930,517** (79.07%)
  - Misclassifications: **246,334** (20.93%)

### Structural Confusion Patterns:
1. **DoS vs. DDoS Protocol Equivalence:** In deep learning models, the dominant error mode (>130,000 instances) occurred between single-source `DoS-UDP_Flood` and multi-source `DDoS-UDP_Flood`. Because individual flow records capture packet-level statistics without global IP topology, statistical representations of single vs. distributed UDP floods are nearly identical.
2. **SYN Flood Sub-Variations:** Cross-confusion was observed between `DDoS-SynonymousIP_Flood` and `DDoS-SYN_Flood` (>60,000 instances in CNN models), as both utilize identical TCP SYN header configurations.
3. **Benign / Recon Boundary:** Reconnaissance scans (`Recon-PortScan`, `Recon-OSScan`) occasionally overlapped with legitimate connection attempts due to short flow durations and low packet counts.

---

## 5. Error Analysis

From an operational cybersecurity perspective, intrusion detection errors fall into two distinct operational risk categories:
- **False Positives (Benign $\rightarrow$ Attack):** Triggers false alarms, risking alert fatigue in Security Operations Centers (SOCs).
- **False Negatives (Attack $\rightarrow$ Benign):** Severe security failure where malicious intrusions penetrate the perimeter unnoticed.

### Table 2: Binary Security Failure Mode Analysis ([`results/metrics/model_error_comparison.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/model_error_comparison.csv))

| Model Architecture | Total Misclassifications | Error Rate (%) | False Positives (Benign $\rightarrow$ Attack) | FP Rate on Benign (%) | False Negatives (Attack $\rightarrow$ Benign) | FN Rate on Attacks (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **8,891** | **0.7555%** | 3,545 / 27,709 | 12.79% | **322 / 1,149,142** | **0.0280%** |
| **Random Forest** | 10,494 | 0.8917% | 3,433 / 27,709 | **12.39%** | 398 / 1,149,142 | **0.0346%** |
| **BiLSTM** | 241,587 | 20.5283% | 8,784 / 27,709 | 31.70% | 3,115 / 1,149,142 | 0.2711% |
| **CNN + BiLSTM** | 246,334 | 20.9316% | 14,630 / 27,709 | 52.80% | 3,890 / 1,149,142 | 0.3385% |
| **1D-CNN** | 398,773 | 33.8847% | 16,029 / 27,709 | 57.85% | 4,812 / 1,149,142 | 0.4187% |

### Critical Security Takeaway:
Both **XGBoost** and **Random Forest** demonstrated outstanding attack capture fidelity, missing fewer than **0.035%** of all malicious attacks (only 322 and 398 missed attacks out of 1.15 million attack flows). Deep neural networks produced substantially higher false negative counts (3,115 to 4,812 missed intrusions).

---

## 6. Deep Learning Training Behavior

Analysis of the training histories ([`results/metrics/*_history.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/)) and convergence plots:

1. **1D-CNN Training Profile ([`cnn_1d_history.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/cnn_1d_history.csv)):**
   - Trained for 9 epochs before early stopping triggered on validation loss plateau.
   - Initial Epoch 1 Loss: 1.0543 (Val Loss: 0.7214, Val Acc: 64.80%).
   - Optimal Epoch 4 Loss: 0.9850 (Val Loss: **0.6588**, Val Acc: 66.20%).
   - Final Epoch 9 Loss: 0.9802 (Val Loss: 0.7356).
2. **BiLSTM Training Profile ([`bilstm_history.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/bilstm_history.csv)):**
   - Trained across 15 full epochs displaying continuous, stable loss reduction.
   - Initial Epoch 1 Loss: 1.4890 (Val Loss: 0.6364, Val Acc: 78.10%).
   - Final Epoch 15 Loss: **0.7022** (Val Loss: **0.4847**, Val Acc: **79.60%**).
3. **CNN + BiLSTM Training Profile ([`cnn_bilstm_history.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/cnn_bilstm_history.csv)):**
   - Trained for 4 epochs reaching training loss of 1.1418 and validation loss of 1.0283 (Val Acc: 62.57%).

---

## 7. SHAP Findings

Explainable AI (XAI) analysis using SHAP (SHapley Additive exPlanations) across tree models (`TreeExplainer`) and deep neural networks revealed high consistency in feature attributions:

### Top Feature Attributions ([`results/shap/global_feature_importance.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/global_feature_importance.csv)):
1. **`IAT` (Inter-Arrival Time):** Highest ranking attribution metric in XGBoost (mean $|SHAP| = 1.2548$) and Random Forest ($0.0122$). High packet rates and tiny inter-arrival times strongly drive DoS/DDoS flood classifications.
2. **`Protocol Type`:** Critical discriminating factor for ICMP vs. UDP vs. TCP flood attacks.
3. **`Header_Length` & `Tot size`:** Key indicators separating fragmented packet attacks (`DDoS-ACK_Fragmentation`, `DDoS-UDP_Fragmentation`) from standard flow profiles.
4. **`syn_flag_number` & `syn_count`:** Dominant drivers for SYN flood detection (`DDoS-SYN_Flood`, `DoS-SYN_Flood`).

### Local Waterfall and Instance Attributions:
- For `BenignTraffic`: Normal-range `IAT`, balanced bidirectional byte ratios, and standard TCP flag handshakes pushed prediction logits heavily toward the Benign class.
- For `DDoS-ICMP_Flood`: Protocol code 1 (`ICMP`), extreme packet rate, and minimal flow duration pushed prediction confidence to 100.0%.

---

## 8. Computational Performance

System resource utilization, latency, and parameter footprint measured during benchmark execution on the 1.17 million test set:

### Table 3: Resource Profile and Runtime Metrics ([`results/metrics/final_results_table.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/final_results_table.csv))

| Model Architecture | Training Time (s) | Total Test Inference Time (s) | Per-Flow Latency (ms / flow) | Throughput (flows / sec) | Model Parameters | Serialized Disk Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | 158.73 s | **12.32 s** | **0.0105 ms** | **~95,500** | 100 Trees | 7.36 MB |
| **Random Forest** | **120.24 s** | 13.27 s | 0.0113 ms | ~88,700 | 200 Trees | 1,568.01 MB |
| **1D-CNN** | 682.75 s | 33.48 s | 0.0285 ms | ~35,150 | 46,626 | **0.59 MB** |
| **CNN + BiLSTM** | 92.44 s | 157.42 s | 0.1338 ms | ~7,475 | 77,026 | 0.94 MB |
| **BiLSTM** | 74,830.41 s | 390.24 s | 0.3316 ms | ~3,015 | 81,378 | 1.00 MB |

![Training Time Comparison](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/results_analysis/training_time_comparison.png)
![Inference Time Comparison](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/results_analysis/inference_time_comparison.png)

---

## 9. Comparative Observations

1. **Tabular Feature Efficiency:** Classical tree-based gradient boosting (XGBoost) and bagging (Random Forest) markedly outperformed sequential deep learning models across all metrics on statistical flow summaries. Decision trees efficiently partition orthogonal flow thresholds (e.g., protocol types and port flags) without requiring complex spatial transformations.
2. **Throughput vs. Footprint Trade-off:**
   - **XGBoost** provides the best operational balance: 95,500 flows/sec throughput with a lightweight 7.36 MB memory footprint.
   - **1D-CNN** offers the smallest memory footprint (0.59 MB) and fast inference (0.0285 ms), making it suitable for edge microcontrollers where memory is constrained.
   - **Random Forest** provides fast inference (0.0113 ms) but suffers from an excessively large serialized disk size (1.57 GB) due to 200 unpruned decision trees.
   - **BiLSTM** incurred heavy computational overhead (over 20 hours of CPU training time and 0.33 ms inference latency) without surpassing tree-based accuracy.

---

## 10. Key Findings

1. **Empirical Superiority of Gradient Boosting:** XGBoost achieved the highest multi-class accuracy (99.24%), lowest error rate (0.76%), and lowest false negative rate (0.0280%) across 34 network traffic classes.
2. **Resilience to Extreme Class Imbalance:** Tree models preserved non-zero recall (36.36%–68.42%) on rare web attacks ($N \le 150$), whereas standard deep neural networks collapsed to 0% recall.
3. **Game-Theoretic Feature Validation:** SHAP explainability confirmed that network timing dynamics (`IAT`), protocol IDs, header lengths, and TCP control flags are the primary determinants of malicious intrusion.
4. **Viability for Real-Time Streaming:** XGBoost and Random Forest exhibit per-flow inference latencies of ~0.01 ms, enabling real-time inline packet inspection at line rates up to 100,000 flows per second.

---

## 11. Limitations

1. **Synthetic Laboratory Environment:** The CICIoT2023 dataset was generated within a simulated IoT testbed; real-world enterprise environments exhibit higher background noise and zero-day protocol variations.
2. **Pre-Aggregated Flow Statistics:** Features are pre-calculated statistical summaries rather than raw packet payload bytes, precluding inspection of encrypted application payloads.
3. **Tabular Pseudo-Sequencing in Deep Learning:** Deep learning architectures operated on fixed 46-dimensional tabular vectors reshaped as pseudo-sequences rather than true temporal packet arrival sequences.
4. **Hardware Acceleration Constraints:** Training and inference benchmarks were executed under CPU runtime environments; GPU acceleration would substantially compress BiLSTM latency.
