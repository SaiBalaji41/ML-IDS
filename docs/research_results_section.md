# 4. Results and Discussion

This section presents the empirical evaluation, comparative benchmarking, and explainability analysis of the five evaluated Intrusion Detection System (IDS) architectures on the **CICIoT2023** benchmark dataset. All experiments were conducted under a standardized, leakage-free methodology on a held-out test partition comprising $N = 1,176,851$ network flows across 34 multi-class categories (33 cyber-attack types and 1 benign traffic class).

---

## 4.1 Overall Performance Comparison

Table 4.1 details the multi-class classification performance metrics achieved across all candidate architectures on the identical held-out test split.

### Table 4.1: Multi-Class Evaluation Metrics on Held-Out Test Split ($N = 1,176,851$)

| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Baseline)** | **0.9924** | **0.7701** | **0.8597** | **0.7928** | **0.9939** | **0.9924** | **0.9930** |
| **Random Forest (Baseline)** | 0.9911 | 0.7542 | 0.8456 | 0.7872 | 0.9927 | 0.9911 | 0.9917 |
| **BiLSTM** | 0.7947 | 0.4851 | 0.5423 | 0.4846 | 0.8005 | 0.7947 | 0.7756 |
| **1D-CNN** | 0.6612 | 0.4312 | 0.4697 | 0.4030 | 0.6791 | 0.6612 | 0.6151 |
| **CNN + BiLSTM (Hybrid)** | 0.6247 | 0.3847 | 0.4190 | 0.3465 | 0.6375 | 0.6247 | 0.5532 |

Gradient boosted decision trees (**XGBoost**) demonstrated the highest classification efficacy, attaining a multi-class accuracy of 99.24%, a macro-averaged F1-score of 0.7928, and a weighted F1-score of 0.9930. **Random Forest** achieved competitive performance with 99.11% accuracy and 0.7872 macro F1-score. Among the deep learning models, **BiLSTM** achieved the strongest baseline performance (79.47% accuracy, 0.4846 macro F1), outperforming standalone **1D-CNN** (66.12% accuracy, 0.4030 macro F1).

---

## 4.2 Class-Wise Performance Observations

Evaluation across individual class categories revealed stark differences driven by sample representation and traffic characteristics:

1. **Volumetric Flood Dominance:** For volumetric flood attacks with large test representation ($N > 10,000$), all five models exhibited near-optimal discrimination ($F_1 > 0.98$). Specifically, on `DDoS-ICMP_Flood` ($N = 180,447$), `DDoS-RSTFINFlood` ($N = 101,819$), and `Mirai-udpplain` ($N = 22,536$), all models achieved F1-scores exceeding 0.999.
2. **Intermediate Protocol & Reconnaissance Vectors:** Intermediate-frequency attacks ($1,000 \le N \le 10,000$) were robustly identified by ensemble tree models: `Recon-HostDiscovery` ($F_1 = 0.9540$), `MITM-ArpSpoofing` ($F_1 = 0.9120$), and `DNS_Spoofing` ($F_1 = 0.8124$).
3. **Severe Minority Class Collapse:** For rare web-based attack vectors ($N \le 150$), tree-based models maintained non-zero recall rates:
   - `SqlInjection` ($N = 19$): XGBoost Recall = 68.42% ($F_1 = 0.4333$).
   - `CommandInjection` ($N = 16$): XGBoost Recall = 62.50% ($F_1 = 0.3846$).
   - `BrowserHijacking` ($N = 20$): XGBoost Recall = 40.00% ($F_1 = 0.3478$).
   - `Uploading_Attack` ($N = 27$): XGBoost Recall = 48.15% ($F_1 = 0.3611$).
   - In contrast, deep neural architectures suffered from gradient starvation on these extreme minorities, registering $0.00\%$ recall ($F_1 = 0.0000$).
4. **Benign Traffic Identification ($N = 27,709$):** Random Forest and XGBoost achieved F1-scores of 0.9138 and 0.9133, respectively, maintaining high precision (0.9550 and 0.9585) against false positive alarms.

---

## 4.3 Confusion Matrix Observations

Analysis of the $34 \times 34$ confusion matrices revealed specific structural confusion patterns:
- **Random Forest:** 1,166,357 correct classifications (99.11%) vs. 10,494 misclassifications (0.89%).
- **XGBoost:** 1,167,960 correct classifications (99.24%) vs. 8,891 misclassifications (0.76%).
- **1D-CNN:** 778,078 correct classifications (66.12%) vs. 398,773 misclassifications (33.88%).
- **BiLSTM:** 935,264 correct classifications (79.47%) vs. 241,587 misclassifications (20.53%).
- **CNN + BiLSTM:** 930,517 correct classifications (79.07%) vs. 246,334 misclassifications (20.93%).

Primary confusion modes in neural architectures were concentrated along protocol symmetries:
- `DDoS-UDP_Flood` misclassified as `DoS-UDP_Flood` (>130,000 samples) due to identical packet header attributes without multi-source IP topology context.
- `DDoS-SynonymousIP_Flood` misclassified as `DDoS-SYN_Flood` (>60,000 samples) due to identical TCP SYN handshake metrics.

---

## 4.4 Error Analysis & Security Reliability

Table 4.2 presents the breakdown of operational security failures across models.

### Table 4.2: Security Error Rates on Held-Out Test Partition

| Model Architecture | Total Errors | Error Rate (%) | False Positives (Benign $\rightarrow$ Attack) | FP Rate (%) | False Negatives (Attack $\rightarrow$ Benign) | FN Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **8,891** | **0.7555%** | 3,545 / 27,709 | 12.79% | **322 / 1,149,142** | **0.0280%** |
| **Random Forest** | 10,494 | 0.8917% | **3,433 / 27,709** | **12.39%** | 398 / 1,149,142 | 0.0346% |
| **BiLSTM** | 241,587 | 20.5283% | 8,784 / 27,709 | 31.70% | 3,115 / 1,149,142 | 0.2711% |
| **CNN + BiLSTM** | 246,334 | 20.9316% | 14,630 / 27,709 | 52.80% | 3,890 / 1,149,142 | 0.3385% |
| **1D-CNN** | 398,773 | 33.8847% | 16,029 / 27,709 | 57.85% | 4,812 / 1,149,142 | 0.4187% |

Both tree ensemble models achieved false negative rates below **0.035%**, successfully intercepting over 99.96% of all malicious traffic flows.

---

## 4.5 Deep Learning Training Behavior

Training convergence histories across epochs demonstrated:
- **1D-CNN:** Achieved its minimum validation loss of 0.6588 at Epoch 4 before loss plateauing triggered early stopping at Epoch 9 (validation accuracy: 66.20%).
- **BiLSTM:** Displayed steady monotonic convergence over 15 training epochs, reducing validation loss from 0.6364 to 0.4847 and achieving 79.60% validation accuracy.
- **CNN + BiLSTM:** Completed 4 training epochs, reaching 1.0283 validation loss and 62.57% validation accuracy.

---

## 4.6 SHAP Interpretability Observations

Game-theoretic SHAP attributions provided interpretability across decision boundaries:
1. **Global Attribution Leaders:** `IAT` (Inter-Arrival Time), `Protocol Type`, `Header_Length`, `Tot size`, and `syn_flag_number` emerged as the top contributing features across all models.
2. **Timing Discrimination:** TreeExplainer attributions revealed that low `IAT` and high flow packet rates are the strongest drivers for volumetric flood classifications.
3. **Local Explanations:** Local waterfall plots confirmed that normal inter-arrival times and standard TCP flags drove benign traffic predictions, whereas anomalous header structures and protocol codes drove attack alerts.

---

## 4.7 Computational Performance & Deployment Feasibility

Table 4.3 summarizes computational overhead, latency, parameter complexity, and storage footprints.

### Table 4.3: Computational Profile and Inference Latency on Test Partition

| Model Architecture | Training Time (s) | Total Test Inference Time (s) | Per-Flow Latency (ms / flow) | Throughput (flows / sec) | Model Parameters | Serialized Disk Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | 158.73 s | **12.32 s** | **0.0105 ms** | **~95,500** | 100 Trees | 7.36 MB |
| **Random Forest** | **120.24 s** | 13.27 s | 0.0113 ms | ~88,700 | 200 Trees | 1,568.01 MB |
| **1D-CNN** | 682.75 s | 33.48 s | 0.0285 ms | ~35,150 | 46,626 | **0.59 MB** |
| **CNN + BiLSTM** | 92.44 s | 157.42 s | 0.1338 ms | ~7,475 | 77,026 | 0.94 MB |
| **BiLSTM** | 74,830.41 s | 390.24 s | 0.3316 ms | ~3,015 | 81,378 | 1.00 MB |

XGBoost and Random Forest demonstrated sub-millisecond per-flow latency (~0.01 ms/flow), capable of sustaining high-throughput line-rate network monitoring up to ~95,000 flows/second.
