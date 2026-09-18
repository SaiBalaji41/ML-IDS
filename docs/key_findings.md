# Key Experimental Findings

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Dataset:** CICIoT2023 Held-Out Test Partition ($N = 1,176,851$)  
**Scope:** Multi-Model Empirical Evaluation, Error Analysis, and XAI Attribution

---

## 1. Summary of Supported Empirical Findings

The following key findings are grounded directly in the numerical results obtained from the experimental evaluation:

### 1. Tabular Flow Data Performance Differences
- The tree-based ensembles (XGBoost and Random Forest) achieved multi-class test accuracies of **99.24%** and **99.11%**, with macro F1-scores of **0.7928** and **0.7872**, respectively.
- The deep neural network architectures achieved test accuracies of **79.47%** (BiLSTM), **66.12%** (1D-CNN), and **62.47%** (CNN + BiLSTM in baseline testing, with 79.07% in full-pass confusion matrix analysis), yielding macro F1-scores between **0.3465** and **0.4846**.

### 2. High-Volume Attack Classification Consistency
- For high-volume volumetric flood attacks (e.g., `DDoS-ICMP_Flood`, `DDoS-RSTFINFlood`, `Mirai-udpplain`), all five evaluated models achieved class-wise F1-scores above **0.96**.

### 3. Extreme Minority Attack Sensitivity
- For rare attack classes with test support $N \le 150$ (`SqlInjection`, `CommandInjection`, `BrowserHijacking`, `Uploading_Attack`, `XSS`):
  - Tree-based models maintained non-zero recall ranging from **36.36%** to **68.07%**.
  - Deep learning models obtained zero true positives (**0.00% recall** and **0.00 F1-score**).

### 4. Symmetric DoS vs. DDoS Confusion
- The largest source of misclassification across deep learning models occurred symmetrically between single-source DoS and distributed DDoS attacks sharing identical protocol signatures (e.g., `DDoS-UDP_Flood` vs. `DoS-UDP_Flood` with over 130,000 misclassified flows per neural model).

### 5. False Alarm and Security Leakage Dynamics
- On 27,709 benign test flows, Random Forest misclassified 3,433 flows as attacks (12.39% FP rate), while XGBoost misclassified 3,545 flows (12.79% FP rate).
- On 1,149,142 malicious flows, XGBoost allowed 322 attacks through as benign (0.0280% attack FN rate), and Random Forest allowed 398 attacks through as benign (0.0346% attack FN rate).

### 6. Computational Latency and Resource Tradeoffs
- XGBoost demonstrated the lowest per-flow inference latency at **0.0105 ms** (~95,200 flows/sec) with a model disk size of **7.36 MB**.
- BiLSTM required the largest training duration on CPU (**74,830 seconds**, ~20.8 hours) and highest inference latency (**0.3316 ms** per flow).
- 1D-CNN produced the smallest serialized storage footprint (**0.59 MB**, 46,626 parameters).
- Random Forest required the largest storage footprint (**1,568.01 MB**) due to 200 unpruned decision trees.

### 7. SHAP Feature Attribution Consensus
- Game-theoretic attribution using SHAP identified `IAT` (Inter-Arrival Time), `Protocol Type`, `Header_Length`, `Tot size`, and `syn_flag_number` as the primary features influencing model predictions across tree ensembles and deep learning baselines.
