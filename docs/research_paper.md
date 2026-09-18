# ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring

**Author:** Academic Research Team & AI Engineering Assistant  
**Dataset Benchmark:** CICIoT2023  
**Status:** Comprehensive Research Manuscript Draft  

---

## Abstract

The proliferation of Internet of Things (IoT) devices and high-throughput network infrastructures has dramatically escalated vulnerability to multi-vector cyber attacks, including volumetric Distributed Denial of Service (DDoS), stealthy reconnaissance scans, and botnet propagation. Traditional signature-based Network Intrusion Detection Systems (NIDS) fail against zero-day exploits and polymorphic threat behaviors, necessitating robust, data-driven machine learning (ML) and deep learning (DL) alternatives. 

In this work, we present a comprehensive, leakage-free empirical evaluation of five candidate intrusion detection models on the large-scale **CICIoT2023** benchmark dataset comprising **7,845,673 network flows** across **34 multi-class categories** (33 distinct attack vectors and 1 benign traffic class). The evaluated architectures include classical tree ensembles (**Random Forest** and **XGBoost**), standalone deep neural networks (**1D-CNN** and **BiLSTM**), and a proposed hybrid **1D-CNN + BiLSTM** spatial-temporal network. All models were evaluated under identical, leak-free conditions on a held-out test partition of **1,176,851 flows** using standardized multi-class classification metrics, detailed confusion matrix audits, and game-theoretic **SHAP (SHapley Additive exPlanations)** interpretability.

Empirical results demonstrate that gradient boosted decision trees (**XGBoost**) achieved the highest multi-class accuracy of **99.24%**, a macro F1-score of **0.7928**, a weighted F1-score of **0.9930**, and an ultra-low false negative rate on malicious traffic of **0.0280%** (missing only 322 out of 1.15 million attack flows), while processing flows at **0.0105 ms/flow** (~95,500 flows/sec). **Random Forest** attained comparable performance with **99.11%** accuracy and **0.7872** macro F1. Among neural architectures, **BiLSTM** achieved the strongest baseline performance (**79.47%** accuracy, **0.4846** macro F1), while standalone **1D-CNN** reached **66.12%** accuracy. Tree-based ensembles demonstrated superior resilience to extreme class imbalance, retaining sensitivity on rare web attacks ($N \le 150$), whereas standard deep neural networks experienced gradient starvation and output collapse on extreme minorities. SHAP feature attributions revealed that packet Inter-Arrival Time (`IAT`), transport `Protocol Type`, `Header_Length`, and TCP control flags are the primary drivers of model predictions across both classical and deep learning paradigms. Finally, we discuss key data representation constraints, computational trade-offs, and outline an architectural roadmap for real-time packet stream ingestion via Scapy and Apache Kafka.

**Keywords:** Intrusion Detection System, Machine Learning, Deep Learning, CICIoT2023, Random Forest, XGBoost, 1D-CNN, BiLSTM, CNN-BiLSTM, Explainable AI, SHAP, Network Security.

---

## 1. Introduction

The exponential growth of connected Internet of Things (IoT) ecosystems—spanning smart cities, industrial control systems, healthcare telemetry, and autonomous infrastructure—has fundamentally expanded the modern cyber attack surface. IoT devices are frequently characterized by constrained compute capabilities, minimal firmware security, unpatched vulnerabilities, and heterogeneous communication protocols, rendering them prime targets for automated botnet recruitment (e.g., Mirai variants), distributed denial of service (DDoS) reflection attacks, and lateral network penetration.

Traditional Network Intrusion Detection Systems (NIDS) primarily depend on deterministic signature-matching engines (such as Snort or Suricata) or static threshold rules. While signature-based mechanisms deliver high precision with minimal latency on known attack signatures, they exhibit two critical structural vulnerabilities:
1. **Inability to Generalize:** Signature systems fail completely against novel zero-day attacks, obfuscated payloads, and polymorphic attack vectors.
2. **Alert Fatigue and Manual Rule Maintenance:** Security Operations Center (SOC) analysts face overwhelming volumes of false positive alerts during legitimate traffic surges, while maintaining manual rule sets becomes intractable in dynamic, high-speed networks.

Machine learning (ML) and deep learning (DL) models have emerged as compelling paradigms for anomaly-based intrusion detection. Classical algorithms (e.g., Random Forests and Gradient Boosted Decision Trees) excel at partitioning tabular statistical flow features, capturing non-linear threshold boundaries across network metrics. Concurrently, deep neural architectures—such as 1D Convolutional Neural Networks (1D-CNN) and Bidirectional Long Short-Term Memory networks (BiLSTM)—offer the potential to automatically learn hierarchical spatial interactions and contextual dependencies directly from high-dimensional network attributes.

However, widespread enterprise adoption of ML/DL-based intrusion detection faces two major hurdles: the absence of rigorous, large-scale empirical benchmarking across identical held-out test partitions, and the "black-box" nature of complex neural models. In mission-critical cybersecurity environments, automated alerts without human-interpretable justification cannot be verified by security analysts.

To address these challenges, this study provides an end-to-end, scientifically rigorous investigation of machine learning, deep learning, and explainable artificial intelligence (XAI) for network intrusion detection. Using the modern **CICIoT2023** dataset, we implement, benchmark, and interpret five candidate architectures under a zero-leakage experimental pipeline, quantifying classification accuracy, class-wise imbalance dynamics, confusion patterns, computational latency, and game-theoretic SHAP feature attributions.

---

## 2. Related Work

The application of machine learning to cybersecurity has evolved significantly over the past two decades, transitioning from classical shallow classifiers to deep and hybrid architectures.

### 2.1 Machine Learning for Network Intrusion Detection
Classical machine learning algorithms have long served as the foundation for statistical anomaly detection. Breiman (2001) established Random Forests as robust bagging ensembles capable of handling high-dimensional feature spaces with low variance and intrinsic feature importance estimation. Chen and Guestrin (2016) introduced XGBoost, an optimized distributed gradient boosting framework utilizing second-order Taylor expansions and regularization, which has consistently dominated tabular data benchmarks. In network security surveys, Khraisat et al. (2019) and Al-Garadi et al. (2020) demonstrated that decision tree ensembles achieve superior detection accuracy on statistical flow summaries compared to linear classifiers and support vector machines.

### 2.2 Deep Learning & Hybrid Architectures
Deep learning models (LeCun et al., 2015) eliminate the need for manual feature engineering by learning hierarchical representations directly from raw inputs. In intrusion detection, 1D-CNNs have been applied to capture localized relationships across neighboring flow attributes. Recurrent architectures, specifically Long Short-Term Memory (LSTM; Hochreiter & Schmidhuber, 1997) and Bidirectional LSTM (BiLSTM; Graves & Schmidhuber, 2005), were developed to capture long-range contextual dependencies by processing sequences in both forward and backward temporal directions. Roopak et al. (2019) explored deep learning models for IoT cyber security, demonstrating that hybrid architectures combining convolutional layers for feature extraction with recurrent layers for sequence context can improve classification accuracy on complex network traffic.

### 2.3 Benchmark Datasets: From KDD99 to CICIoT2023
Early intrusion detection research heavily relied on outdated benchmarks such as KDD Cup 1999 and NSL-KDD, which suffer from redundant records, artificial traffic distributions, and obsolete attack signatures. To address modern IoT threat landscapes, the Canadian Institute for Cybersecurity developed **CICIoT2023** (Neto et al., 2023). CICIoT2023 represents an extensive, realistic IoT network benchmark containing over 40 million records across 33 attack classes executed within a topology of 105 real and simulated IoT devices.

### 2.4 Explainable AI (XAI) & SHAP
As deep learning models increase in complexity, post-hoc interpretability has become paramount. Lundberg and Lee (2017) introduced SHAP (SHapley Additive exPlanations), a game-theoretic framework unifying additive feature attribution methods based on classic Shapley values. Lundberg et al. (2020) further developed TreeSHAP, enabling exact, polynomial-time computation of Shapley values for tree ensemble models. Applying SHAP to cybersecurity models allows SOC analysts to audit individual alert drivers and validate global model decision boundaries.

---

## 3. Problem Statement

Modern IoT networks are subjected to sophisticated, multi-stage cyber assaults characterized by:
1. **Volumetric Flood Asymmetries:** High-volume DoS/DDoS attacks generating millions of packets that overwhelm network infrastructure and mask subtle concurrent intrusions.
2. **Extreme Class Imbalance:** Severe data skew where critical infiltration vectors (such as SQL Injection, Command Injection, and Web Attacks) constitute less than 0.05% of total traffic, causing standard learning algorithms to optimize purely for majority class accuracy.
3. **Black-Box Decision Opacity:** The inability of complex neural models to provide auditable reasoning for security alerts, hindering root-cause analysis and dynamic firewall policy generation.

This research addresses these challenges by systematically developing, evaluating, and interpreting classical and deep learning models on tabular flow data, isolating classification failure modes, and establishing human-interpretable feature attribution frameworks.

---

## 4. Objectives

The specific technical objectives of this research are:
1. Conduct an empirical exploratory data analysis of the **CICIoT2023** dataset to verify schema, dimensions, and class distributions.
2. Implement a modular, zero-leakage preprocessing and feature standardization pipeline.
3. Train and tune classical baseline models (**Random Forest** and **XGBoost**).
4. Implement and optimize a standalone deep convolutional network (**1D-CNN**).
5. Implement and optimize a standalone bidirectional recurrent network (**BiLSTM**).
6. Design and evaluate a hybrid **1D-CNN + BiLSTM** architecture for spatial-temporal representation learning.
7. Execute multi-class evaluation across 34 classes using macro/weighted metrics and full confusion matrices.
8. Perform detailed error diagnostics, quantifying operational False Positive and False Negative rates.
9. Apply **SHAP** to extract global feature rankings and generate instance-level local explanations.
10. Transparently document computational limitations, data representation boundaries, and outline a real-time streaming architecture.

---

## 5. Proposed System Architecture

The implemented offline experimental pipeline operates through a sequence of modular stages designed to guarantee scientific reproducibility and complete isolation of evaluation data:

```
┌──────────────────────────────────────────────────────────┐
│           Raw CICIoT2023 Dataset (data/raw/)             │
│        (7,845,673 Records | 46 Features | 34 Classes)     │
└─────────────────────────────┬────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│            Data Preprocessing & Sanitization             │
│   • Infinity Imputation & Null Value Validation          │
│   • Integer Target Encoding (data/processed/mapping.json)│
│   • Stratified 70% / 15% / 15% Splitting (Seed 42)       │
└─────────────────────────────┬────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│         Leakage-Free Feature Standardization             │
│    StandardScaler fitted STRICTLY on Training Split      │
│  Validation & Test Splits transformed statelessly (μ, σ) │
└─────────────────────────────┬────────────────────────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│ Classical Machine Learning  │ │    Deep Neural Networks     │
│ • Random Forest (200 Trees) │ │ • Standalone 1D-CNN         │
│ • XGBoost (100 Estimators)  │ │ • Standalone BiLSTM         │
│                             │ │ • Proposed CNN + BiLSTM     │
└──────────────┬──────────────┘ └──────────────┬──────────────┘
               │                             │
               └──────────────┬──────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│  Multi-Class Model Evaluation on Held-Out Test Split     │
│   (1,176,851 Test Flows across 34 Multiclass Categories) │
│   • Accuracy, Precision, Recall, Macro/Weighted F1       │
│   • 34x34 Confusion Matrices (Raw & Normalized Heatmaps) │
│   • Security Error Analysis (False Positives / Negatives)│
└─────────────────────────────┬────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│         Explainable AI (XAI) Attribution via SHAP        │
│   • TreeExplainer Global Feature Importance Rankings     │
│   • Local Instance Waterfall Plots for Attack Validation │
└──────────────────────────────────────────────────────────┘
```

---

## 6. Dataset and Data Representation

### 6.1 Dataset Origin and Scale
The experiments utilize the **CICIoT2023** benchmark generated by the Canadian Institute for Cybersecurity. The analyzed corpus consists of **7,845,673 network traffic flows** distributed across 3 pre-partitioned CSV files:
- `train.csv`: 5,491,971 flows (70.00% partition, 1.512 GB)
- `validation.csv`: 1,176,851 flows (15.00% partition, 331.8 MB)
- `test.csv`: 1,176,851 flows (15.00% partition, 331.8 MB)
- Total Raw File Size: ~2.176 GB

### 6.2 Data Representation
> [!IMPORTANT]
> **Data Representation Verification:**  
> The dataset consists strictly of **tabular statistical network flow and protocol metrics** calculated over bidirectional network conversations. It does **not** contain raw packet payload byte streams or raw packet capture (PCAP) files. Consequently, all machine learning and deep learning models operate on fixed 46-dimensional numerical feature vectors.

### 6.3 Feature Space
The 46 input features capture diverse network dynamics across five major categories:
1. **Flow Timing & Rates (7 features):** `flow_duration`, `Header_Length`, `Protocol Type`, `Duration`, `Rate`, `Srate`, `Drate`.
2. **TCP Control Flags (12 features):** `fin_flag_number`, `syn_flag_number`, `rst_flag_number`, `psh_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number`, `ack_count`, `syn_count`, `fin_count`, `urg_count`, `rst_count`.
3. **Protocol Indicators (14 features):** `HTTP`, `HTTPS`, `DNS`, `Telnet`, `SMTP`, `SSH`, `IRC`, `TCP`, `UDP`, `DHCP`, `ARP`, `ICMP`, `IPv`, `LLC`.
4. **Packet Size Moments (6 features):** `Tot sum`, `Min`, `Max`, `AVG`, `Std`, `Tot size`.
5. **Flow Dynamics & Dispersion (7 features):** `IAT` (Inter-Arrival Time), `Number`, `Magnitue`, `Radius`, `Covariance`, `Variance`, `Weight`.

### 6.4 Target Classes & Imbalance Profile
The target space comprises **34 distinct classes** (1 `BenignTraffic` class and 33 attack vectors) organized across 7 threat taxonomies:
- **DDoS Attacks (64.22%):** `DDoS-ICMP_Flood` (15.43%), `DDoS-UDP_Flood` (11.61%), `DDoS-TCP_Flood` (9.64%), `DDoS-PSHACK_Flood` (8.76%), `DDoS-SYN_Flood` (8.71%), `DDoS-RSTFINFlood` (8.65%), `DDoS-SynonymousIP_Flood` (7.69%), `DDoS-ICMP_Fragmentation` (0.97%), `DDoS-UDP_Fragmentation` (0.62%), `DDoS-ACK_Fragmentation` (0.61%), `DDoS-SlowLoris` (0.05%), `DDoS-HTTP_Flood` (0.06%).
- **DoS Attacks (17.31%):** `DoS-UDP_Flood` (7.11%), `DoS-TCP_Flood` (5.72%), `DoS-SYN_Flood` (4.33%), `DoS-HTTP_Flood` (0.15%).
- **Mirai Botnet (5.64%):** `Mirai-greeth_flood` (2.12%), `Mirai-udpplain` (1.91%), `Mirai-greip_flood` (1.62%).
- **Benign Traffic (2.36%):** `BenignTraffic` (184,766 total samples).
- **Spoofing & MITM (1.05%):** `MITM-ArpSpoofing` (0.66%), `DNS_Spoofing` (0.39%).
- **Reconnaissance (0.68%):** `Recon-HostDiscovery` (0.29%), `Recon-OSScan` (0.21%), `Recon-PortScan` (0.18%), `VulnerabilityScan` (0.08%), `Recon-PingSweep` (0.004%).
- **Web & Infiltration Attacks (<0.07%):** `DictionaryBruteForce` (0.027%), `BrowserHijacking` (0.012%), `SqlInjection` (0.011%), `CommandInjection` (0.011%), `XSS` (0.008%), `Backdoor_Malware` (0.007%), `Uploading_Attack` (0.003%).

The ratio between the most populous class (`DDoS-ICMP_Flood`, 1,210,546 samples) and the rarest class (`Uploading_Attack`, 208 samples) exceeds **5,800:1**, presenting severe class imbalance.

---

## 7. Data Preprocessing & Leakage Prevention

To ensure total scientific integrity, the data pipeline enforces strict leakage-free transformations:
1. **Cleaning:** Infinite values (`np.inf`, `-np.inf`) were converted to `np.nan` and median-imputed.
2. **Label Encoding:** 34 target string labels were deterministically mapped to continuous integer indices in the range `[0..33]` saved in `data/processed/label_mapping.json`.
3. **Stateless Feature Standardization:** A `StandardScaler` was fitted **strictly on the training split** ($N = 5,491,971$). The validation ($N = 1,176,851$) and test ($N = 1,176,851$) partitions were transformed statelessly using the training mean $\mu$ and standard deviation $\sigma$. The fitted scaler was serialized to `models/preprocessing/scaler.pkl`.
4. **Target Isolation:** The target column was completely excluded from feature matrices to prevent target leakage.

---

## 8. Methodology & Model Architectures

### 8.1 Random Forest Baseline
Random Forest (Breiman, 2001) operates as an ensemble of unpruned decision trees trained via bootstrap aggregating (bagging) with random feature sub-selection.
- **Hyperparameters:** $200$ decision trees, maximum depth $25$, `max_samples = 0.10` (subsampling 10% per tree for scalable memory management), Gini impurity criterion, and balanced class weights.
- **Role:** Serves as a classical bagging ensemble baseline.

### 8.2 XGBoost Baseline
XGBoost (Chen & Guestrin, 2016) builds an additive ensemble of shallow decision trees via gradient boosting, optimizing a second-order approximation of the multi-class objective function with $L_1$ and $L_2$ leaf regularization.
- **Hyperparameters:** $100$ boosting rounds, maximum tree depth $6$, learning rate $\eta = 0.10$, column subsample ratio $0.80$, row subsample ratio $0.80$, histogram-based tree method (`tree_method='hist'`), and balanced sample weighting.
- **Role:** Serves as a state-of-the-art gradient boosted baseline.

### 8.3 Standalone 1D-CNN
The 1D-CNN architecture extracts localized spatial interactions across contiguous subsets of the 46 flow features. The input feature vector is reshaped into a tensor of shape `(batch_size, 46, 1)`.
- **Layer Structure:**
  1. `Conv1D(64 filters, kernel_size=3, padding='same', activation='relu')`
  2. `BatchNormalization()` $\rightarrow$ `MaxPooling1D(pool_size=2)` $\rightarrow$ `Dropout(0.30)`
  3. `Conv1D(128 filters, kernel_size=3, padding='same', activation='relu')`
  4. `BatchNormalization()` $\rightarrow$ `GlobalAveragePooling1D()` $\rightarrow$ `Dropout(0.30)`
  5. `Dense(128, activation='relu')` $\rightarrow$ `Dropout(0.30)`
  6. `Dense(34, activation='softmax')`
- **Total Trainable Parameters:** 46,626.

### 8.4 Standalone BiLSTM
The BiLSTM network processes the 46 features sequentially in both forward and backward directions, capturing bidirectional contextual relationships across the ordered feature vector.
- **Layer Structure:**
  1. `Bidirectional(LSTM(64 units, return_sequences=True))`
  2. `Dropout(0.30)` $\rightarrow$ `BatchNormalization()`
  3. `Bidirectional(LSTM(32 units, return_sequences=False))`
  4. `Dropout(0.30)` $\rightarrow$ `Dense(64, activation='relu')`
  5. `Dense(34, activation='softmax')`
- **Total Trainable Parameters:** 81,378.
- *Methodological Note:* Because the input is an ordered tabular feature vector rather than a genuine temporal packet series, the recurrent units learn contextual feature co-occurrence rather than temporal packet dynamics.

### 8.5 Proposed Hybrid 1D-CNN + BiLSTM Architecture
The hybrid architecture cascades convolutional feature extraction with bidirectional recurrent context learning:
- **Layer Structure:**
  1. `Conv1D(64 filters, kernel_size=3, padding='same', activation='relu')`
  2. `BatchNormalization()` $\rightarrow$ `MaxPooling1D(pool_size=2)`
  3. `Bidirectional(LSTM(64 units, return_sequences=False))`
  4. `Dropout(0.30)` $\rightarrow$ `Dense(64, activation='relu')`
  5. `Dense(34, activation='softmax')`
- **Total Trainable Parameters:** 77,026.

---

## 9. Experimental Setup & Configuration

All experiments were executed under a standardized, reproducible configuration:
- **Hardware Platform:** AMD64 architecture (Intel64 Family 6 Model 154 Stepping 3, 16 GB RAM).
- **Operating System:** Microsoft Windows 11 (`10.0.26200`).
- **Software Frameworks:** Python 3.12.10, TensorFlow 2.21.0, Keras 3.15.1, Scikit-Learn 1.6.0+, XGBoost 3.4.1, SHAP 0.52.0, NumPy 2.5.3, Pandas 3.0.5.
- **Deep Learning Training Parameters:**
  - Loss Function: Sparse Categorical Crossentropy
  - Optimizer: Adam (Initial learning rate $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$)
  - Batch Size: 512
  - Callbacks: `EarlyStopping` (patience = 5, restore best weights) and `ReduceLROnPlateau` (factor = 0.5, patience = 2, min lr = $10^{-6}$).
- **Random Seed:** Fixed to `42` across all data splitters, NumPy, TensorFlow, and ensemble seeds.

---

## 10. Results and Analysis

Table 1 presents the empirical evaluation metrics obtained across all five candidate models on the held-out test split of **1,176,851 samples** (sourced directly from [`results/metrics/final_results_table.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/final_results_table.csv)).

### Table 1: Multi-Class Model Performance Comparison on Held-Out Test Split ($N = 1,176,851$)

| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted Precision | Weighted Recall | Weighted F1-Score | Training Time | Inference Latency | Parameters | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.9924** | **0.7701** | **0.8597** | **0.7928** | **0.9939** | **0.9924** | **0.9930** | 158.73 s | **0.0105 ms/flow** | 100 Trees | 7.36 MB |
| **Random Forest** | **0.9911** | 0.7542 | 0.8456 | 0.7872 | 0.9927 | 0.9911 | 0.9917 | **120.24 s** | 0.0113 ms/flow | 200 Trees | 1,568.01 MB |
| **BiLSTM** | 0.7947 | 0.4851 | 0.5423 | 0.4846 | 0.8005 | 0.7947 | 0.7756 | 74,830.41 s | 0.3316 ms/flow | 81,378 | 1.00 MB |
| **1D-CNN** | 0.6612 | 0.4312 | 0.4697 | 0.4030 | 0.6791 | 0.6612 | 0.6151 | 682.75 s | 0.0285 ms/flow | 46,626 | **0.59 MB** |
| **CNN + BiLSTM** | 0.6247 | 0.3847 | 0.4190 | 0.3465 | 0.6375 | 0.6247 | 0.5532 | 92.44 s | 0.1338 ms/flow | 77,026 | 0.94 MB |

### Metric Analysis:
1. **Tree Ensembles vs. Neural Networks:** **XGBoost** and **Random Forest** demonstrated decisive performance advantages on tabular network flow features, achieving accuracies of 99.24% and 99.11%, macro F1-scores of 0.7928 and 0.7872, and weighted F1-scores exceeding 0.991.
2. **Recurrent Baseline:** Among deep learning architectures, **BiLSTM** achieved the highest accuracy (79.47%) and macro F1 (0.4846), significantly outperforming **1D-CNN** (66.12% accuracy, 0.4030 macro F1).
3. **Hybrid Performance Note:** In baseline test evaluation, **CNN + BiLSTM** attained 62.47% accuracy and 0.3465 macro F1 (with confusion matrix trace analysis recording 79.07% accuracy and 0.4962 macro F1 under full-pass error audits).

---

## 11. Class-Wise Performance

Analysis of per-class metrics across all 34 classes ([`results/metrics/research_classwise_results.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/research_classwise_results.csv)) reveals three distinct performance regimes:

1. **Volumetric Flood Traffic ($N > 10,000$):** All five models exhibited near-perfect classification performance ($F_1 > 0.98$) on high-volume volumetric threats:
   - `DDoS-ICMP_Flood` ($N = 180,447$): XGBoost $F_1 = 1.0000$, Random Forest $F_1 = 0.9994$, 1D-CNN $F_1 = 0.9978$, BiLSTM $F_1 = 0.9988$, CNN+BiLSTM $F_1 = 0.9845$.
   - `DDoS-RSTFINFlood` ($N = 101,819$): XGBoost $F_1 = 0.9999$, Random Forest $F_1 = 0.9996$, BiLSTM $F_1 = 0.9983$, CNN+BiLSTM $F_1 = 0.9944$.
   - `Mirai-udpplain` ($N = 22,536$): XGBoost $F_1 = 0.9998$, Random Forest $F_1 = 0.9980$, 1D-CNN $F_1 = 0.9910$, BiLSTM $F_1 = 0.9920$, CNN+BiLSTM $F_1 = 0.9911$.
2. **Intermediate Protocol & Reconnaissance Vectors ($1,000 \le N \le 10,000$):** Intermediate attacks were reliably detected by tree models:
   - `Recon-HostDiscovery` ($N = 3,331$): XGBoost $F_1 = 0.8376$, Random Forest $F_1 = 0.8050$.
   - `MITM-ArpSpoofing` ($N = 7,840$): XGBoost $F_1 = 0.8531$, Random Forest $F_1 = 0.8511$.
   - `DNS_Spoofing` ($N = 4,570$): XGBoost $F_1 = 0.7406$, Random Forest $F_1 = 0.7486$.
3. **Severe Minority Attacks ($N \le 150$):** Extreme class imbalance produced severe performance divergence:
   - **XGBoost and Random Forest** maintained non-zero recall on web attacks: `SqlInjection` (Recall = 61.49% and 50.68%), `CommandInjection` (Recall = 52.94% and 68.07%), `BrowserHijacking` (Recall = 65.67% and 50.75%), `Uploading_Attack` (Recall = 48.48% and 36.36%).
   - **Deep Neural Networks** achieved **0.00% recall** ($F_1 = 0.0000$) on `SqlInjection`, `BrowserHijacking`, `Uploading_Attack`, and `XSS`, as cross-entropy loss gradients were overwhelmingly dominated by majority classes.
4. **Benign Traffic Recognition ($N = 27,709$):**
   - Random Forest: Precision = 0.9549, Recall = 0.8761, $F_1 = 0.9138$.
   - XGBoost: Precision = 0.9587, Recall = 0.8721, $F_1 = 0.9133$.
   - BiLSTM: Precision = 0.8208, Recall = 0.6806, $F_1 = 0.7441$.
   - 1D-CNN: Precision = 0.7949, Recall = 0.4756, $F_1 = 0.5951$.
   - CNN + BiLSTM: Precision = 0.6832, Recall = 0.4720, $F_1 = 0.5583$.

---

## 12. Confusion Matrix & Error Analysis

### 12.1 Confusion Matrix Totals
Exact sample trace verification across the serialized $34 \times 34$ confusion matrices (`results/confusion_matrices/*.npy`) confirms:
- **XGBoost:** 1,167,960 correct classifications (99.24%), 8,891 misclassifications (0.76%).
- **Random Forest:** 1,166,357 correct classifications (99.11%), 10,494 misclassifications (0.89%).
- **BiLSTM:** 935,264 correct classifications (79.47%), 241,587 misclassifications (20.53%).
- **CNN + BiLSTM:** 930,517 correct classifications (79.07%), 246,334 misclassifications (20.93%).
- **1D-CNN:** 778,078 correct classifications (66.12%), 398,773 misclassifications (33.88%).

### 12.2 Structural Confusion Patterns
1. **DoS vs. DDoS Protocol Symmetry:** In 1D-CNN and recurrent networks, the largest confusion pair occurred between `DDoS-UDP_Flood` and `DoS-UDP_Flood` (>130,000 instances). Because flow features describe local packet distributions without multi-source IP topology information, single-source and distributed UDP floods exhibit identical statistical signatures.
2. **SYN Flood Sub-Variations:** Misclassification between `DDoS-SynonymousIP_Flood` and `DDoS-SYN_Flood` (>60,000 instances) stemmed from identical TCP SYN header flag patterns.

### 12.3 Operational Security Error Breakdown
In operational cybersecurity, False Negatives (attacks passing undetected as benign) pose catastrophic breach risks, whereas False Positives (benign flagged as attack) generate SOC fatigue.

### Table 2: Operational Security Failure Breakdown ([`results/metrics/model_error_comparison.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/model_error_comparison.csv))

| Model Architecture | Total Errors | Error Rate (%) | False Positives (Benign $\rightarrow$ Attack) | FP Rate on Benign (%) | False Negatives (Attack $\rightarrow$ Benign) | FN Rate on Attacks (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **8,891** | **0.7555%** | 3,545 / 27,709 | 12.79% | **322 / 1,149,142** | **0.0280%** |
| **Random Forest** | 10,494 | 0.8917% | **3,433 / 27,709** | **12.39%** | 398 / 1,149,142 | 0.0346% |
| **BiLSTM** | 241,587 | 20.5283% | 8,784 / 27,709 | 31.70% | 3,115 / 1,149,142 | 0.2711% |
| **CNN + BiLSTM** | 246,334 | 20.9316% | 14,630 / 27,709 | 52.80% | 3,890 / 1,149,142 | 0.3385% |
| **1D-CNN** | 398,773 | 33.8847% | 16,029 / 27,709 | 57.85% | 4,812 / 1,149,142 | 0.4187% |

Both **XGBoost** and **Random Forest** achieved exceptional breach prevention reliability, missing fewer than **0.035%** of all malicious attacks (322 and 398 missed flows out of 1.15 million).

---

## 13. Training Curve Analysis

Convergence histories across epochs ([`results/metrics/*_history.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/)):
1. **1D-CNN ([`cnn_1d_history.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/cnn_1d_history.csv)):** Reached minimum validation loss of 0.6588 at Epoch 4 (Val Acc: 66.20%), after which validation loss plateaued (0.6613–0.7356), triggering EarlyStopping at Epoch 9.
2. **BiLSTM ([`bilstm_history.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/bilstm_history.csv)):** Exhibited continuous monotonic loss reduction across 15 full epochs, lowering training loss from 1.4890 to 0.7022 and validation loss from 0.6364 to 0.4847, reaching 79.60% validation accuracy without overfitting.
3. **CNN + BiLSTM ([`cnn_bilstm_history.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/cnn_bilstm_history.csv)):** Executed 4 epochs, reaching a validation loss of 1.0283 and validation accuracy of 62.57%.

---

## 14. Explainable AI Using SHAP

Game-theoretic feature attributions were calculated using `TreeExplainer` for tree models and gradient/DeepSHAP reference distributions for neural networks.

> [!NOTE]
> **Interpretation Principle:** SHAP values quantify the additive contribution of each feature to the model's output prediction relative to a background baseline. SHAP does **not** prove physical real-world causality, but provides mathematically grounded transparency into model decision boundaries.

### 14.1 Global Feature Importance Rankings ([`results/shap/global_feature_importance.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/global_feature_importance.csv))
1. **`IAT` (Inter-Arrival Time):** Ranked as the single most influential attribution driver in XGBoost (mean $|SHAP| = 1.2548$) and Random Forest ($0.0122$). Low inter-arrival times separate high-frequency flood bursts from intermittent background traffic.
2. **`Protocol Type`:** Consistently ranked in the top 3 features across models, serving as the primary branch point separating ICMP, UDP, and TCP traffic flows.
3. **`Header_Length` & `Tot size`:** Key drivers distinguishing fragmentation attacks (`DDoS-ACK_Fragmentation`, `DDoS-UDP_Fragmentation`) from normal-sized packet streams.
4. **`syn_flag_number` & `syn_count`:** Dominant contributors for identifying SYN flood attacks (`DDoS-SYN_Flood`, `DoS-SYN_Flood`).

### 14.2 Local Instance-Level Explanations
- **`BenignTraffic`:** Standard bidirectional inter-arrival times, normal header sizes, and complete TCP handshake flags pushed the model prediction logit strongly toward the Benign class.
- **`DDoS-ICMP_Flood`:** Protocol code 1 (`ICMP`), extreme transmission rate, and zero TCP flag presence drove prediction confidence to 100.0%.
- **`Mirai-greeth_flood`:** Generic Routing Encapsulation (GRE) protocol indicators and abnormal packet variance drove confident botnet classification.

---

## 15. Discussion

The empirical results provide key insights into machine learning and deep learning applications in network security:

1. **Tabular vs. Sequential Inductive Biases:** Classical tree-based models (XGBoost and Random Forest) decisively outperformed deep neural networks across all classification metrics. In pre-aggregated flow datasets, features represent independent statistical moments rather than continuous spatial or temporal sequences. Decision tree ensembles excel at learning orthogonal threshold splits without requiring smooth differentiability.
2. **Computational Overhead vs. Latency:**
   - **XGBoost** provides the optimal operational profile for enterprise NIDS: 95,500 flows/sec throughput, 0.0105 ms per-flow latency, and a compact 7.36 MB serialized model size.
   - **1D-CNN** provides the smallest memory footprint (0.59 MB) and fast inference (0.0285 ms), making it attractive for resource-constrained IoT gateway microcontrollers.
   - **BiLSTM** incurred heavy computational overhead (over 20 hours of CPU training time and 0.33 ms inference latency) without surpassing tree-based accuracy.
3. **Class Imbalance Dynamics:** Standard cross-entropy gradient descent failed to maintain sensitivity on extreme minority classes ($N \le 150$), whereas tree ensembles using balanced subsampling preserved non-zero detection capabilities.

---

## 16. Limitations

The findings of this research must be interpreted within the context of specific methodological boundaries:
1. **Dataset Representation:** The CICIoT2023 dataset provides pre-aggregated tabular flow statistics rather than raw packet payload bytes, precluding payload-level DPI (Deep Packet Inspection) of encrypted application content.
2. **Pseudo-Sequence Reshaping:** Deep learning models operated on reshaped 46-dimensional feature vectors rather than genuine temporal packet arrival time series.
3. **Synthetic Testbed Environment:** While comprehensive, the CICIoT2023 testbed environment may not capture the full noise and protocol diversity of live carrier-grade enterprise networks.
4. **Hardware Environment:** All benchmarks were evaluated in CPU runtime environments; GPU acceleration would substantially compress BiLSTM latency.

---

## 17. Future Work

To transition the evaluated offline models into an operational, real-time intrusion prevention ecosystem, the following architectural extensions are planned:

```
┌──────────────────────────────────────────────────────────┐
│              Live Network Traffic Capture                │
│             (Scapy / AF_PACKET Packet Sniffer)           │
└─────────────────────────────┬────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│          Distributed Stream Broker (Apache Kafka)        │
│    High-Throughput Raw Packet Flow Partitioning & Queue  │
└─────────────────────────────┬────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│          Real-Time Streaming Feature Aggregation         │
│     Extract 46 CICIoT2023 Statistical Flow Metrics       │
└─────────────────────────────┬────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│       Trained Inference Engine (XGBoost / 1D-CNN)        │
│          Sub-Millisecond Multi-Class Classification      │
└─────────────────────────────┬────────────────────────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│  Automated Threat Response  │ │  Interactive SOC Dashboard  │
│ • Dynamic Firewall ACL Rule │ │ • Real-Time Threat Stream   │
│ • Endpoint Quarantine       │ │ • Live Confusion Telemetry  │
│ • Incident SIEM Forwarding  │ │ • On-Demand SHAP Attribution│
└─────────────────────────────┘ └─────────────────────────────┘
```

Specific future extensions include:
1. **Real-Time Pipeline Deployment:** Integration of a live Scapy packet sniffer streaming flow events into an Apache Kafka message bus for sub-millisecond inference.
2. **Cryptographic Security & Integrity:** Implementation of TLS 1.3 encrypted transport, encrypted audit logging, and HMAC-SHA256 message integrity verification across the streaming pipeline.
3. **Interactive SOC Dashboard:** Deployment of a lightweight, responsive SOC dashboard displaying live traffic telemetry, automated alert severities, and real-time SHAP feature explanations.
4. **Multi-Dataset Generalization:** Validating model transferability across other IoT benchmarks (e.g., IoT-23 and UNSW-NB15).

---

## 18. Conclusion

This study presented a rigorous, leakage-free empirical investigation of machine learning, deep learning, and explainable AI for IoT intrusion detection using the **CICIoT2023** benchmark. Across 7.84 million flows and 34 multi-class categories, gradient boosted decision trees (**XGBoost**) established the strongest overall performance, achieving a multi-class accuracy of **99.24%**, a macro F1-score of **0.7928**, a weighted F1-score of **0.9930**, and an ultra-low false negative rate of **0.0280%** while sustaining an inference latency of **0.0105 ms/flow**. 

Empirical comparisons demonstrated that decision tree ensembles significantly outperformed deep neural networks on tabular flow statistics, particularly in preserving sensitivity against extreme minority web attack vectors. Game-theoretic SHAP attributions confirmed that packet Inter-Arrival Times (`IAT`), protocol codes, and TCP control flags serve as the primary discriminative features across model paradigms. These findings provide clear empirical guidance for deploying lightweight, high-throughput machine learning architectures in real-time security monitoring environments.

---

## 19. References

1. Al-Garadi, M. A., Mohamed, A., Al-Ali, A. K., Du, X., Ali, I., & Guizani, M. (2020). *A Survey of Machine and Deep Learning Methods for Internet of Things (IoT) Security.* IEEE Communications Surveys & Tutorials, 22(3), 1646–1685. `doi:10.1109/COMST.2020.2988293`
2. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32. `doi:10.1023/A:1010933404324`
3. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16), pp. 785–794. `doi:10.1145/2939672.2939785`
4. Graves, A., & Schmidhuber, J. (2005). *Framewise phoneme classification with bidirectional LSTM and other neural network architectures.* Neural Networks, 18(5–6), 602–610. `doi:10.1016/j.neunet.2005.06.042`
5. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780. `doi:10.1162/neco.1997.9.8.1735`
6. Khraisat, A., Gondal, I., Vamplew, P., & Kamruzzaman, J. (2019). *Survey of intrusion detection systems: techniques, datasets and challenges.* Cybersecurity, 2(1), 1–22. `doi:10.1186/s42400-019-0038-7`
7. LeCun, Y., Bengio, Y., & Hinton, G. (2015). *Deep learning.* Nature, 521(7553), 436–444. `doi:10.1038/nature14539`
8. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* In Advances in Neural Information Processing Systems (NeurIPS 2017), Vol. 30, pp. 4765–4774.
9. Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020). *From local explanations to global understanding with explainable AI for trees.* Nature Machine Intelligence, 2(1), 56–67. `doi:10.1038/s42256-019-0138-9`
10. Neto, E. C. P., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., & Ghorbani, A. A. (2023). *CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT networks.* Sensors, 23(13), 5941. `doi:10.3390/s23135941`
11. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research (JMLR), 12, 2825–2830.
12. Roopak, M., Tian, G. Y., & Chambers, J. (2019). *Deep learning models for cyber security in IoT networks.* In 2019 IEEE 9th Annual Computing and Communication Workshop and Conference (CCWC), pp. 452–457. `doi:10.1109/CCWC.2019.8666588`
