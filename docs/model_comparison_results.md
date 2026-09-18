# Model Comparison Results and Analysis

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Data Sources:** [`results/metrics/final_model_comparison.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/final_model_comparison.csv), [`results/metrics/model_error_comparison.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/model_error_comparison.csv)  
**Evaluated Scope:** Five Candidate Machine Learning and Deep Learning Models

---

## 1. Multi-Dimensional Comparative Framework

The experimental design evaluates five candidate architectures under identical data splits ($N = 1,176,851$ test flows, 46 standardized tabular features, 34 target classes). This section provides an objective comparison across four operational dimensions: Performance, Computational Efficiency, Error Behavior, and Explainability.

---

## 2. Performance Comparison

Table 1 presents the classification performance metrics across all candidate models.

### Table 1: Classification Performance Summary

| Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.9911 | 0.7542 | 0.8456 | 0.7872 | 0.9927 | 0.9911 | 0.9917 |
| **XGBoost** | 0.9924 | 0.7701 | 0.8597 | 0.7928 | 0.9939 | 0.9924 | 0.9930 |
| **1D-CNN** | 0.6612 | 0.4312 | 0.4697 | 0.4030 | 0.6791 | 0.6612 | 0.6151 |
| **BiLSTM** | 0.7947 | 0.4851 | 0.5423 | 0.4846 | 0.8005 | 0.7947 | 0.7756 |
| **CNN + BiLSTM** | 0.6247 | 0.3847 | 0.4190 | 0.3465 | 0.6375 | 0.6247 | 0.5532 |

- **Observations:**
  - Tree-based architectures (Random Forest and XGBoost) achieved higher macro and weighted F1-scores across the 34 classes than the deep neural architectures on this tabular dataset.
  - Among deep learning models, BiLSTM obtained a macro F1 of 0.4846 and accuracy of 79.47%, compared to 0.4030 (66.12% accuracy) for 1D-CNN and 0.3465 (62.47% accuracy) for CNN + BiLSTM in baseline testing.

---

## 3. Computational Efficiency and Resource Footprint

Table 2 details the computational demands, training duration, inference latency, parameter counts, and serialized storage footprint.

### Table 2: Computational and Resource Comparison

| Architecture | Training Time (s) | Inference Latency (ms / flow) | Throughput (flows / sec) | Model Complexity | Serialized Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 120.24 s | 0.0113 ms | ~88,500 flows/s | 200 Decision Trees | 1,568.01 MB |
| **XGBoost** | 158.73 s | 0.0105 ms | ~95,200 flows/s | 100 Boosted Trees | 7.36 MB |
| **1D-CNN** | 682.75 s | 0.0285 ms | ~35,080 flows/s | 46,626 Parameters | 0.59 MB |
| **BiLSTM** | 74,830.41 s | 0.3316 ms | ~3,015 flows/s | 81,378 Parameters | 1.00 MB |
| **CNN + BiLSTM** | 92.44 s | 0.1338 ms | ~7,470 flows/s | 77,026 Parameters | 0.94 MB |

- **Observations:**
  - XGBoost demonstrated the lowest inference latency at 0.0105 ms per flow (highest throughput at 95,200 flows/sec).
  - 1D-CNN yielded the smallest storage footprint at 0.59 MB.
  - BiLSTM required the longest training duration (74,830 seconds on CPU) and highest inference latency (0.3316 ms per flow).
  - Random Forest occupied the largest storage footprint (1,568 MB) due to unpruned leaf structure across 200 trees.

---

## 4. Error Behavior and Security Metrics

Table 3 compares the error distributions, false alarm rates, and security leakage rates across models.

### Table 3: Error and Security Failure Mode Comparison

| Architecture | Total Test Errors | Overall Error Rate (%) | False Positives (Benign $\rightarrow$ Attack) | Benign FP Rate (%) | False Negatives (Attack $\rightarrow$ Benign) | Attack FN Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 10,494 | 0.8917% | 3,433 | 12.39% | 398 | 0.0346% |
| **XGBoost** | 8,891 | 0.7555% | 3,545 | 12.79% | 322 | 0.0280% |
| **1D-CNN** | 398,773 | 33.8847% | 16,029 | 57.85% | 4,812 | 0.4187% |
| **BiLSTM** | 241,587 | 20.5283% | 8,784 | 31.70% | 3,115 | 0.2711% |
| **CNN + BiLSTM** | 246,334 | 20.9316% | 14,630 | 52.80% | 3,890 | 0.3385% |

---

## 5. Explainability and Attribution Support

| Architecture | Explainer Applied | Primary Contributing Feature Group | Convergence / Exactness |
| :--- | :--- | :--- | :--- |
| **Random Forest** | `shap.TreeExplainer` | `IAT`, `Magnitue`, `syn_count`, `Protocol Type` | Exact game-theoretic tree attribution |
| **XGBoost** | `shap.TreeExplainer` | `IAT`, `Protocol Type`, `Header_Length`, `Tot size` | Exact game-theoretic tree attribution |
| **1D-CNN** | `shap.GradientExplainer` | `HTTPS`, `syn_flag_number`, `ICMP`, `syn_count` | Gradient-based attribution approximation |
| **BiLSTM** | `shap.GradientExplainer` | `flow_duration`, `Header_Length`, `Tot size`, `Std` | Gradient-based attribution approximation |
| **CNN + BiLSTM** | `shap.DeepExplainer` | `Srate`, `Drate`, `Rate`, `ece_flag_number` | Deep SHAP attribution approximation |
