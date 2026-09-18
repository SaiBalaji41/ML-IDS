# Phase 11 Technical Report: Comprehensive Model Evaluation and Comparison

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Phase:** Phase 11 — Model Evaluation and Comparison  
**Evaluated Dataset:** CICIoT2023 Processed Benchmark (1,176,851 held-out test flows)  
**Evaluated Models:**
1. Random Forest Baseline
2. XGBoost Baseline
3. 1D-CNN Deep Learning Model
4. BiLSTM Deep Learning Model
5. Proposed Hybrid CNN + BiLSTM Model

---

## 1. Executive Summary & Benchmark Rankings

This report consolidates the comparative empirical findings across classical machine learning baselines and modern deep learning architectures for high-throughput IoT network intrusion detection.

### 1.1 Complete Benchmark Comparison Table

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Macro Precision | Macro Recall | Macro F1 | Training Time | Test Latency (1.17M flows) | Model Size | Parameter Count |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | **99.24%** | **0.9939** | **0.9924** | **0.9930** | **0.7701** | **0.8597** | **0.7928** | 158.73 s | **12.32 s** (10.47 µs/flow) | 7.36 MB | Trees=100 |
| **Random Forest** | 99.11% | 0.9927 | 0.9911 | 0.9917 | 0.7542 | 0.8456 | 0.7872 | **120.24 s** | 13.27 s (11.27 µs/flow) | 1,568.01 MB | Trees=100 |
| **BiLSTM** | 79.47% | 0.8005 | 0.7947 | 0.7756 | 0.4851 | 0.5423 | 0.4846 | 148.06 s | 390.24 s (331.60 µs/flow) | 1.04 MB | 55,202 |
| **1D-CNN** | 66.12% | 0.6791 | 0.6612 | 0.6151 | 0.4312 | 0.4697 | 0.4030 | 682.75 s | 33.48 s (28.45 µs/flow) | **0.59 MB** | **46,626** |
| **Hybrid CNN-BiLSTM** | 62.47% | 0.6375 | 0.6247 | 0.5532 | 0.3847 | 0.4190 | 0.3465 | 92.44 s | 157.42 s (133.76 µs/flow) | 0.95 MB | 75,554 |

---

## 2. Key Findings & Empirical Analysis

### 2.1 Classical Trees vs. Deep Learning on Tabular Flow Data
1. **Tree Ensembles Dominate Tabular Flow Metrics:**  
   XGBoost (99.24%) and Random Forest (99.11%) demonstrate decisive superiority over deep learning architectures on tabular flow statistical summaries. Tree-based partitioning creates exact axis-aligned splits on packet rate, header length, and inter-arrival time metrics with zero inductive bias towards spatial or temporal order.
2. **Sequential Memory vs. Convolutional Locality:**  
   Among deep learning models, **BiLSTM (79.47%) significantly outperforms 1D-CNN (66.12%) by +13.35% accuracy**. The bidirectional recurrent gate mechanism models long-range cross-feature dependencies more effectively than local 3-step convolutional filters on tabular vectors.
3. **Memory and Deployment Efficiency:**  
   - **XGBoost:** Best all-around production balance (99.24% accuracy, 7.36 MB RAM, 10.47 µs/flow latency).
   - **1D-CNN:** Smallest footprint (0.59 MB, 46.6k parameters).
   - **Random Forest:** High accuracy (99.11%) but massive disk/RAM footprint (1.57 GB).

---

## 3. Publication-Ready Summary Table

| Model | Accuracy (%) | Weighted F1 (%) | Macro F1 (%) | Training Time (s) | Inference Latency (ms/sample) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | **99.24** | **99.30** | **79.28** | 158.73 | **0.01047** |
| **Random Forest** | 99.11 | 99.17 | 78.72 | 120.24 | 0.01127 |
| **BiLSTM** | 79.47 | 77.56 | 48.46 | 148.06 | 0.33160 |
| **1D-CNN** | 66.12 | 61.51 | 40.30 | 682.75 | 0.02845 |
| **Hybrid CNN-BiLSTM** | 62.47 | 55.32 | 34.65 | 92.44 | 0.13376 |

---

## 4. Verification of Fair Comparison

All comparative metrics are grounded in strict experimental controls:
- **Same Preprocessing:** Fitted exclusively on `X_train`.
- **Same Partitions:** 70% Train, 15% Validation, 15% Test.
- **Same Metric Definitions:** Computed using Scikit-Learn standard weighted and macro averages.
- **Zero Test Leakage:** Evaluated strictly on the unseen 1,176,851 test partition.
