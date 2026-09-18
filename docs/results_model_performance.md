# Overall Model Performance Results

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Dataset:** CICIoT2023 Held-Out Test Partition ($N = 1,176,851$)  
**Evaluation Scope:** 5 Candidate Machine Learning & Deep Learning Architectures

---

## 1. Summary of Observed Test Metrics

All five models were evaluated on the exact same held-out test split of 1,176,851 network flows across 34 multiclass categories. Table 1 reports the comprehensive performance and computational profile of each architecture.

### Table 1: Comprehensive Performance and Operational Profile

| Model Architecture | Multi-Class Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 | Training Time | Inference Latency | Trainable Parameters | Serialized Disk Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.9911 (99.11%) | 0.7542 | 0.8456 | 0.7872 | 0.9927 | 0.9911 | 0.9917 | 120.24 s | 0.0113 ms / flow | 200 Trees | 1,568.01 MB |
| **XGBoost** | 0.9924 (99.24%) | 0.7701 | 0.8597 | 0.7928 | 0.9939 | 0.9924 | 0.9930 | 158.73 s | 0.0105 ms / flow | 100 Trees | 7.36 MB |
| **1D-CNN** | 0.6612 (66.12%) | 0.4312 | 0.4697 | 0.4030 | 0.6791 | 0.6612 | 0.6151 | 682.75 s | 0.0285 ms / flow | 46,626 params | 0.59 MB |
| **BiLSTM** | 0.7947 (79.47%) | 0.4851 | 0.5423 | 0.4846 | 0.8005 | 0.7947 | 0.7756 | 74,830.41 s | 0.3316 ms / flow | 81,378 params | 1.00 MB |
| **CNN + BiLSTM** | 0.6247 (62.47%) | 0.3847 | 0.4190 | 0.3465 | 0.6375 | 0.6247 | 0.5532 | 92.44 s | 0.1338 ms / flow | 77,026 params | 0.94 MB |

---

## 2. Metric-by-Metric Analysis

### 2.1 Multi-Class Accuracy
- **Tree-Based Models:** XGBoost achieved an accuracy of 0.9924 (99.24%), while Random Forest achieved 0.9911 (99.11%).
- **Deep Learning Models:** BiLSTM obtained an accuracy of 0.7947 (79.47%), 1D-CNN achieved 0.6612 (66.12%), and CNN + BiLSTM achieved 0.6247 (62.47%) in baseline testing.

### 2.2 Macro-Averaged Performance (Unweighted Across 34 Classes)
Macro metrics treat all classes with equal weight regardless of sample size, reflecting model effectiveness on rare attack vectors:
- **Macro Precision:** XGBoost achieved 0.7701, Random Forest achieved 0.7542, BiLSTM reached 0.4851, 1D-CNN obtained 0.4312, and CNN + BiLSTM scored 0.3847.
- **Macro Recall:** XGBoost reached 0.8597, Random Forest reached 0.8456, BiLSTM scored 0.5423, 1D-CNN scored 0.4697, and CNN + BiLSTM scored 0.4190.
- **Macro F1-Score:** XGBoost obtained 0.7928, Random Forest obtained 0.7872, BiLSTM obtained 0.4846, 1D-CNN obtained 0.4030, and CNN + BiLSTM reached 0.3465.

### 2.3 Weighted Performance (Proportional to Class Support)
Weighted metrics account for the natural class distribution dominated by high-volume DoS/DDoS traffic:
- **Weighted Precision:** XGBoost achieved 0.9939, Random Forest achieved 0.9927, BiLSTM reached 0.8005, 1D-CNN scored 0.6791, and CNN + BiLSTM reached 0.6375.
- **Weighted Recall:** XGBoost achieved 0.9924, Random Forest achieved 0.9911, BiLSTM obtained 0.7947, 1D-CNN obtained 0.6612, and CNN + BiLSTM obtained 0.6247.
- **Weighted F1-Score:** XGBoost achieved 0.9930, Random Forest achieved 0.9917, BiLSTM reached 0.7756, 1D-CNN scored 0.6151, and CNN + BiLSTM scored 0.5532.

---

## 3. Computational Efficiency and Complexity

### 3.1 Training Time
- **Tree-Based Models:** Random Forest completed training in 120.24 seconds, while XGBoost required 158.73 seconds.
- **Deep Learning Models:** The CNN + BiLSTM architecture trained for 4 epochs in 92.44 seconds, 1D-CNN trained for 9 epochs in 682.75 seconds, and BiLSTM required 74,830.41 seconds across 15 epochs on CPU.

### 3.2 Inference Latency
- XGBoost demonstrated a per-flow test latency of **0.0105 ms** (processing approximately 95,200 flows per second).
- Random Forest exhibited a latency of **0.0113 ms** (processing approximately 88,500 flows per second).
- 1D-CNN exhibited a latency of **0.0285 ms** (processing approximately 35,080 flows per second).
- CNN + BiLSTM exhibited a latency of **0.1338 ms** (processing approximately 7,470 flows per second).
- BiLSTM exhibited a latency of **0.3316 ms** (processing approximately 3,015 flows per second).

### 3.3 Memory and Storage Footprint
- **Deep Learning Architectures:** 1D-CNN required **0.59 MB** (46,626 parameters), CNN + BiLSTM required **0.94 MB** (77,026 parameters), and BiLSTM required **1.00 MB** (81,378 parameters).
- **Tree-Based Ensembles:** XGBoost occupied **7.36 MB** for 100 boosted trees, whereas Random Forest occupied **1,568.01 MB** for 200 unpruned trees.
