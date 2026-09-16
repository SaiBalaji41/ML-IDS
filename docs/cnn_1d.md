# Phase 8 Technical Report: 1D-CNN Deep Learning Model

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Phase:** Phase 8 — 1D-CNN Deep Learning Model  
**Target Dataset:** CICIoT2023  
**Execution Date:** 2026-09-16  
**Status:** Completed & Empirically Verified  

---

## 1. Executive Summary

This report documents the implementation, training, and empirical evaluation of the **1D Convolutional Neural Network (1D-CNN)** deep learning baseline for the ML-Powered Intrusion Detection System.

- **Classification Type:** Multiclass Intrusion Detection (34 distinct classes: 1 Benign Traffic class + 33 IoT Attack variants).
- **Deep Learning Framework:** TensorFlow `2.21.0` / Keras `3.15.1`
- **Dataset Source:** Verified preprocessed numeric feature partitions (`data/processed/`).
- **Input Dimension:** 46 standardized continuous and discrete network flow statistical features.
- **Input Tensor Shape:** `[batch_size, 46, 1]` (ordered tabular feature vector with 1 channel).
- **Training Samples:** 500,408 stratified samples (from 5,491,971 available training records).
- **Validation Samples:** 1,176,851 records (15.0% stratified partition).
- **Test Samples:** 1,176,851 records (15.0% stratified partition).
- **Total Parameters:** 46,626 trainable parameters.
- **Model Size:** 0.59 MB (`models/cnn_1d/best_model.keras`).
- **Epochs Completed:** 9 (EarlyStopping triggered with patience=5; optimal weights restored from Epoch 4).
- **Test Accuracy:** **66.12%** (778,078 correctly classified test flows out of 1,176,851).
- **Test Weighted F1-Score:** **0.6151**
- **Test Macro F1-Score:** **0.4030**
- **Test Macro Recall:** **0.4697**
- **Test Macro Precision:** **0.4312**
- **Training Wall-Clock Time:** **682.75 seconds**.
- **Test Inference Latency:** **33.4847 seconds** (~28.45 microseconds per network flow).

---

## 2. Dataset Representation & Sequence Formulation

### 2.1 Dataset Representation Verification
The dataset consists of tabular network flow statistical summaries (e.g., inter-arrival times, header flags, flow durations, packet counts).

> [!IMPORTANT]
> **Key Architectural Finding & Limitation Regarding Sequence Representation:**  
> The 1D-CNN treats the **ordered tabular feature vector as a 1D sequence** of shape `(46, 1)`.  
> The 1D convolution kernel operates across adjacent columns in the feature array.  
> **This does NOT represent raw packet-payload bytes, byte-level PCAP streams, or a true physical temporal sequence.**  
> The sequence order is determined by the arbitrary ordering of tabular feature columns. Consequently, 1D convolutional filters attempt to learn local relationships between arbitrarily adjacent statistical features, which explains why tree-based ensembles (Random Forest at 99.11% and XGBoost at 99.24%) significantly outperform standalone 1D-CNNs on this tabular representation.

---

## 3. Model Architecture & Hyperparameter Configuration

The 1D-CNN is implemented in [src/models/cnn_1d.py](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/src/models/cnn_1d.py):

```
Input Tensor: (batch_size, 46, 1)
  ↓
Conv1D (64 filters, kernel_size=3, padding="same")
  ↓
Batch Normalization
  ↓
ReLU Activation
  ↓
MaxPooling1D (pool_size=2)
  ↓
Spatial Dropout (0.30)
  ↓
Conv1D (128 filters, kernel_size=3, padding="same")
  ↓
Batch Normalization
  ↓
ReLU Activation
  ↓
MaxPooling1D (pool_size=2)
  ↓
Spatial Dropout (0.30)
  ↓
GlobalAveragePooling1D (reduces tensor from [batch, 11, 128] to [batch, 128])
  ↓
Dense (128 units, ReLU)
  ↓
Dropout (0.30)
  ↓
Dense (34 units, Softmax)
```

| Hyperparameter | Value | Rationale |
|:---|:---:|:---|
| `conv1d_1 filters` | 64 | Extracts initial low-level cross-feature patterns. |
| `conv1d_2 filters` | 128 | Expands channel capacity for higher-level feature combinations. |
| `kernel_size` | 3 | Spans 3 adjacent feature elements in the ordered vector. |
| `pooling` | `GlobalAveragePooling1D` | Drastically reduces parameter count (to 46.6k) compared to Flatten (>150k), preventing severe overfitting. |
| `dense_units` | 128 | Non-linear classification head before output projection. |
| `dropout` | 0.30 | Regularizes both convolutional feature maps and dense projections. |
| `optimizer` | Adam ($\eta = 0.001$) | Standard adaptive gradient descent. |
| `loss` | Sparse Categorical Crossentropy | Multi-class cross-entropy over 34 mutually exclusive integer class indices. |
| `batch_size` | 512 | Balances gradient stability and vectorization throughput on CPU/GPU. |

---

## 4. Class Imbalance Handling

Balanced class weights were calculated **exclusively on the training partition**:
$$w_c = \frac{N_{\text{train}}}{K \cdot N_c}$$
Weights were clipped to the range $[0.2, 50.0]$ to prevent gradient explosion from ultra-rare classes (`Uploading_Attack`, `Recon-PingSweep`). Validation and test distributions were preserved in their natural, unaltered state.

---

## 5. Training Procedure & Convergence Trajectory

- **Training Set:** 500,408 samples.
- **Validation Monitoring:** `val_loss` evaluated at each epoch.
- **Callbacks:**
  - `EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)`: Triggered at Epoch 9, restoring optimal weights from **Epoch 4**.
  - `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)`: Halved learning rate to $0.0005$ at Epoch 6 and $0.00025$ at Epoch 8.
  - `ModelCheckpoint`: Saved best checkpoint to `models/cnn_1d/best_model.keras`.

---

## 6. Empirical Evaluation Results

### 6.1 Validation vs. Test Metrics

| Partition | Samples | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 | Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Validation** | 1,176,851 | 66.20% | 0.4325 | 0.4718 | 0.4042 | 0.6802 | 0.6620 | 0.6157 | 35.6882s |
| **Test** | 1,176,851 | 66.12% | 0.4312 | 0.4697 | 0.4030 | 0.6791 | 0.6612 | 0.6151 | 33.4847s |

---

## 7. Complete 34-Class Breakdown (Test Set)

| Class Index | Class Name | Precision | Recall | F1-Score | Support |
|:---:|:---|:---:|:---:|:---:|:---:|
| 0 | `Backdoor_Malware` | 0.0000 | 0.0000 | 0.0000 | 89 |
| 1 | `BenignTraffic` | 0.7949 | 0.4756 | 0.5951 | 27,709 |
| 2 | `BrowserHijacking` | 0.0000 | 0.0000 | 0.0000 | 134 |
| 3 | `CommandInjection` | 0.0680 | 0.1176 | 0.0862 | 119 |
| 4 | `DDoS-ACK_Fragmentation` | 0.9262 | 0.9299 | 0.9281 | 7,292 |
| 5 | `DDoS-HTTP_Flood` | 0.2602 | 0.7024 | 0.3797 | 709 |
| 6 | `DDoS-ICMP_Flood` | 0.9999 | 0.9957 | 0.9978 | 180,447 |
| 7 | `DDoS-ICMP_Fragmentation` | 0.9988 | 0.9420 | 0.9696 | 11,402 |
| 8 | `DDoS-PSHACK_Flood` | 0.9839 | 0.9814 | 0.9827 | 103,326 |
| 9 | `DDoS-RSTFINFlood` | 1.0000 | 0.9951 | 0.9975 | 101,819 |
| 10 | `DDoS-SYN_Flood` | 0.4521 | 0.8010 | 0.5780 | 102,208 |
| 11 | `DDoS-SlowLoris` | 0.1366 | 0.1817 | 0.1560 | 622 |
| 12 | `DDoS-SynonymousIP_Flood` | 0.5378 | 0.3025 | 0.3872 | 90,480 |
| 13 | `DDoS-TCP_Flood` | 0.6452 | 0.8529 | 0.7347 | 113,735 |
| 14 | `DDoS-UDP_Flood` | 0.4771 | 0.0311 | 0.0585 | 136,717 |
| 15 | `DDoS-UDP_Fragmentation` | 0.9465 | 0.9740 | 0.9600 | 7,224 |
| 16 | `DNS_Spoofing` | 0.2625 | 0.3575 | 0.3028 | 4,570 |
| 17 | `DictionaryBruteForce` | 0.3539 | 0.1975 | 0.2535 | 319 |
| 18 | `DoS-HTTP_Flood` | 0.3114 | 0.7247 | 0.4356 | 1,805 |
| 19 | `DoS-SYN_Flood` | 0.5010 | 0.1078 | 0.1774 | 51,116 |
| 20 | `DoS-TCP_Flood` | 0.4441 | 0.2007 | 0.2765 | 67,697 |
| 21 | `DoS-UDP_Flood` | 0.3716 | 0.9358 | 0.5320 | 83,627 |
| 22 | `MITM-ArpSpoofing` | 0.5288 | 0.2847 | 0.3701 | 7,840 |
| 23 | `Mirai-greeth_flood` | 0.3684 | 0.0093 | 0.0181 | 24,934 |
| 24 | `Mirai-greip_flood` | 0.4333 | 0.9674 | 0.5985 | 19,279 |
| 25 | `Mirai-udpplain` | 0.9930 | 0.9889 | 0.9910 | 22,536 |
| 26 | `Recon-HostDiscovery` | 0.3830 | 0.2594 | 0.3093 | 3,331 |
| 27 | `Recon-OSScan` | 0.0800 | 0.0008 | 0.0016 | 2,433 |
| 28 | `Recon-PingSweep` | 0.0469 | 0.1132 | 0.0663 | 53 |
| 29 | `Recon-PortScan` | 0.0767 | 0.6662 | 0.1375 | 2,082 |
| 30 | `SqlInjection` | 0.0000 | 0.0000 | 0.0000 | 148 |
| 31 | `Uploading_Attack` | 0.0000 | 0.0000 | 0.0000 | 33 |
| 32 | `VulnerabilityScan` | 0.2780 | 0.8740 | 0.4219 | 913 |
| 33 | `XSS` | 0.0000 | 0.0000 | 0.0000 | 103 |

---

## 8. Confusion Matrix & Misclassification Findings

- **Total Test Misclassifications:** 398,773 samples (33.88% error rate).
- **Major Confusion Clusters:**
  1. `DDoS-UDP_Flood` $\rightarrow$ `DoS-UDP_Flood` (124,310 flows): Both attack types share identical UDP header metrics; the 1D convolution lacks the tree split precision to separate packet rate magnitude.
  2. `DoS-TCP_Flood` $\rightarrow$ `DDoS-TCP_Flood` (48,129 flows): TCP flood sub-types with identical TCP flag distributions.
  3. `DoS-SYN_Flood` $\rightarrow$ `DDoS-SYN_Flood` (42,875 flows): SYN packet flag combinations.
  4. `BenignTraffic` $\rightarrow$ `Recon-PortScan` (6,211 flows).

---

## 9. Direct Benchmark Comparison: RF vs. XGBoost vs. 1D-CNN

| Metric | Random Forest (Phase 6) | XGBoost (Phase 7) | 1D-CNN (Phase 8) | Analysis |
|:---|:---:|:---:|:---:|:---|
| **Test Accuracy** | 99.11% | **99.24%** | 66.12% | Tree ensembles decisively outperform 1D-CNN on tabular flow data. |
| **Weighted F1** | 0.9917 | **0.9930** | 0.6151 | 1D-CNN struggles with overlapping UDP/TCP flood distinctions. |
| **Macro F1** | 0.7872 | **0.7928** | 0.4030 | Tree models handle extreme class imbalance significantly better. |
| **Training Time** | **120.24 s** | 158.73 s | 682.75 s | Neural network backpropagation on CPU/GPU takes longer. |
| **Inference Time** | 13.27 s | **12.32 s** | 33.48 s | Feedforward batch inference is slower than compiled C++ tree traversal. |
| **Model Size** | 1,568.01 MB | 7.36 MB | **0.59 MB** | 1D-CNN produces the most compact model binary (46.6k parameters). |

---

## 10. Summary of Generated Artifacts

- **Model Checkpoint:** `models/cnn_1d/best_model.keras` (0.59 MB)
- **Metadata JSON:** `models/cnn_1d/metadata.json`
- **Training History CSV:** `results/metrics/cnn_1d_history.csv`
- **Training Curves Plot:** `results/graphs/cnn_1d_training_history.png`
- **Validation Metrics:** `results/metrics/cnn_1d_validation.json`
- **Test Metrics:** `results/metrics/cnn_1d_test.json`
- **Per-Class Classification Report:** `results/metrics/cnn_1d_classification_report.csv`
- **Confusion Matrix Plot:** `results/confusion_matrices/cnn_1d_confusion_matrix.png`
- **Misclassifications CSV:** `results/metrics/cnn_1d_misclassifications.csv`
- **Updated Comparison Table:** `results/metrics/model_comparison.csv`
- **Training Script:** `scripts/train_cnn_1d.py`
- **Interactive Notebook:** `notebooks/06_1d_cnn.ipynb`
- **Test Suite:** `tests/test_cnn_1d.py` (8 tests, all passing)
