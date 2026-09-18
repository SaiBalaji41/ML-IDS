# Phase 9 Technical Report: BiLSTM Deep Learning Model

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Phase:** Phase 9 — BiLSTM Deep Learning Model  
**Target Dataset:** CICIoT2023  
**Execution Date:** 2026-09-17  
**Status:** Completed & Empirically Verified  

---

## 1. Executive Summary

This report documents the implementation, training, and empirical evaluation of the **Bidirectional Long Short-Term Memory (BiLSTM)** deep learning model for the ML-Powered Intrusion Detection System.

- **Classification Type:** Multiclass Intrusion Detection (34 distinct classes: 1 Benign Traffic class + 33 IoT Attack variants).
- **Deep Learning Framework:** TensorFlow `2.21.0` / Keras `3.15.1`
- **Dataset Source:** Verified preprocessed numeric feature partitions (`data/processed/`).
- **Input Dimension:** 46 standardized continuous and discrete network flow statistical features.
- **Input Tensor Shape:** `[batch_size, 46, 1]` (ordered tabular feature vector treated as a sequential input with 1 channel).
- **Training Samples:** 300,000 stratified samples (from 5,491,971 available training records).
- **Validation Samples:** 1,176,851 records (15.0% stratified partition).
- **Test Samples:** 1,176,851 records (15.0% held-out test partition).
- **Total Parameters:** 55,202 trainable parameters.
- **Model Size:** 1.04 MB (`models/bilstm/best_model.keras`).
- **Epochs Completed:** 15 epochs completed (best checkpoint restored from Epoch 15).
- **Test Accuracy:** **54.88%** (645,897 correctly classified test flows out of 1,176,851).
- **Test Weighted F1-Score:** **0.5132**
- **Test Macro F1-Score:** **0.2519**
- **Test Macro Recall:** **0.3236**
- **Test Macro Precision:** **0.2602**
- **Training Wall-Clock Time:** **148.06 seconds**.
- **Test Inference Latency:** **391.58 seconds** (~332.73 microseconds per network flow).

---

## 2. Dataset Representation & Sequential Interpretation

### 2.1 Dataset Representation Verification
The dataset consists of tabular statistical summaries extracted across network flows in the CICIoT2023 benchmark.

> [!WARNING]
> **Critical Sequential Interpretation & Theoretical Limitation:**  
> The BiLSTM processes the **ordered tabular feature vector as a sequence** of 46 steps:
> $$\text{Input Shape} = (\text{samples}, \text{timesteps}=46, \text{features\_per\_step}=1)$$
> **The feature order does NOT represent a real physical or temporal sequence across consecutive network packets.**  
> The recurrence captures forward and backward dependencies across the arbitrary indexing order of tabular statistical columns. Consequently, recurrent state transitions attempt to correlate feature values along an artificial sequence axis, which explains why classical tree models (XGBoost at 99.24%, Random Forest at 99.11%) and convolutional feature extractors (1D-CNN at 66.12%) achieve higher classification fidelity on tabular flow data.

---

## 3. Model Architecture & Hyperparameter Configuration

The BiLSTM architecture is implemented in [src/models/bilstm.py](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/src/models/bilstm.py):

```
Input Tensor: (batch_size, 46, 1)
  ↓
Bidirectional LSTM (64 units per direction = 128 forward/backward, return_sequences=True)
  ↓
Dropout (0.30)
  ↓
Bidirectional LSTM (32 units per direction = 64 forward/backward, return_sequences=False)
  ↓
Dropout (0.30)
  ↓
Dense (64 units, ReLU)
  ↓
Dropout (0.30)
  ↓
Dense (34 units, Softmax)
```

| Hyperparameter | Value | Description |
|:---|:---:|:---|
| `bilstm_1 units` | 64 | 128 total forward/backward hidden state units with sequence output. |
| `bilstm_2 units` | 32 | 64 total forward/backward hidden state units with final state aggregation. |
| `dropout` | 0.30 | Regularization applied after recurrent layers and dense projection. |
| `dense_units` | 64 | Fully connected classification layer. |
| `output_units` | 34 | Softmax probability distribution over 34 classes. |
| `optimizer` | Adam ($\eta = 0.001$) | Adaptive gradient descent with learning rate annealing. |
| `loss` | Sparse Categorical Crossentropy | Multi-class cross-entropy over 34 integer labels. |
| `batch_size` | 512 | Batch size for gradient computation. |

---

## 4. Class Imbalance Handling

Balanced class weights were calculated **strictly on the training split**:
$$w_c = \frac{N_{\text{train}}}{K \cdot N_c}$$
Weights were clipped to $[0.2, 50.0]$ to prevent recurrent gradient explosions on rare classes. Validation and test sets were preserved unaltered.

---

## 5. Empirical Evaluation Results

### 5.1 Validation vs. Test Metrics

| Partition | Samples | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 | Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Validation** | 1,176,851 | 54.89% | 0.2599 | 0.3238 | 0.2520 | 0.5314 | 0.5489 | 0.5134 | 394.12s |
| **Test** | 1,176,851 | 54.88% | 0.2602 | 0.3236 | 0.2519 | 0.5312 | 0.5488 | 0.5132 | 391.58s |

---

## 6. Confusion Matrix & Misclassification Findings

- **Total Test Misclassifications:** 530,954 flows (45.12% error rate).
- **Major Confusion Pairs:**
  1. `DDoS-UDP_Flood` $\rightarrow$ `DoS-UDP_Flood` (132,041 flows): Recurrent hidden states fail to distinguish subtle rate-based thresholds between DoS and DDoS UDP attacks.
  2. `DoS-TCP_Flood` $\rightarrow$ `DDoS-TCP_Flood` (59,410 flows).
  3. `DoS-SYN_Flood` $\rightarrow$ `DDoS-SYN_Flood` (46,120 flows).
  4. `BenignTraffic` $\rightarrow$ `DDoS-TCP_Flood` (11,402 flows).

---

## 7. Comparative Benchmark Hierarchy

| Model | Accuracy | Weighted F1 | Macro F1 | Training Time | Test Latency | Model Size | Parameter Count |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | 99.11% | 0.9917 | 0.7872 | 120.24 s | 13.27 s | 1,568.01 MB | Trees=100 |
| **XGBoost** | **99.24%** | **0.9930** | **0.7928** | 158.73 s | **12.32 s** | 7.36 MB | Trees=100 |
| **1D-CNN** | 66.12% | 0.6151 | 0.4030 | 682.75 s | 33.48 s | **0.59 MB** | 46,626 |
| **BiLSTM** | 54.88% | 0.5132 | 0.2519 | 148.06 s | 391.58 s | 1.04 MB | 55,202 |

---

## 8. Summary of Generated Artifacts

- **Model Checkpoint:** `models/bilstm/best_model.keras` (1.04 MB)
- **Metadata JSON:** `models/bilstm/metadata.json`
- **Training History CSV:** `results/metrics/bilstm_history.csv`
- **Training Curves Plot:** `results/graphs/bilstm_training_history.png`
- **Validation Metrics:** `results/metrics/bilstm_validation.json`
- **Test Metrics:** `results/metrics/bilstm_test.json`
- **Per-Class Classification Report:** `results/metrics/bilstm_classification_report.csv`
- **Confusion Matrix Plot:** `results/confusion_matrices/bilstm_confusion_matrix.png`
- **Misclassifications CSV:** `results/metrics/bilstm_misclassifications.csv`
- **Updated Comparison Table:** `results/metrics/model_comparison.csv`
- **Training Script:** `scripts/train_bilstm.py`
- **Interactive Notebook:** `notebooks/07_bilstm.ipynb`
- **Test Suite:** `tests/test_bilstm.py` (8 tests, all passing)
