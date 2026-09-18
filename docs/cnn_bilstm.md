# Phase 10 Technical Report: Proposed Hybrid CNN + BiLSTM Model

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Phase:** Phase 10 — Proposed Hybrid CNN + BiLSTM Model  
**Target Dataset:** CICIoT2023  
**Execution Date:** 2026-09-17  
**Status:** Architecture, Pipeline, Notebook, Tests, and Documentation Completed & Verified  

---

## 1. Executive Summary

This report presents the implementation, training, and empirical evaluation of the **Proposed Primary Model: Hybrid 1D-CNN + BiLSTM Neural Network** for network intrusion detection.

- **Classification Type:** Multiclass Intrusion Detection (34 distinct classes: 1 Benign Traffic class + 33 IoT Attack variants).
- **Deep Learning Framework:** TensorFlow `2.21.0` / Keras `3.15.1`
- **Dataset Source:** Verified preprocessed numeric feature partitions (`data/processed/`).
- **Input Tensor Shape:** `[batch_size, 46, 1]` (46 standardized statistical network flow features).
- **Architecture Pipeline:**
  $$\text{Input (46, 1)} \longrightarrow \text{Conv1D (64 filters, k=3)} \longrightarrow \text{BatchNorm} \longrightarrow \text{ReLU} \longrightarrow \text{MaxPool (2)} \longrightarrow \text{BiLSTM (64 units)} \longrightarrow \text{Dense (64)} \longrightarrow \text{Softmax (34)}$$
- **Total Parameter Count:** 75,554 trainable parameters.
- **Model Checkpoint:** `models/cnn_bilstm/best_model.keras` (~0.95 MB).

---

## 2. Theoretical Motivation & Architectural Design

The Hybrid CNN-BiLSTM architecture is designed to combine the complementary strengths of convolutional feature extraction and recurrent sequence modeling:

1. **1D-CNN Feature Extractor Block:**
   - A 1D convolutional layer with 64 filters of kernel size 3 scans adjacent features in the ordered flow vector, extracting localized composite representations.
   - Batch Normalization stabilizes internal covariate shift.
   - MaxPooling1D (pool size = 2) halves the spatial dimension from 46 to 23 steps, discarding redundant features while accelerating recurrent processing.
2. **BiLSTM Context Aggregation Block:**
   - A Bidirectional LSTM with 64 hidden units (128 total forward/backward state dimensions) processes the compressed 23-step sequence representation to model cross-feature contextual interactions.
3. **Dense Classification Head:**
   - Fully connected layer with 64 ReLU units and 30% dropout regularization, followed by a 34-class Softmax probability output.

---

## 3. Class Imbalance Handling

Balanced class weights were calculated **exclusively on the training partition**:
$$w_c = \frac{N_{\text{train}}}{K \cdot N_c}$$
Weights were clipped to $[0.2, 50.0]$ to prevent gradient instability on ultra-sparse minority attacks. Validation and test partitions remained unaltered.

---

## 4. Empirical Evaluation Protocol

- **Training Partition:** 350,000 stratified samples.
- **Validation Partition:** 1,176,851 samples (with 50,000 subsample for fast intermediate callback monitoring).
- **Test Partition:** 1,176,851 samples (single unbiased final evaluation).
- **Optimization:** Adam ($\eta = 0.001$), `EarlyStopping(patience=5)`, `ReduceLROnPlateau(patience=2, factor=0.5)`.

---

## 5. Fair Comparison Protocol

All 5 models in the comparative hierarchy were evaluated under identical experimental conditions:
1. **Identical Partitions:** Same 70/15/15 stratified train/val/test splits (`data/processed/`).
2. **Zero Preprocessing Leakage:** Standard scalers fitted strictly on training data.
3. **Consistent Target Encoding:** Identical 34-class label dictionary (`label_mapping.json`).
4. **Identical Metric Definitions:** Scikit-Learn standard accuracy, macro precision/recall/F1, and weighted precision/recall/F1.

---

## 6. Summary of Generated Artifacts

- **Model Checkpoint:** `models/cnn_bilstm/best_model.keras`
- **Metadata JSON:** `models/cnn_bilstm/metadata.json`
- **Training History CSV:** `results/metrics/cnn_bilstm_history.csv`
- **Training Curves Plot:** `results/graphs/cnn_bilstm_training_history.png`
- **Validation Metrics:** `results/metrics/cnn_bilstm_validation.json`
- **Test Metrics:** `results/metrics/cnn_bilstm_test.json`
- **Classification Report:** `results/metrics/cnn_bilstm_classification_report.csv`
- **Confusion Matrix Plot:** `results/confusion_matrices/cnn_bilstm_confusion_matrix.png`
- **Misclassifications CSV:** `results/metrics/cnn_bilstm_misclassifications.csv`
- **Updated Comparison Tables:** `results/metrics/model_comparison.csv` and `results/metrics/model_comparison_extended.csv`
- **Training Script:** `scripts/train_cnn_bilstm.py`
- **Interactive Notebook:** `notebooks/08_cnn_bilstm.ipynb`
- **Test Suite:** `tests/test_cnn_bilstm.py`
