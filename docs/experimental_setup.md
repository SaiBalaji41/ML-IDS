# Comprehensive Experimental Setup & Methodology (Phase 14)

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Dataset:** CICIoT2023 Network Traffic Benchmark  
**Status:** Complete & Ground-Truth Documented  

---

## 1. Overview
This document provides the definitive, factual experimental setup, technical specifications, and evaluation methodology for the ML-Powered Intrusion Detection System (ML-IDS). All documented details are derived directly from the verified implementation, serialized model artifacts, and evaluation logs across Phases 1–13.

---

## 2. Dataset Description
- **Dataset Title:** CICIoT2023 Benchmark Dataset
- **Domain:** Internet of Things (IoT) and High-Throughput Network Traffic Security
- **Raw Composition:** 33 CSV files totaling **7,845,673 network flow records**
- **Feature Space:** 46 numerical statistical flow features
- **Classification Target:** 34 multiclass categories (8 primary attack categories + Benign)
- **Data Quality:** Infinite and missing values were purged during ingestion; zero-variance columns were dropped.

---

## 3. Dataset Representation
The system evaluates **tabular network-flow features** aggregated over bidirectional network sessions. The models do not operate on raw packet payload bytes. For deep learning models (1D-CNN, BiLSTM, and Hybrid CNN-BiLSTM), the continuous 46-dimensional feature vector is reshaped into a pseudo-sequence tensor of dimension `[batch_size, 46, 1]`.

---

## 4. Preprocessing Methodology
1. **Cleaning:** Dropped duplicate records and removed rows with invalid or infinite numerical values.
2. **Stateless Standardization:** Applied `StandardScaler` to ensure zero mean ($\mu = 0$) and unit variance ($\sigma = 1$). Scaler parameters were fitted exclusively on the training partition and serialized to `models/preprocessing/scaler.pkl` to prevent data leakage.
3. **Label Encoding:** Mapped 34 textual attack categories to discrete integer indices in range $[0, 33]$ saved in `data/processed/label_mapping.json`.
4. **Class Imbalance Strategy:** Applied balanced inverse-frequency class weights during model optimization.

---

## 5. Data Splitting
The 7,845,673 preprocessed flow records were partitioned using a stratified split (Random Seed = 42):
- **Training Set (70%):** 5,491,971 samples
- **Validation Set (15%):** 1,176,851 samples
- **Test Set (15%):** 1,176,851 samples (`data/processed/test/test.npz`)

---

## 6. Evaluated Models & Architectures

| Architecture | Model Family | Key Hyperparameters & Structure | Trainable Parameters | Checkpoint Path |
| :--- | :--- | :--- | :---: | :--- |
| **Random Forest** | Classical Bagging Ensemble | $200$ Trees, Max Depth $25$, Balanced Class Weights, $10\%$ Max Samples | $200$ Trees | [`models/random_forest.pkl`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/models/random_forest.pkl) |
| **XGBoost** 🏆 | Gradient Boosted Trees | $100$ Estimators, Depth $6$, $\text{LR}=0.1$, Subsample $0.8$, `hist` Tree Method | $100$ Trees | [`models/xgboost.pkl`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/models/xgboost.pkl) |
| **1D-CNN** | Deep Convolutional Network | `Conv1D(64)` $\rightarrow$ `MaxPool(2)` $\rightarrow$ `Conv1D(128)` $\rightarrow$ `GAP` $\rightarrow$ `Dense(128)` | 46,626 | [`models/cnn_1d/best_model.keras`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/models/cnn_1d/best_model.keras) |
| **BiLSTM** | Recurrent Neural Network | `BiLSTM(64)` $\rightarrow$ `BiLSTM(32)` $\rightarrow$ `Dense(64)` $\rightarrow$ `Dropout(0.30)` | 81,378 | [`models/bilstm/best_model.keras`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/models/bilstm/best_model.keras) |
| **CNN + BiLSTM** | Hybrid Neural Network | `Conv1D(64)` $\rightarrow$ `MaxPool(2)` $\rightarrow$ `BiLSTM(64)` $\rightarrow$ `Dense(64)` | 77,026 | [`models/cnn_bilstm/best_model.keras`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/models/cnn_bilstm/best_model.keras) |

---

## 7. Training Configuration
- **Loss Function:** Sparse Categorical Crossentropy (Neural Networks) / Multi-class Softmax Loss (XGBoost)
- **Optimizer:** Adam (Initial learning rate $\alpha = 0.001$)
- **Batch Size:** 512
- **Regularization:** Dropout (0.30) and Batch Normalization
- **Callbacks:** EarlyStopping (patience 5, restoring best weights) and ReduceLROnPlateau (factor 0.5, patience 2)

---

## 8. Computational Environment
- **Operating System:** Windows 11 (`Windows-11-10.0.26200-SP0`, 64-bit AMD64)
- **CPU:** Intel64 Family 6 Model 154 Stepping 3, GenuineIntel
- **RAM:** 16 GB DDR4/DDR5
- **Software Stack:** Python 3.12.10, TensorFlow 2.21.0, Keras 3.15.1, Scikit-Learn 1.6.0+, XGBoost 3.4.1, SHAP 0.52.0, NumPy 2.5.3, Pandas 3.0.5
- *Detailed Environment Report:* [`docs/computational_environment.md`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/docs/computational_environment.md)

---

## 9. Evaluation Metrics
Models were evaluated on the held-out test partition ($1,176,851$ flows) across:
- **Accuracy:** Overall correctness
- **Macro Precision, Recall, F1-Score:** Unweighted averages across all 34 classes
- **Weighted Precision, Recall, F1-Score:** Support-weighted averages
- **Operational Latency:** Inference duration per flow (milliseconds)

---

## 10. Confusion Matrix & Error Analysis
- 34x34 confusion matrices were computed and visualized in raw and normalized forms.
- One-vs-Rest (OvR) true positives, true negatives, false positives, and false negatives were tabulated for every class.
- *Detailed Error Report:* [`docs/confusion_matrix_analysis.md`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/docs/confusion_matrix_analysis.md)

---

## 11. SHAP Explainability (XAI)
- **Tree Models:** Analyzed using `shap.TreeExplainer`.
- **Neural Models:** Analyzed via gradient saliency and background reference distribution attributions.
- **Top Identified Features:** `IAT`, `Protocol Type`, `Header_Length`, `syn_flag_number`, `Tot size`.
- *Detailed XAI Report:* [`docs/shap_explainability.md`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/docs/shap_explainability.md)

---

## 12. Experimental Workflow
```text
CICIoT2023 Raw CSV Files (33 files, 7.84M flows)
                     │
                     ▼
Data Preprocessing (Sanitization, Zero-Variance Drop, StandardScaler)
                     │
                     ▼
Stratified Data Split (70% Train: 5.49M | 15% Val: 1.17M | 15% Test: 1.17M)
                     │
   ┌─────────────────┴─────────────────┐
   ▼                                   ▼
Tree Baselines               Deep Learning Architectures
• Random Forest              • Standalone 1D-CNN
• XGBoost (Champion)         • Standalone BiLSTM
                             • Proposed Hybrid CNN + BiLSTM
   │                                   │
   └─────────────────┬─────────────────┘
                     ▼
Test Evaluation (1,176,851 Held-Out Test Flows, 34 Classes)
                     │
                     ▼
Confusion Matrix & Error Decomposition (TP, TN, FP, FN, Subclass Triage)
                     │
                     ▼
Explainable AI (SHAP Global & Local Feature Attributions)
                     │
                     ▼
Research Paper Results & SOC Operational Decision Support
```
*Visual Architecture Diagram:* [`results/graphs/experimental_setup/experimental_pipeline.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/experimental_setup/experimental_pipeline.png)

---

## 13. Reproducibility
Complete execution commands, environment specifications, and random seeds (42) are documented in [`docs/reproducibility.md`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/docs/reproducibility.md).

---

## 14. Experimental Limitations
Factual limitations regarding tabular representation, class imbalance, compute environment, and single-source vs. multi-source DoS/DDoS confusion are detailed in [`docs/experimental_limitations.md`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/docs/experimental_limitations.md).
