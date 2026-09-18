# Reproducibility & Replication Guide (Phase 14)

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Dataset:** CICIoT2023 Network Traffic Benchmark  
**Core Random Seed:** **42**  

---

## 1. Overview & Replication Principles
To ensure exact scientific replication of all experimental metrics, confusion matrices, and SHAP explanations, this project enforces strict reproducibility protocols:
1. **Deterministic Pseudo-Random Seeds:** All partitioning, cross-validation, batch shuffling, and weight initializations utilize fixed random seed `42`.
2. **Stateless Transformation Pipelines:** Preprocessing scalers are fitted exclusively on the training split and serialized to `models/preprocessing/scaler.pkl` to prevent leakage into validation and test sets.
3. **Immutable Saved Model Binaries:** All five candidate models are saved as deterministic serialized artifacts.

---

## 2. Environment Replication

### 2.1 Virtual Environment Setup
```bash
# Clone the repository
git clone <repository_url>
cd ML-IDS

# Initialize isolated Python 3.12 virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install exact pinned dependencies
pip install -r requirements.txt
```

---

## 3. Data Partitions & Pipeline Execution

### 3.1 Data Directory Layout
```text
data/
├── raw/                      # Original CICIoT2023 CSV files (33 files)
└── processed/
    ├── train/                # 5,491,971 samples (70%)
    ├── val/                  # 1,176,851 samples (15%)
    ├── test/                 # 1,176,851 samples (15%) [test.npz]
    ├── label_mapping.json    # 34-class label dictionary
    └── preprocessing_metadata.json
```

---

## 4. Execution Commands for All Experimental Phases

### 4.1 Master Pipeline Execution
```bash
# Run entire end-to-end evaluation and real-time streaming pipeline
python scripts/run_pipeline.py --phase all

# Run specific phases individually
python scripts/run_pipeline.py --phase evaluate
python scripts/run_pipeline.py --phase explain
python scripts/run_pipeline.py --phase realtime --samples 25
python scripts/run_pipeline.py --phase dashboard
```

### 4.2 Confusion Matrix & Detailed Error Analysis (Phase 12)
```bash
python scripts/analyze_confusion_matrices.py
```

### 4.3 SHAP Explainability & Feature Attribution (Phase 13)
```bash
python scripts/compute_phase13_shap.py
```

### 4.4 Automated Unit & Regression Test Suite
```bash
pytest tests/
```

---

## 5. Serialized Model Checkpoints & Artifact Registry

| Model Architecture | Checkpoint File Path | File Size | Serialization Library |
| :--- | :--- | :---: | :--- |
| **Random Forest** | [`models/random_forest.pkl`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/models/random_forest.pkl) | 1,568 MB | Joblib 1.6.0 |
| **XGBoost** | [`models/xgboost.pkl`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/models/xgboost.pkl) | 7.36 MB | Joblib / XGBoost 3.4.1 |
| **1D-CNN** | [`models/cnn_1d/best_model.keras`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/models/cnn_1d/best_model.keras) | 0.59 MB | Keras 3 Native Format |
| **BiLSTM** | [`models/bilstm/best_model.keras`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/models/bilstm/best_model.keras) | 1.00 MB | Keras 3 Native Format |
| **CNN + BiLSTM** | [`models/cnn_bilstm/best_model.keras`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/models/cnn_bilstm/best_model.keras) | 0.94 MB | Keras 3 Native Format |
| **Scaler** | [`models/preprocessing/scaler.pkl`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/models/preprocessing/scaler.pkl) | 2.6 KB | Joblib `StandardScaler` |
