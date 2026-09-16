# ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Project Phase](https://img.shields.io/badge/Phase-Phase%202%20Complete-green.svg)]()
[![License](https://img.shields.io/badge/Academic-B.Tech%20Major%20Project-orange.svg)]()

---

## 1. Project Title
**ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring**

---

## 2. Project Overview
This project develops an intelligent, data-driven Intrusion Detection System (IDS) for IoT and high-throughput network environments. By integrating machine learning and deep learning models with explainable AI (XAI), the system aims to accurately detect and classify complex malicious activities from network traffic data while providing transparent, human-interpretable feature attributions to assist cybersecurity analysts.

The primary research dataset for offline experimentation is **CICIoT2023**.

---

## 3. Problem Statement
Traditional intrusion detection systems rely primarily on fixed signatures, static rules, or manual heuristics:
- **Zero-Day Vulnerability:** Inability to detect unseen or polymorphic attack variants.
- **Complex Traffic Dynamics:** Modern IoT traffic exhibits localized intra-flow correlations and sequential inter-flow temporal dependencies that single-paradigm architectures fail to capture simultaneously.
- **Black-Box Decision Making:** Standard deep learning models lack interpretability, reducing actionable trust in Security Operations Centers (SOCs).

This project addresses these challenges through a hybrid deep learning model (**1D-CNN + BiLSTM**) combined with **SHAP** for prediction explainability.

---

## 4. Project Objectives
1. Perform empirical exploratory data analysis on the **CICIoT2023** dataset.
2. Develop a leak-free, modular data preprocessing and transformation pipeline.
3. Establish traditional and deep learning baselines (**Random Forest**, **XGBoost**, standalone **1D-CNN**, standalone **BiLSTM**).
4. Implement the proposed hybrid **1D-CNN + BiLSTM** architecture for spatial-temporal representation learning.
5. Integrate **SHAP** to provide global feature rankings and per-prediction local explanations.
6. Evaluate all models using standardized classification metrics and detailed confusion matrix analysis.
7. Design a future streaming architecture using **Scapy** and **Apache Kafka** with a real-time monitoring dashboard.

---

## 5. Planned Methodology
- **Data Discovery & Audit:** Inspect raw files without prior schema assumptions to identify exact features, data types, missing values, infinite values, and class distributions.
- **Leakage Prevention:** Fit scalers, imputers, and encoders strictly on the training partition; apply statelessly to validation and test splits.
- **Empirical Imbalance Strategy:** Formulate class balancing (class weights or resampling) based on verified class frequencies.
- **Multi-Model Benchmarking:** Execute reproducible experiments with fixed random seeds across all models on identical test partitions.
- **Explainability:** Compute SHAP attributions using background reference distributions.

---

## 6. Model Pipeline
```text
Raw CICIoT2023 Data
        │
        ▼
Data Preprocessing (Cleaning, Scaling, Encoding, Stratified Splitting)
        │
        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Baseline Models            │ Proposed Hybrid Architecture   │
   │ • Random Forest            │ • 1D-CNN (Spatial Extraction)  │
   │ • XGBoost                  │ • BiLSTM (Sequential Context)  │
   │ • Standalone 1D-CNN        │ • Dense Classification Head    │
   │ • Standalone BiLSTM        │                                │
   └─────────────────────────────────────────────────────────────┘
        │
        ▼
Model Evaluation (Accuracy, Precision, Recall, F1-Score, Confusion Matrices)
        │
        ▼
SHAP Explainability (Global Feature Importance & Local Attribution Plots)
        │
        ▼ (Future Scope)
Real-Time Stream Pipeline (Scapy Sniffer → Apache Kafka → Inference → Dashboard)
```

---

## 7. Technology Stack
- **Programming Language:** Python 3.10+
- **Data Manipulation:** NumPy, Pandas, SciPy, PyYAML
- **Classical ML:** Scikit-Learn, XGBoost
- **Deep Learning:** TensorFlow / Keras (or PyTorch)
- **Model Explainability:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **Testing & QA:** Pytest
- **Future Streaming (Phases 17–20):** Scapy, Apache Kafka

---

## 8. Project Structure
```text
ML-IDS/
├── configs/
│   └── default_config.yaml         # Pipeline and hyperparameter configuration
├── data/
│   ├── raw/                        # Original immutable raw dataset files (e.g. CICIoT2023)
│   ├── processed/                  # Cleaned, scaled, and split datasets
│   └── sample/                     # Representative sample datasets for rapid testing
├── dashboard/                      # Future real-time monitoring web interface
├── docs/
│   ├── PRD.md                      # Comprehensive Product Requirements Document (Phase 1)
│   └── MASTER_ROADMAP.md           # Master project specification and roadmap
├── models/                         # Serialized trained model weights and checkpoints
├── notebooks/                      # Exploratory and experimental Jupyter notebooks
├── results/
│   ├── confusion_matrices/         # Confusion matrix heatmaps and error logs
│   ├── graphs/
│   │   └── dataset_analysis/       # Class distributions and feature analysis plots
│   ├── metrics/                    # Quantitative evaluation metrics (CSV/JSON)
│   └── shap/                       # SHAP summary, force, and waterfall plots
├── scripts/
│   └── run_pipeline.py             # CLI execution entrypoint
├── src/
│   ├── __init__.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py              # Performance evaluation and matrix routines
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── shap_explainer.py       # SHAP explanation wrappers
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_rf.py          # Random Forest baseline implementation
│   │   ├── baseline_xgb.py         # XGBoost baseline implementation
│   │   ├── bilstm.py               # Standalone BiLSTM implementation
│   │   ├── cnn_1d.py               # Standalone 1D-CNN implementation
│   │   └── hybrid_cnn_bilstm.py    # Proposed 1D-CNN + BiLSTM architecture
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── preprocessor.py         # Data cleaning, scaling, and splitting module
│   └── realtime/
│       ├── __init__.py
│       └── packet_sniffer.py       # Future Scapy packet capture module
├── tests/
│   ├── __init__.py
│   └── test_basic.py               # Unit and regression test suite
├── .env.example                    # Environment variable configuration template
├── .gitignore                      # Version control exclusion rules
├── README.md                       # Main project documentation
└── requirements.txt                # Core Python dependency manifest
```

---

## 9. Setup Instructions & Environment Setup

### 9.1 Clone and Navigate to Repository
```bash
git clone <repository_url>
cd ML-IDS
```

### 9.2 Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 9.3 Install Dependencies
```bash
pip install -r requirements.txt
```

### 9.4 Configure Environment Variables
```bash
cp .env.example .env
```

---

## 10. Dataset Placement Instructions
Place the raw CICIoT2023 dataset file(s) directly into the `data/raw/` directory:
```text
ML-IDS/data/raw/<your_dataset_files>.csv
```
> [!NOTE]
> The dataset files are excluded from version control via `.gitignore` to maintain repository lightweightness.

---

## 11. Planned Evaluation Metrics
All candidate models will be evaluated on the identical unseen test set using:
- **Accuracy**
- **Precision (Macro & Weighted)**
- **Recall (Macro & Weighted)**
- **F1-Score (Macro & Weighted)**
- **Per-Class Metrics & Confusion Matrix Heatmaps**

---

## 12. Phased Development Roadmap
- **Phase 1:** PRD & Requirements Engineering *(Completed)*
- **Phase 2:** Project Foundation & Environment Setup *(Current Completed Phase)*
- **Phase 3:** CICIoT2023 Dataset Setup & Empirical Analysis
- **Phase 4:** Data Preprocessing Pipeline
- **Phase 5:** Stratified Train/Val/Test Splitting
- **Phase 6–10:** Model Implementations (RF, XGBoost, 1D-CNN, BiLSTM, Hybrid CNN-BiLSTM)
- **Phase 11–13:** Multi-Model Evaluation, Error Diagnostics & SHAP Explainability
- **Phase 14–16:** Experimental Documentation & Research Paper Preparation
- **Phase 17–20:** Future Real-Time Pipeline (Scapy, Kafka, Dashboard)
- **Phase 21–23:** Final Validation, User Guides & Defense Preparation

---

## 13. Current Project Status & Important Notice
> [!IMPORTANT]
> **Dataset analysis and model training have not yet been performed.**
> No performance benchmarks, accuracy metrics, confusion matrices, or SHAP plots exist at this stage. All quantitative results will be derived from executed experiments on verified data in subsequent phases.

---

## 14. Research & Documentation Notes
- Official Product Requirements Document: [docs/PRD.md](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/docs/PRD.md)
- Master Specification & Roadmap: [docs/MASTER_ROADMAP.md](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/docs/MASTER_ROADMAP.md)
