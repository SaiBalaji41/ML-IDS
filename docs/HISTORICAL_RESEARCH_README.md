# Historical research snapshot

This is the earlier project README, retained as research history. Its completion claims, platform instructions and dashboard descriptions are superseded by the repository README and output/delivery/delivery_report.md.

# ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Project Status](https://img.shields.io/badge/Project%20Status-Complete%20(Phases%201--23)-brightgreen.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-CICIoT2023-orange.svg)]()
[![Tests](https://img.shields.io/badge/Tests-73%20Passed-success.svg)]()
[![License](https://img.shields.io/badge/Academic-B.Tech%20Major%20Project-blueviolet.svg)]()

---

## 1. Project Title
**ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring**

---

## 2. Executive Overview
This project presents an end-to-end, reproducible, and explainable Intrusion Detection System (IDS) tailored for IoT and high-speed network topologies. Benchmarked on the **CICIoT2023** dataset (**7.84 Million network flows**, 46 statistical flow features, 34 multi-class categories), the framework evaluates classical tree-based ensembles (**Random Forest**, **XGBoost**), standalone deep neural networks (**1D-CNN**, **BiLSTM**), and a proposed hybrid **1D-CNN + BiLSTM** architecture with game-theoretic **SHAP (SHapley Additive exPlanations)** interpretability and a real-time SOC monitoring dashboard.

---

## 3. Core Architecture & Pipeline

```text
                      CICIoT2023 Network Flows (46 Features)
                                       │
                                       ▼
                       Zero-Leakage Preprocessing Pipeline
                     (Robust Scaling & Stratified Splitting)
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
  Classical Machine Learning Baselines           Deep Learning & Hybrid Models
  • Random Forest (99.11% Acc)                   • Standalone 1D-CNN (Spatial)
  • XGBoost       (99.24% Acc)                   • Standalone BiLSTM (Sequential)
                                                 • Proposed 1D-CNN + BiLSTM Hybrid
                └──────────────────────┬──────────────────────┘
                                       │
                                       ▼
                      Evaluation & Empirical Comparisons
                    (Multi-Class Metrics & Confusion Matrices)
                                       │
                                       ▼
                   Explainable AI (SHAP Interpretability)
                 (Global Beeswarm, Class Rankings, Local Plots)
                                       │
                                       ▼
                Real-Time Streaming & Interactive SOC Dashboard
               (Scapy Sniffer Simulation & Pure-Python Web UI)
```

---

## 4. Empirical Benchmark Performance

Evaluated on the identical held-out test partition ($N = 1,176,851$ network flows across 34 multiclass categories):

| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score | Training Time | Inference Latency | Trainable Parameters | Serialized Disk Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost 🏆** | **99.24%** | **77.01%** | **85.97%** | **0.7928** | **0.9930** | 158.73 s | **0.0105 ms/flow** | 100 Trees | 7.36 MB |
| **Random Forest** | **99.11%** | 75.42% | 84.56% | 0.7872 | 0.9917 | **120.24 s** | 0.0113 ms/flow | 200 Trees | 1,568.01 MB |
| **BiLSTM** | 79.47% | 48.51% | 54.23% | 0.4846 | 0.7756 | 74,830.41 s | 0.3316 ms/flow | 81,378 | 1.00 MB |
| **1D-CNN** | 66.12% | 43.12% | 46.97% | 0.4030 | 0.6151 | 682.75 s | 0.0285 ms/flow | 46,626 | **0.59 MB** |
| **CNN + BiLSTM (Hybrid)** | 62.47% | 38.47% | 41.90% | 0.3465 | 0.5532 | 92.44 s | 0.1338 ms/flow | 77,026 | 0.94 MB |

---

## 5. Phased Development Roadmap

- [x] **Phase 1:** PRD & Requirements Engineering
- [x] **Phase 2:** Project Foundation & Environment Setup
- [x] **Phase 3:** CICIoT2023 Dataset Setup & Empirical Analysis (7.84M flows, 46 features, 34 classes)
- [x] **Phase 4:** Leakage-Free Preprocessing Pipeline (StandardScaler on train split)
- [x] **Phase 5:** Stratified Train/Val/Test Splitting (70% / 15% / 15%)
- [x] **Phase 6:** Random Forest Baseline Implementation & Evaluation (99.11% Accuracy)
- [x] **Phase 7:** XGBoost Baseline Implementation & Evaluation (99.24% Accuracy)
- [x] **Phase 8:** Standalone 1D-CNN Deep Learning Baseline (66.12% Accuracy)
- [x] **Phase 9:** Standalone BiLSTM Deep Learning Baseline (79.47% Accuracy)
- [x] **Phase 10:** Proposed Hybrid 1D-CNN + BiLSTM Architecture & Training
- [x] **Phase 11:** Multi-Model Evaluation & Consolidated Benchmarking
- [x] **Phase 12:** 34x34 Confusion Matrix Audits & Security Error Diagnostics
- [x] **Phase 13:** Explainable AI (SHAP Global Feature Importance & Local Attribution Plots)
- [x] **Phase 14:** Experimental Methodology Documentation & Setup Specifications
- [x] **Phase 15:** Results & Analysis Recovery (Preserved Artifacts & Comparison Figures)
- [x] **Phase 16:** Academic Research Paper Draft & Verified Bibliography
- [x] **Phase 17–20:** Real-Time Stream Ingestion, Packet Sniffer Simulation & SOC Dashboard
- [x] **Phase 21–23:** Comprehensive Viva Defense Guide, Test Suite Verification (73/73 tests pass)

---

## 6. Quick Start & Execution

### 6.1 Installation
```bash
# Clone repository
git clone <repository_url>
cd ML-IDS

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate    # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 6.2 Launch Interactive SOC Dashboard (Browser UI)
```bash
python scripts/run_pipeline.py --phase dashboard --port 8080
```
Open **`http://localhost:8080`** in your browser to interact with:
- Live streaming event simulation and real-time threat detection
- Interactive 34x34 confusion matrix viewer
- SHAP feature importance charts and local flow explanations
- Multi-model comparative benchmark dashboards

### 6.3 Run Real-Time Stream Ingestion Simulation
```bash
python scripts/run_pipeline.py --phase realtime --samples 25
```

### 6.4 Execute Full Test Suite
```bash
pytest
```
*73 tests passing covering basic structure, preprocessing, Random Forest, XGBoost, 1D-CNN, BiLSTM, CNN-BiLSTM, error analysis, SHAP, packet sniffer, and dashboard server.*

---

## 7. Key Documentation & Research Artifacts

- **Academic Research Paper Draft:** [`docs/research_paper.md`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/docs/research_paper.md)
- **Comprehensive Project Clarity & Viva Defense Guide:** [`docs/PROJECT_CLARITY_AND_VIVA_GUIDE.md`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/docs/PROJECT_CLARITY_AND_VIVA_GUIDE.md) — 25 examiner Q&As, architecture justifications, and evaluation proofs.
- **Results & Analysis Report:** [`docs/results_and_analysis.md`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/docs/results_and_analysis.md)
- **SHAP Explainability Report:** [`docs/shap_explainability.md`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/docs/shap_explainability.md)
- **Verified Bibliography:** [`docs/references_verified.md`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/docs/references_verified.md)
- **Research Figures Catalog:** [`docs/research_figures_index.md`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/docs/research_figures_index.md)
- **Product Requirements Document (PRD):** [`docs/PRD.md`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/docs/PRD.md)
