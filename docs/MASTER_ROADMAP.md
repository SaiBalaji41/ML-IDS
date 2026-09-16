# MASTER SPECIFICATION & ROADMAP
## ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring

**Project Type:** B.Tech Major Project  
**Primary Dataset:** CICIoT2023  
**Target Architecture:** Hybrid 1D-CNN + BiLSTM with SHAP Explainability  
**Status:** In Progress (Phase: Dataset Ingestion & Step 3 Analysis)

---

## 1. Project Overview & Core Philosophy
The objective is to build an intelligent Intrusion Detection System tailored for IoT/network environments capable of:
1. Learning spatial and temporal patterns from network traffic flows.
2. Accurately classifying benign and diverse malicious activities.
3. Providing interpretable feature attributions via SHAP.
4. Enabling a future real-time streaming pipeline (Scapy + Kafka + IDS Model + Alert Dashboard).

---

## 2. Strict Dataset & Experimental Rules
- **No Fabricated Statistics:** No metrics, counts, distributions, or column assumptions will be made prior to empirical inspection of raw files.
- **Empirical Grounding:** All dataset metadata (number of samples, feature names, data types, target labels, class distribution, imbalance ratio) must be calculated directly from files placed in `data/raw/`.
- **Modular Pipeline:** Clear separation of concerns across `src/preprocessing/`, `src/models/`, `src/evaluation/`, `src/explainability/`, and `src/realtime/`.
- **Reproducibility:** Every preprocessing transformation, split seed, and hyperparameter configuration must be logged and version-controlled.

---

## 3. Systematic Execution Roadmap (20 Project Objectives)

### Phase 1: Dataset Discovery & Exploratory Data Analysis (Current Phase)
1. **Analyze CICIoT2023 Dataset:** Inspect actual files, extensions, byte sizes, and formats.
2. **Understand Structure:** Extract exact features, data types, labels, classes, and sample distributions.
3. **Data Quality Audit:** Quantify missing, duplicate, infinite, and invalid numerical values.
4. **Input Representation Verification:** Determine whether data is flow-feature-based, packet-level, or raw payload.

### Phase 2: Data Preprocessing & Pipeline Engineering
5. **Preprocessing Pipeline:** Clean data, scale features without leakage, encode targets, and create stratified splits (train/val/test).
6. **Class Imbalance Strategy:** Formulate class balancing (weighting/resampling) grounded in actual class ratios.
7. **Sequence/Window Transformation:** Prepare data tensors suitable for 1D-CNN and BiLSTM input layers.

### Phase 3: Model Implementation & Baseline Benchmarking
8. **Baseline Models:** Implement and evaluate Random Forest and XGBoost.
9. **Standalone Deep Learning:** Implement standalone 1D-CNN and standalone BiLSTM.
10. **Proposed Hybrid Architecture:** Develop and tune the hybrid **1D-CNN + BiLSTM** network.
11. **Performance Evaluation:** Compute multi-class metrics (Accuracy, Precision, Recall, Macro/Weighted F1-score, Confusion Matrices).

### Phase 4: Model Explainability (XAI)
12. **SHAP Integration:** Compute SHAP values (DeepSHAP / KernelExplainer / TreeExplainer for baselines).
13. **Global & Local Interpretability:** Generate summary plots, dependence plots, and force plots for attack explanations.

### Phase 5: Experimental Documentation & Academic Reporting
14. **Experimental Documentation:** Log all training histories, loss curves, validation curves, and confusion heatmaps.
15. **Research Paper Preparation:** Author a structured paper/report containing only verified empirical results.

### Phase 6: Real-Time Stream Ingestion & Monitoring Pipeline (Future Scope)
16. **Scapy Packet Capture:** Sniff and parse live network packets.
17. **Apache Kafka Streaming:** Publish and subscribe network flow events through Kafka message brokers.
18. **Real-Time Inference:** Stream incoming flows through the trained CNN-BiLSTM model.
19. **Alerting & Dashboard:** Build an interactive security monitoring dashboard displaying threat alerts and traffic health.
