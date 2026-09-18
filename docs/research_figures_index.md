# Research Paper Figure & Visualization Index

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Manuscript Draft:** [`docs/research_paper.md`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/docs/research_paper.md)  
**Status:** All figures verified and mapped to existing project artifact paths.

---

## 1. Overall Performance and Model Comparison Figures

| Fig. # | Figure Title | Artifact Path | Description | Research Purpose |
| :---: | :--- | :--- | :--- | :--- |
| **Fig. 1** | Multi-Class Classification Accuracy Comparison | [`results/graphs/results_analysis/accuracy_comparison.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/results_analysis/accuracy_comparison.png) | Bar plot comparing multi-class accuracy across all 5 candidate models on 1,176,851 test flows. | Illustrates baseline superiority of tree models over deep neural models on tabular data. |
| **Fig. 2** | Macro F1-Score Comparison Across 34 Classes | [`results/graphs/results_analysis/macro_f1_comparison.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/results_analysis/macro_f1_comparison.png) | Unweighted macro-averaged F1-score across all 34 classes. | Highlights resistance to class imbalance and minority class sensitivity. |
| **Fig. 3** | Total Training Time Comparison (Log Scale) | [`results/graphs/results_analysis/training_time_comparison.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/results_analysis/training_time_comparison.png) | Logarithmic execution time comparison for training over 5.49 million flows. | Demonstrates the extreme computational efficiency advantage of XGBoost and Random Forest over recurrent models. |
| **Fig. 4** | Per-Flow Inference Latency (ms / Flow) | [`results/graphs/results_analysis/inference_time_comparison.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/results_analysis/inference_time_comparison.png) | Wall-clock latency per network flow evaluated on the test set. | Establishes feasibility for high-throughput inline packet inspection at line rates. |
| **Fig. 5** | Model Serialized Disk Footprint | [`results/graphs/results_analysis/model_size_comparison.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/results_analysis/model_size_comparison.png) | Memory and disk storage requirements across serialized model binaries. | Evaluates edge and resource-constrained deployment feasibility (e.g. 1D-CNN vs. RF). |

---

## 2. Confusion Matrices ($34 \times 34$)

| Fig. # | Figure Title | Artifact Path | Description | Research Purpose |
| :---: | :--- | :--- | :--- | :--- |
| **Fig. 6** | XGBoost Normalized Confusion Matrix | [`results/confusion_matrices/xgboost_confusion_matrix_normalized.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/xgboost_confusion_matrix_normalized.png) | Row-normalized recall heatmap for XGBoost across 34 classes. | Confirms diagonal dominance and high true positive rates across major attack vectors. |
| **Fig. 7** | Random Forest Normalized Confusion Matrix | [`results/confusion_matrices/random_forest_confusion_matrix_normalized.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/random_forest_confusion_matrix_normalized.png) | Row-normalized recall heatmap for Random Forest across 34 classes. | Evaluates bagging decision tree classification consistency. |
| **Fig. 8** | 1D-CNN Normalized Confusion Matrix | [`results/confusion_matrices/cnn_1d_confusion_matrix_normalized.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/cnn_1d_confusion_matrix_normalized.png) | Row-normalized heatmap for standalone 1D-CNN. | Reveals structural confusion between single and multi-source UDP floods. |
| **Fig. 9** | BiLSTM Normalized Confusion Matrix | [`results/confusion_matrices/bilstm_confusion_matrix_normalized.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/bilstm_confusion_matrix_normalized.png) | Row-normalized heatmap for standalone BiLSTM. | Demonstrates sequential contextual learning improvements over 1D-CNN. |
| **Fig. 10** | Hybrid CNN+BiLSTM Normalized Confusion Matrix | [`results/confusion_matrices/cnn_bilstm_confusion_matrix_normalized.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/cnn_bilstm_confusion_matrix_normalized.png) | Row-normalized heatmap for proposed CNN+BiLSTM hybrid. | Analyzes spatial-temporal representation capabilities on tabular flow features. |

---

## 3. Deep Learning Training History & Convergence

| Fig. # | Figure Title | Artifact Path | Description | Research Purpose |
| :---: | :--- | :--- | :--- | :--- |
| **Fig. 11** | 1D-CNN Training & Validation Trajectory | [`results/graphs/cnn_1d_training_history.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/cnn_1d_training_history.png) | Loss and accuracy curves across 9 epochs showing early stopping at Epoch 4. | Documents convergence rate and validation loss plateau behavior. |
| **Fig. 12** | BiLSTM Training & Validation Trajectory | [`results/graphs/bilstm_training_history.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/bilstm_training_history.png) | Loss and accuracy curves across 15 epochs showing continuous monotonic convergence. | Confirms stable optimization and gradient flow through bidirectional recurrence. |
| **Fig. 13** | Hybrid CNN+BiLSTM Training Trajectory | [`results/graphs/cnn_bilstm_training_history.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/cnn_bilstm_training_history.png) | Loss and accuracy trajectories across 4 epochs. | Tracks hybrid feature extraction layer interactions during training. |

---

## 4. Operational Security & Error Diagnostics

| Fig. # | Figure Title | Artifact Path | Description | Research Purpose |
| :---: | :--- | :--- | :--- | :--- |
| **Fig. 14** | Benign vs. Malicious Security Errors | [`results/graphs/error_analysis/benign_vs_attack_confusion.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/error_analysis/benign_vs_attack_confusion.png) | Comparative plot of False Positives (SOC alert fatigue) vs False Negatives (breach risk). | Demonstrates operational reliability of tree models (<0.035% FN rate). |
| **Fig. 15** | Class-Wise F1 Heatmap Across Models | [`results/graphs/error_analysis/classwise_f1_comparison.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/error_analysis/classwise_f1_comparison.png) | Matrix of F1 scores across 34 classes for all 5 models. | Visually pinpoints the minority web attack performance divergence. |

---

## 5. SHAP Explainability (XAI) Figures

| Fig. # | Figure Title | Artifact Path | Description | Research Purpose |
| :---: | :--- | :--- | :--- | :--- |
| **Fig. 16** | XGBoost Global SHAP Importance Summary | [`results/shap/shap_global_summary_xgboost.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/shap_global_summary_xgboost.png) | Mean absolute SHAP contribution ranking across all 46 features. | Identifies `IAT`, `Protocol Type`, `Header_Length`, and TCP flags as primary drivers. |
| **Fig. 17** | Random Forest Global SHAP Summary | [`results/shap/shap_global_summary_random_forest.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/shap_global_summary_random_forest.png) | Mean absolute SHAP ranking for Random Forest. | Confirms game-theoretic feature consensus across ensemble tree architectures. |
| **Fig. 18** | Local Attribution — Benign Traffic | [`results/shap/shap_local_instance_BenignTraffic.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/shap_local_instance_BenignTraffic.png) | Instance-level SHAP explanation for a verified benign flow. | Shows how normal inter-arrival times and flag handshakes push logit to Benign. |
| **Fig. 19** | Local Attribution — DDoS ICMP Flood | [`results/shap/shap_local_instance_DDoS_ICMP_Flood.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/shap_local_instance_DDoS_ICMP_Flood.png) | Waterfall explanation for an ICMP flood packet flow. | Demonstrates how ICMP protocol type and packet rate drive 100% attack alert confidence. |
| **Fig. 20** | Local Attribution — Reconnaissance Port Scan | [`results/shap/shap_local_instance_Recon_PortScan.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/shap_local_instance_Recon_PortScan.png) | Attribution plot for stealth port scanning flow. | Explains subtle port discovery detection mechanics. |
