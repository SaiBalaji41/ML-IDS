# Results Figure and Visualization Index

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Phase:** Phase 15 — Results & Analysis  
**Repository Path:** `c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS`

---

## 1. Overall Performance and Model Comparison Figures

| Fig. # | Figure Title | Description | File Link |
| :---: | :--- | :--- | :--- |
| **Fig. 1** | Test Multi-Class Accuracy Comparison | Bar chart comparing test multi-class accuracy across all 5 models ($N = 1,176,851$). | [accuracy_comparison.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/results_analysis/accuracy_comparison.png) |
| **Fig. 2** | Macro Precision Comparison | Unweighted macro precision across 34 multiclass categories. | [macro_precision_comparison.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/results_analysis/macro_precision_comparison.png) |
| **Fig. 3** | Macro Recall Comparison | Unweighted macro recall across 34 multiclass categories. | [macro_recall_comparison.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/results_analysis/macro_recall_comparison.png) |
| **Fig. 4** | Macro F1-Score Comparison | Unweighted macro F1-score across all 34 classes. | [macro_f1_comparison.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/results_analysis/macro_f1_comparison.png) |
| **Fig. 5** | Weighted F1-Score Comparison | Sample-weighted F1-score proportional to class distribution. | [weighted_f1_comparison.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/results_analysis/weighted_f1_comparison.png) |
| **Fig. 6** | Training Time Comparison (Log Scale) | Total training execution duration in seconds across models. | [training_time_comparison.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/results_analysis/training_time_comparison.png) |
| **Fig. 7** | Per-Flow Inference Latency | Per-flow evaluation latency in milliseconds on held-out test split. | [inference_time_comparison.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/results_analysis/inference_time_comparison.png) |
| **Fig. 8** | Deep Learning Parameter Counts | Trainable parameter complexity for 1D-CNN, BiLSTM, and CNN+BiLSTM. | [parameter_count_comparison.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/results_analysis/parameter_count_comparison.png) |
| **Fig. 9** | Model Artifact Size on Disk | Serialized binary storage size comparison in megabytes (log scale). | [model_size_comparison.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/results_analysis/model_size_comparison.png) |

---

## 2. Confusion Matrices ($34 \times 34$)

| Fig. # | Figure Title | Description | File Link |
| :---: | :--- | :--- | :--- |
| **Fig. 10** | Random Forest Confusion Matrix (Raw) | Raw $34 \times 34$ confusion matrix for Random Forest ($N=1,176,851$). | [random_forest_confusion_matrix.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/random_forest_confusion_matrix.png) |
| **Fig. 11** | Random Forest Confusion Matrix (Normalized) | Row-normalized confusion matrix showing recall proportions for Random Forest. | [random_forest_confusion_matrix_normalized.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/random_forest_confusion_matrix_normalized.png) |
| **Fig. 12** | XGBoost Confusion Matrix (Raw) | Raw $34 \times 34$ confusion matrix for XGBoost. | [xgboost_confusion_matrix.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/xgboost_confusion_matrix.png) |
| **Fig. 13** | XGBoost Confusion Matrix (Normalized) | Row-normalized recall confusion matrix for XGBoost. | [xgboost_confusion_matrix_normalized.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/xgboost_confusion_matrix_normalized.png) |
| **Fig. 14** | 1D-CNN Confusion Matrix (Raw) | Raw $34 \times 34$ confusion matrix for 1D-CNN. | [cnn_1d_confusion_matrix.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/cnn_1d_confusion_matrix.png) |
| **Fig. 15** | 1D-CNN Confusion Matrix (Normalized) | Row-normalized recall confusion matrix for 1D-CNN. | [cnn_1d_confusion_matrix_normalized.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/cnn_1d_confusion_matrix_normalized.png) |
| **Fig. 16** | BiLSTM Confusion Matrix (Raw) | Raw $34 \times 34$ confusion matrix for BiLSTM. | [bilstm_confusion_matrix.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/bilstm_confusion_matrix.png) |
| **Fig. 17** | BiLSTM Confusion Matrix (Normalized) | Row-normalized recall confusion matrix for BiLSTM. | [bilstm_confusion_matrix_normalized.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/bilstm_confusion_matrix_normalized.png) |
| **Fig. 18** | CNN + BiLSTM Confusion Matrix (Raw) | Raw $34 \times 34$ confusion matrix for Hybrid CNN+BiLSTM. | [cnn_bilstm_confusion_matrix.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/cnn_bilstm_confusion_matrix.png) |
| **Fig. 19** | CNN + BiLSTM Confusion Matrix (Normalized) | Row-normalized recall confusion matrix for Hybrid CNN+BiLSTM. | [cnn_bilstm_confusion_matrix_normalized.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/cnn_bilstm_confusion_matrix_normalized.png) |

---

## 3. Deep Learning Training History Curves

| Fig. # | Figure Title | Description | File Link |
| :---: | :--- | :--- | :--- |
| **Fig. 20** | 1D-CNN Training History | Loss and accuracy curves across 9 epochs for 1D-CNN. | [cnn_1d_training_history.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/cnn_1d_training_history.png) |
| **Fig. 21** | BiLSTM Training History | Loss and accuracy convergence curves across 15 epochs for BiLSTM. | [bilstm_training_history.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/bilstm_training_history.png) |
| **Fig. 22** | CNN + BiLSTM Training History | Training and validation trajectories across 4 epochs for CNN+BiLSTM. | [cnn_bilstm_training_history.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/cnn_bilstm_training_history.png) |
| **Fig. 23** | Comparative Neural Loss Curves | Overlay of training and validation loss for all three DL models. | [dl_loss_curves.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/training_curves/dl_loss_curves.png) |
| **Fig. 24** | Comparative Neural Accuracy Curves | Overlay of training and validation accuracy for all three DL models. | [dl_accuracy_curves.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/training_curves/dl_accuracy_curves.png) |

---

## 4. Error Diagnostics and Misclassification Figures

| Fig. # | Figure Title | Description | File Link |
| :---: | :--- | :--- | :--- |
| **Fig. 25** | Error Count by Model | Total test misclassification count comparison across all 5 models. | [error_count_by_model.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/error_analysis/error_count_by_model.png) |
| **Fig. 26** | Error Rate by Model | Percentage error rate on held-out test split. | [error_rate_by_model.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/error_analysis/error_rate_by_model.png) |
| **Fig. 27** | Benign vs. Attack Confusion | Binary security matrix comparing False Positives vs False Negatives. | [benign_vs_attack_confusion.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/error_analysis/benign_vs_attack_confusion.png) |
| **Fig. 28** | Class-Wise F1 Heatmap Comparison | Heatmap of per-class F1-scores across all 34 classes for all 5 models. | [classwise_f1_comparison.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/error_analysis/classwise_f1_comparison.png) |
| **Fig. 29** | Top Errors — XGBoost | Top 10 confusion pairs for XGBoost. | [top_misclassifications_xgboost.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/error_analysis/top_misclassifications_xgboost.png) |
| **Fig. 30** | Top Errors — Random Forest | Top 10 confusion pairs for Random Forest. | [top_misclassifications_random_forest.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/error_analysis/top_misclassifications_random_forest.png) |
| **Fig. 31** | Top Errors — BiLSTM | Top 10 confusion pairs for BiLSTM. | [top_misclassifications_bilstm.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/error_analysis/top_misclassifications_bilstm.png) |
| **Fig. 32** | Top Errors — 1D-CNN | Top 10 confusion pairs for 1D-CNN. | [top_misclassifications_cnn_1d.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/error_analysis/top_misclassifications_cnn_1d.png) |
| **Fig. 33** | Top Errors — CNN + BiLSTM | Top 10 confusion pairs for CNN + BiLSTM. | [top_misclassifications_cnn_bilstm.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/error_analysis/top_misclassifications_cnn_bilstm.png) |

---

## 5. SHAP Explainability Figures

| Fig. # | Figure Title | Description | File Link |
| :---: | :--- | :--- | :--- |
| **Fig. 34** | XGBoost Global SHAP Summary | Mean absolute SHAP attribution bar plot for top features in XGBoost. | [shap_global_summary_xgboost.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/shap_global_summary_xgboost.png) |
| **Fig. 35** | Random Forest Global SHAP Summary | Mean absolute SHAP attribution plot for top features in Random Forest. | [shap_global_summary_random_forest.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/shap_global_summary_random_forest.png) |
| **Fig. 36** | CNN + BiLSTM Global SHAP Summary | Deep SHAP global feature attribution summary for Hybrid CNN+BiLSTM. | [shap_global_summary_cnn_bilstm.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/shap_global_summary_cnn_bilstm.png) |
| **Fig. 37** | Local Attribution — BenignTraffic | Force plot explaining benign network flow decision. | [shap_local_instance_BenignTraffic.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/shap_local_instance_BenignTraffic.png) |
| **Fig. 38** | Local Attribution — DDoS-ICMP_Flood | Attribution waterfall plot for ICMP flood flow detection. | [shap_local_instance_DDoS_ICMP_Flood.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/shap_local_instance_DDoS_ICMP_Flood.png) |
| **Fig. 39** | Local Attribution — Mirai-greeth_flood | Attribution plot for Mirai GRE Ethernet attack flow. | [shap_local_instance_Mirai_greeth_flood.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/shap_local_instance_Mirai_greeth_flood.png) |
| **Fig. 40** | Local Attribution — Recon-PortScan | Attribution plot for port scanning traffic instance. | [shap_local_instance_Recon_PortScan.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/shap_local_instance_Recon_PortScan.png) |
| **Fig. 41** | Misclassification Explanation — UDP Flood | Attribution profile for `DDoS-UDP_Flood` misclassified as `DoS-UDP_Flood`. | [misclass_DDoS_UDP_Flood_to_DoS_UDP_Flood.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/misclassification_examples/misclass_DDoS_UDP_Flood_to_DoS_UDP_Flood.png) |
| **Fig. 42** | Misclassification Explanation — ARP Spoofing | Attribution profile for `BenignTraffic` misclassified as `DNS_Spoofing`. | [misclass_BenignTraffic_to_DNS_Spoofing.png](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/misclassification_examples/misclass_BenignTraffic_to_DNS_Spoofing.png) |
