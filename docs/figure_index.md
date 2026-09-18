# Complete Figure & Visual Artifact Index (Phase 14)

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Repository Path:** `results/graphs/`, `results/confusion_matrices/`, `results/shap/`  

---

## 1. Experimental Methodology & Pipeline Figures

| Figure # | Figure Title | File Path | Detailed Description |
| :---: | :--- | :--- | :--- |
| **Fig. 1** | End-to-End Experimental Pipeline | [`results/graphs/experimental_setup/experimental_pipeline.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/experimental_setup/experimental_pipeline.png) | High-level flowchart of data ingestion, preprocessing, 5-model evaluation, confusion matrix diagnostics, and XAI. |

---

## 2. Multi-Model Benchmark & Performance Comparison Figures

| Figure # | Figure Title | File Path | Detailed Description |
| :---: | :--- | :--- | :--- |
| **Fig. 2** | Model Accuracy Comparison | [`results/graphs/model_comparison/accuracy_comparison.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/model_comparison/accuracy_comparison.png) | Bar chart comparing classification accuracy across all 5 models on 1,176,851 test flows. |
| **Fig. 3** | Macro Precision Comparison | [`results/graphs/model_comparison/precision_macro_comparison.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/model_comparison/precision_macro_comparison.png) | Unweighted macro precision across all 34 classes for each model architecture. |
| **Fig. 4** | Macro Recall Comparison | [`results/graphs/model_comparison/recall_macro_comparison.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/model_comparison/recall_macro_comparison.png) | Unweighted macro recall across all 34 classes evaluating attack coverage. |
| **Fig. 5** | Macro F1-Score Comparison | [`results/graphs/model_comparison/f1_macro_comparison.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/model_comparison/f1_macro_comparison.png) | Balanced macro harmonic mean comparison across all 34 classes. |
| **Fig. 6** | Training Time Comparison | [`results/graphs/model_comparison/training_time_comparison.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/model_comparison/training_time_comparison.png) | Wall-clock computational training duration (seconds) per model. |
| **Fig. 7** | Inference Latency Benchmark | [`results/graphs/model_comparison/inference_time_comparison.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/model_comparison/inference_time_comparison.png) | Total test set inference time evaluating real-time classification capability. |

---

## 3. Deep Learning Training & Convergence Curves

| Figure # | Figure Title | File Path | Detailed Description |
| :---: | :--- | :--- | :--- |
| **Fig. 8** | Deep Learning Loss Convergence | [`results/graphs/training_curves/dl_loss_curves.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/training_curves/dl_loss_curves.png) | Epoch-by-epoch training vs validation crossentropy loss for 1D-CNN, BiLSTM, and Hybrid CNN-BiLSTM. |
| **Fig. 9** | Deep Learning Accuracy Convergence | [`results/graphs/training_curves/dl_accuracy_curves.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/training_curves/dl_accuracy_curves.png) | Epoch-by-epoch training vs validation accuracy trajectories across neural architectures. |

---

## 4. Confusion Matrix & Diagnostic Heatmaps

| Figure # | Figure Title | File Path | Detailed Description |
| :---: | :--- | :--- | :--- |
| **Fig. 10** | Random Forest Raw Confusion Matrix | [`results/confusion_matrices/random_forest_confusion_matrix.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/random_forest_confusion_matrix.png) | 34x34 raw sample count confusion matrix for Random Forest baseline. |
| **Fig. 11** | Random Forest Normalized Confusion Matrix | [`results/confusion_matrices/random_forest_confusion_matrix_normalized.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/random_forest_confusion_matrix_normalized.png) | Per-class normalized recall confusion heatmap for Random Forest. |
| **Fig. 12** | XGBoost Raw Confusion Matrix | [`results/confusion_matrices/xgboost_confusion_matrix.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/xgboost_confusion_matrix.png) | 34x34 raw count confusion matrix for XGBoost baseline (Champion). |
| **Fig. 13** | XGBoost Normalized Confusion Matrix | [`results/confusion_matrices/xgboost_confusion_matrix_normalized.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/xgboost_confusion_matrix_normalized.png) | Per-class normalized recall confusion heatmap for XGBoost baseline. |
| **Fig. 14** | 1D-CNN Raw Confusion Matrix | [`results/confusion_matrices/cnn_1d_confusion_matrix.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/cnn_1d_confusion_matrix.png) | 34x34 raw count confusion matrix for Standalone 1D-CNN. |
| **Fig. 15** | 1D-CNN Normalized Confusion Matrix | [`results/confusion_matrices/cnn_1d_confusion_matrix_normalized.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/cnn_1d_confusion_matrix_normalized.png) | Per-class normalized recall confusion heatmap for 1D-CNN. |
| **Fig. 16** | BiLSTM Raw Confusion Matrix | [`results/confusion_matrices/bilstm_confusion_matrix.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/bilstm_confusion_matrix.png) | 34x34 raw count confusion matrix for Standalone BiLSTM. |
| **Fig. 17** | BiLSTM Normalized Confusion Matrix | [`results/confusion_matrices/bilstm_confusion_matrix_normalized.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/bilstm_confusion_matrix_normalized.png) | Per-class normalized recall confusion heatmap for BiLSTM. |
| **Fig. 18** | CNN + BiLSTM Raw Confusion Matrix | [`results/confusion_matrices/cnn_bilstm_confusion_matrix.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/cnn_bilstm_confusion_matrix.png) | 34x34 raw count confusion matrix for Proposed Hybrid CNN + BiLSTM. |
| **Fig. 19** | CNN + BiLSTM Normalized Matrix | [`results/confusion_matrices/cnn_bilstm_confusion_matrix_normalized.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/confusion_matrices/cnn_bilstm_confusion_matrix_normalized.png) | Per-class normalized recall confusion heatmap for Hybrid CNN-BiLSTM. |

---

## 5. Error Analysis & Misclassification Diagnostic Figures

| Figure # | Figure Title | File Path | Detailed Description |
| :---: | :--- | :--- | :--- |
| **Fig. 20** | Total Error Count by Model | [`results/graphs/error_analysis/error_count_by_model.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/error_analysis/error_count_by_model.png) | Comparison of absolute misclassified flow counts across all 5 models. |
| **Fig. 21** | Empirical Error Rate (%) by Model | [`results/graphs/error_analysis/error_rate_by_model.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/error_analysis/error_rate_by_model.png) | Percentage misclassification rate on 1,176,851 test flows. |
| **Fig. 22** | Benign vs Attack Error Decomposition | [`results/graphs/error_analysis/benign_vs_attack_confusion.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/error_analysis/benign_vs_attack_confusion.png) | Decomposition of False Positives, False Negatives, and Attack-to-Attack subclass confusion. |
| **Fig. 23** | Class-wise F1 Heatmap Across Models | [`results/graphs/error_analysis/classwise_f1_comparison.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/graphs/error_analysis/classwise_f1_comparison.png) | Heatmap matrix of all 34 classes against all 5 models comparing per-class F1-scores. |

---

## 6. Explainable AI (SHAP) Attribution Figures

| Figure # | Figure Title | File Path | Detailed Description |
| :---: | :--- | :--- | :--- |
| **Fig. 24** | Random Forest Global SHAP Summary | [`results/shap/random_forest/random_forest_shap_summary.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/random_forest/random_forest_shap_summary.png) | Top 15 global feature importance bar plot for Random Forest using TreeExplainer. |
| **Fig. 25** | XGBoost Global SHAP Summary | [`results/shap/xgboost/xgboost_shap_summary.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/xgboost/xgboost_shap_summary.png) | Top 15 global feature importance bar plot for XGBoost baseline using TreeExplainer. |
| **Fig. 26** | 1D-CNN Neural Attribution Summary | [`results/shap/cnn_1d/cnn_1d_shap_summary.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/cnn_1d/cnn_1d_shap_summary.png) | Saliency attribution magnitude bar plot for Standalone 1D-CNN. |
| **Fig. 27** | BiLSTM Neural Attribution Summary | [`results/shap/bilstm/bilstm_shap_summary.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/bilstm/bilstm_shap_summary.png) | Saliency attribution magnitude bar plot for Standalone BiLSTM. |
| **Fig. 28** | CNN-BiLSTM Neural Attribution Summary | [`results/shap/cnn_bilstm/cnn_bilstm_shap_summary.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/cnn_bilstm/cnn_bilstm_shap_summary.png) | Saliency attribution magnitude bar plot for Proposed Hybrid CNN + BiLSTM. |
| **Fig. 29** | Local SHAP: DDoS-ICMP Flood | [`results/shap/class_specific/shap_class_DDoS_ICMP_Flood.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/class_specific/shap_class_DDoS_ICMP_Flood.png) | Local attribution plot demonstrating ICMP protocol and Header_Length dominance. |
| **Fig. 30** | Local SHAP: Mirai GRE Flood | [`results/shap/class_specific/shap_class_Mirai_greeth_flood.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/class_specific/shap_class_Mirai_greeth_flood.png) | Local attribution plot demonstrating GRE encapsulation protocol attribution. |
| **Fig. 31** | Local SHAP: Recon PortScan | [`results/shap/class_specific/shap_class_Recon_PortScan.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/class_specific/shap_class_Recon_PortScan.png) | Local attribution plot demonstrating SYN flag and short flow duration attribution. |
| **Fig. 32** | Misclassification: Benign to Recon-OSScan | [`results/shap/misclassification_examples/misclass_BenignTraffic_to_Recon_OSScan.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/misclassification_examples/misclass_BenignTraffic_to_Recon_OSScan.png) | Local attribution waterfall explaining why benign traffic was misclassified as an OS scan. |
| **Fig. 33** | Misclassification: DDoS-UDP to DoS-UDP | [`results/shap/misclassification_examples/misclass_DDoS_UDP_Flood_to_DoS_UDP_Flood.png`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/shap/misclassification_examples/misclass_DDoS_UDP_Flood_to_DoS_UDP_Flood.png) | Local attribution plot explaining DoS vs DDoS single-source/multi-source confusion. |
