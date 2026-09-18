# Deep Learning Training Results Analysis

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Data Sources:** [`results/metrics/cnn_1d_history.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/cnn_1d_history.csv), [`results/metrics/bilstm_history.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/bilstm_history.csv), [`results/metrics/cnn_bilstm_history.csv`](file:///c:/Users/LENOVO/OneDrive/Desktop/Projects/ML-IDS/results/metrics/cnn_bilstm_history.csv)  
**Evaluated Architectures:** 1D-CNN, BiLSTM, CNN + BiLSTM

---

## 1. Training and Validation Trajectory Comparison

Table 1 summarizes the key training progression and validation dynamics across the three deep learning architectures.

### Table 1: Deep Learning Training Dynamics and Final Validation Metrics

| Model Architecture | Total Epochs | Best Validation Epoch | Initial Train Loss / Acc | Final Train Loss / Acc | Initial Val Loss / Acc | Best Val Loss / Acc | Learning Rate Schedule |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1D-CNN** | 9 | Epoch 4 | 1.7836 / 52.28% | 0.9802 / 70.92% | 0.8095 / 66.45% | **0.6588 / 66.20%** | $0.001 \rightarrow 0.00025$ |
| **BiLSTM** | 15 | Epoch 15 | 1.4890 / 52.00% | 0.7022 / 75.81% | 0.6364 / 74.35% | **0.4847 / 79.60%** | $0.001 \rightarrow 0.0005$ |
| **CNN + BiLSTM** | 4 | Epoch 4 | 2.4484 / 17.27% | 1.1418 / 60.75% | 2.4247 / 22.09% | **1.0283 / 62.57%** | $0.001$ (Constant) |

---

## 2. Model-by-Model Convergence and Stability Analysis

### 2.1 1D-CNN (Convolutional Neural Network)
- **Convergence Behavior:** The model experienced rapid initial loss descent from epoch 1 (loss 1.7836) to epoch 4 (loss 1.0967), achieving its lowest validation loss of **0.6588** at epoch 4.
- **Subsequent Epochs:** From epoch 5 to epoch 9, training loss continued to decrease gradually from 1.0646 to 0.9802, while validation loss fluctuated between 0.6613 and 0.7356. Early stopping was triggered after patience = 5.
- **Stability and Generalization:** The gap between final training accuracy (70.92%) and validation accuracy (61.92%–66.20%) remained moderate (~5–9 percentage points).

### 2.2 BiLSTM (Bidirectional Long Short-Term Memory)
- **Convergence Behavior:** BiLSTM demonstrated monotonic, stable convergence across all 15 epochs. Training loss steadily decreased from 1.4890 to 0.7022, while training accuracy rose from 52.00% to 75.81%.
- **Validation Progression:** Validation loss decreased continuously from 0.6364 (epoch 1) to **0.4847** (epoch 15), while validation accuracy climbed steadily from 74.35% to **79.60%** (epoch 15 peak: 79.78% at epoch 14).
- **Overfitting / Underfitting Assessment:** Validation loss remained consistently below training loss throughout all 15 epochs (partially due to dropout regularization of 0.3 applied during training passes). There was no indication of catastrophic overfitting.

### 2.3 Hybrid CNN + BiLSTM
- **Convergence Behavior:** The hybrid model started with high initial loss (2.4484, 17.27% accuracy at epoch 1) and showed substantial progress through epoch 4 (loss 1.1418, 60.75% training accuracy, 62.57% validation accuracy).
- **Early Termination:** Training halted after 4 epochs based on early stopping callback criteria.

---

## 3. Comparative Observations on Deep Learning Architectures

1. **Gradient Flow & Representation:** The BiLSTM architecture achieved a lower validation loss (0.4847) and higher validation accuracy (79.60%) than 1D-CNN (val loss 0.6588, val accuracy 66.20%).
2. **Computational Cost vs. Performance Tradeoff:** The sequential recurrence in BiLSTM required 74,830 seconds (~20.8 hours) of training on CPU for 15 epochs, compared to 682.75 seconds for 1D-CNN and 92.44 seconds for CNN + BiLSTM.
3. **Feature-Sequence Nature:** Because tabular features lack intrinsic temporal ordering, convolutional filters and recurrent gates operate over an arbitrary indexing sequence of 46 statistical summary features rather than actual sequential packet arrivals.
