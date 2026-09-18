# Research Paper Methods Section (Phase 14)

**Paper Title:** Machine Learning and Deep Learning Benchmarking with Explainable AI for High-Throughput Network Intrusion Detection  
**Target Venue:** IEEE Transactions on Network and Service Management / Elsevier Computer Networks  
**Section Number:** Section 3 — Proposed Experimental Methodology  

---

## 3. Methodology

### 3.1 Dataset Description
Experiments in this study are conducted using the **CICIoT2023** dataset, a large-scale network traffic benchmark designed for Internet of Things (IoT) security profiling. The raw dataset comprises 33 distinct CSV files containing **7,845,673 total network flow records** representing normal background activity and 33 malicious attack subclasses categorized across eight primary attack families (DDoS, DoS, Reconnaissance, Mirai Botnet, Spoofing, Web Attacks, Brute Force, and Benign). Each flow record is characterized by **46 engineered statistical features** capturing transport layer attributes, packet inter-arrival times, header dimensions, and control flag counts.

### 3.2 Data Preprocessing & Sanitization
To guarantee high-fidelity representation learning and prevent numerical instability:
1. **Data Quality Audit:** Raw records were audited for invalid strings, missing entries, and infinite numerical values, which were purged.
2. **Feature Sanitization:** Columns exhibiting zero statistical variance across the entire corpus were eliminated.
3. **Stateless Feature Scaling:** Continuous numerical features were transformed using standard z-score normalization ($z = (x - \mu)/\sigma$). To eliminate data leakage, the mean ($\mu$) and standard deviation ($\sigma$) parameters were fitted exclusively on the training partition and statelessly applied to the validation and test partitions.
4. **Target Encoding:** The 34 multiclass categorical labels were mapped to integer indices in $[0, 33]$ and serialized to `label_mapping.json`.

### 3.3 Stratified Data Splitting
The preprocessed corpus of 7,845,673 samples was partitioned using a **stratified split** to strictly preserve relative class frequencies across all splits:
- **Training Set (70%):** $5,491,971$ flows for parameter estimation.
- **Validation Set (15%):** $1,176,851$ flows for hyperparameter tuning and early stopping.
- **Test Set (15%):** $1,176,851$ flows strictly reserved for a single unbiased evaluation pass.
All splits were generated using a deterministic pseudo-random seed of **42**.

### 3.4 Baseline Machine Learning Models
Two classical ensemble algorithms were implemented as high-capacity baselines:
- **Random Forest:** An ensemble of $200$ randomized decision trees trained with maximum depth $25$, bootstrap sub-sampling ($10\%$ per tree), and balanced class weights to penalize minority misclassifications.
- **XGBoost:** A gradient-boosted decision tree classifier comprising $100$ estimators, maximum tree depth of $6$, learning rate $\eta=0.1$, subsample ratio $0.8$, and the histogram-based tree method (`tree_method='hist'`) optimizing multi-class logarithmic loss.

### 3.5 Standalone Deep Learning Models
To evaluate single-paradigm neural architectures, two neural baselines were constructed:
- **Standalone 1D-CNN:** Formed by two sequential 1D convolutional layers (`Conv1D(64, kernel=3)` $\rightarrow$ `MaxPool1D(2)` $\rightarrow$ `Conv1D(128, kernel=3)` $\rightarrow$ `GlobalAveragePooling1D`), followed by a fully connected dense layer ($128$ units) and a 34-way Softmax output layer ($46,626$ parameters).
- **Standalone BiLSTM:** Composed of two stacked Bidirectional LSTM layers (`BiLSTM(64)` $\rightarrow$ `BiLSTM(32)`) with intermediate dropout ($0.30$), projecting into a dense classification head ($64$ units) and Softmax output ($81,378$ parameters).

### 3.6 Proposed Hybrid CNN + BiLSTM Architecture
The proposed hybrid model integrates spatial feature extraction with bidirectional sequential context:
1. **Input Tensor:** Standardized 46-dimensional flow vectors reshaped to $[B, 46, 1]$.
2. **Spatial Feature Extraction:** A 1D convolutional layer ($64$ filters, kernel size $3$, ReLU activation) extracts localized multi-feature combinations, followed by `MaxPooling1D(pool_size=2)` to compress the representation.
3. **Bidirectional Recurrent Modeling:** A `Bidirectional(LSTM(64))` layer aggregates forward and backward contextual relationships across the pooled representation.
4. **Classification Head:** Dense layer ($64$ units, ReLU) with dropout ($0.30$) and a 34-class Softmax layer ($77,026$ total parameters).

### 3.7 Training Procedure & Hyperparameter Tuning
Neural network parameters were optimized using the **Adam** optimizer (initial learning rate $\alpha=0.001$) minimizing sparse categorical crossentropy:
$$\mathcal{L} = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i, c} \log(\hat{y}_{i, c})$$
Training utilized mini-batch stochastic gradient descent with batch size $512$. Convergence was monitored on the validation set using **Early Stopping** (patience $= 5$ epochs, restoring best weights) and **Learning Rate Reduction on Plateau** (factor $= 0.5$, patience $= 2$ epochs). Inverse class-frequency weighting was applied to mitigate minority attack under-representation.

### 3.8 Evaluation Metrics
Performance was assessed on the $1,176,851$-sample test set across standard multi-class metrics:
- **Accuracy:** Overall proportion of correctly classified flows.
- **Macro Precision, Recall, and F1-Score:** Unweighted arithmetic means across all 34 classes, evaluating sensitivity on rare exploit subclasses.
- **Weighted Metrics:** Class-support-weighted precision, recall, and F1-scores.
- **Computational Efficiency:** Wall-clock training duration and per-sample inference latency (milliseconds per flow).

### 3.9 Error Analysis Methodology
To isolate exact failure modes, 34x34 confusion matrices were computed and decomposed into One-vs-Rest (OvR) contingency tables ($TP_k, TN_k, FP_k, FN_k$) for every class $k$. Specific error analyses categorized misclassifications into:
1. Benign-to-Attack False Positives ($FP$)
2. Attack-to-Benign False Negatives ($FN$)
3. Attack-to-Attack Subclass Confusion (e.g., DoS vs. DDoS variants of identical protocols)

### 3.10 Explainable AI Using SHAP
To achieve model interpretability for SOC triage, **SHAP (SHapley Additive exPlanations)** was integrated:
- **Tree Ensembles:** Evaluated via `shap.TreeExplainer` computing exact polynomial-time tree-path Shapley attributions.
- **Neural Networks:** Evaluated using gradient saliency attributions across a reference background distribution ($X_{\text{bg}}$, $N=100$) sampled from the test set.
- **Global & Local Attributions:** Mean absolute SHAP values established global feature importance rankings, while local attribution plots explained individual predictions and specific Phase 12 misclassification events.

### 3.11 Experimental Environment
All experiments were executed on a dedicated system running **Windows 11** on an **Intel multi-core x86_64 architecture**, utilizing **Python 3.12.10**, **TensorFlow 2.21.0**, **Keras 3.15.1**, **XGBoost 3.4.1**, **Scikit-Learn 1.6.0+**, and **SHAP 0.52.0**.

### 3.12 Reproducibility & Open Science
All data partitioning, model initializations, and evaluation routines were executed under fixed random seed `42`. Serialized model binaries, scalers, label mappings, and evaluation scripts are stored in the project repository to enable exact deterministic replication.
