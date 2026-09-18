# Limitations of Results and Analysis

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Dataset:** CICIoT2023 Processed Dataset  
**Scope:** Methodological, Data-Centric, and Operational Limitations

---

## 1. Data-Centric Limitations

### 1.1 Synthetic Lab Testbed Generation
- The CICIoT2023 dataset was captured in a controlled laboratory testbed containing 105 IoT devices across 33 predefined attack scripts and benign behaviors.
- Synthetic network traffic patterns, while standardized, may not reflect the full behavioral heterogeneity, user background noise, and zero-day attack mutations encountered in live enterprise and industrial IoT networks.

### 1.2 Extreme Class Imbalance
- The dataset exhibits severe class imbalance, where high-rate flood attacks constitute over 70% of total flows, while web application intrusions (`SqlInjection`, `CommandInjection`, `Uploading_Attack`) represent fewer than 0.005% of instances.
- While tree models with balanced weighting retained partial recall on minority classes, standard mini-batch training for deep learning models resulted in zero true positives for classes with $N \le 150$.

---

## 2. Representation and Architectural Limitations

### 2.1 Tabular Flow Records vs. Deep Packet Payload Inspection
- The input dataset consists of 46 statistical flow summary features (inter-arrival times, packet sizes, flag counts, protocol numbers) aggregated over flow windows.
- Models do not inspect raw packet payload byte strings. Consequently, content-dependent payload attacks (e.g., specific SQL injection query strings, shellcode payloads, HTTP header tampering) must be inferred indirectly through statistical metadata anomalies.

### 2.2 Pseudo-Sequence Representation in Deep Learning
- For 1D-CNN, BiLSTM, and Hybrid CNN-BiLSTM, the 46 tabular features were expanded into a tensor of shape `(batch_size, 46, 1)`.
- This arrangement represents a *pseudo-sequence of indexed tabular features*, not a true temporal sequence of consecutive network packets. Convolutional kernels and LSTM gates therefore operate over feature indices rather than temporal arrival steps.

---

## 3. Evaluation and Deployment Limitations

### 3.1 Offline Batched Evaluation vs. Live Streaming Ingestion
- All reported metrics and latencies were measured using pre-extracted, pre-scaled offline test partitions (`.parquet` / `.npz`).
- These benchmarks do not include the live packet capture overhead (e.g., PCAP kernel-space buffer handling, flow reassembly, real-time feature extraction windowing) required in a physical network deployment.

### 3.2 Hardware and Execution Constraints
- Deep learning architectures were trained and benchmarked on multi-threaded CPU environments without hardware GPU acceleration telemetry.
- Long-duration training (e.g., 74,830 seconds for BiLSTM) constrained extensive hyperparameter grid search across deep neural configurations.

### 3.3 Explainability Boundaries
- SHAP attribution explains feature influence on model decision boundaries; it does not constitute physical causal proof of network compromise.
