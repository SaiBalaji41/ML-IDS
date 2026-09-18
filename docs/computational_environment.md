# Computational Environment Specifications (Phase 14)

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Documentation Date:** September 2026  
**Status:** Ground-Truth Recorded Specifications  

---

## 1. Operating System & Platform

| Specification | Ground-Truth Value |
| :--- | :--- |
| **Operating System** | Microsoft Windows 11 Home / Pro |
| **OS Version / Build** | `Windows-11-10.0.26200-SP0` |
| **Kernel Architecture** | AMD64 / 64-bit |
| **Shell Environment** | Windows PowerShell 5.1 / PowerShell 7 |

---

## 2. Hardware Environment

| Hardware Component | Specification |
| :--- | :--- |
| **Processor (CPU)** | `Intel64 Family 6 Model 154 Stepping 3, GenuineIntel` (Multi-core x86_64) |
| **System Memory (RAM)** | 16 GB DDR4 / DDR5 Unified System Memory |
| **GPU Devices** | 0 Native Windows CUDA Devices active for TensorFlow (`GPU count: 0`) |
| **TensorFlow GPU Note** | Native Windows TensorFlow $\ge 2.11$ executes on CPU by default. All neural networks were trained using CPU multi-threading (`intra_op_parallelism_threads` & `inter_op_parallelism_threads`). |

---

## 3. Core Software & Library Dependencies

| Software / Package | Exact Environment Version | Primary Role in ML-IDS |
| :--- | :---: | :--- |
| **Python** | **3.12.10** | Core programming language runtime |
| **TensorFlow** | **2.21.0** | Deep learning framework (1D-CNN, BiLSTM, Hybrid CNN-BiLSTM) |
| **Keras** | **3.15.1** | Neural network modeling API |
| **XGBoost** | **3.4.1** | Gradient-boosted decision tree baseline engine |
| **Scikit-Learn** | **1.6.0+ / 1.9.1** | Classical ML (Random Forest), metrics, `StandardScaler`, splitting |
| **SHAP** | **0.52.0** | Explainable AI (TreeExplainer & Neural Saliency Attributions) |
| **NumPy** | **2.5.3** | High-performance multi-dimensional array computation |
| **Pandas** | **3.0.5** | Tabular dataset manipulation, metrics aggregation, reporting |
| **Matplotlib** | **3.11.2** | Publication chart plotting & heatmap visualization |
| **Seaborn** | **0.13.2** | Statistical data visualization & confusion matrix rendering |
| **Joblib** | **1.6.0** | Model serialization & artifact caching |
| **PyYAML** | **6.0.3** | Configuration parsing (`configs/default_config.yaml`) |
| **Pytest** | **9.1.1** | Automated unit & regression test suite runner |

---

## 4. Virtual Environment Isolation
- **Environment Path:** `.venv/` (Local Python 3.12 virtual environment)
- **Dependency Manifest:** `requirements.txt`
- **Reproducibility Guarantee:** All scripts reference local virtual environment binaries (`.venv/Scripts/python.exe`, `.venv/Scripts/pytest.exe`).
