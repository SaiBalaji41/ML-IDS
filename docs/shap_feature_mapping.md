# Feature Representation & Preprocessing Mapping for SHAP (Phase 13)

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Dataset:** CICIoT2023 Network Traffic Benchmark  
**Feature Space:** 46 Engineered Network Flow Statistical Features  

---

## 1. Feature Space & Representation
The CICIoT2023 dataset is a **tabular network-flow dataset**. It contains extracted statistical flow attributes aggregated over bidirectional network conversations rather than raw packet payload bytes.

All 46 features fed into the machine learning and deep learning models correspond directly to network transport attributes, flag numbers, inter-arrival times, packet size distributions, and traffic rates.

---

## 2. Transformation Pipeline & Scaling Semantics

```text
Raw Network Flow Records (CICIoT2023 CSV)
                   │
                   ▼
       Data Quality & Sanitization
  (Drop infinite/NaN, zero-variance columns)
                   │
                   ▼
          StandardScaler Transformation
  (Zero mean μ=0, Unit variance σ=1 fitted on Train)
                   │
                   ▼
      Standardized 46-Dimensional Vector
                   │
                   ▼
         Model Feature Input Space
```

### Important Notice on SHAP Value Attribution:
> [!IMPORTANT]
> **Input Space Interpretation:**  
> Because the models were trained on features scaled via `StandardScaler`, all computed SHAP values explain the **model input space** (the standardized continuous representation). A positive SHAP value indicates that a feature's standardized deviation above or below the baseline increases the model's log-odds output toward a target attack class.
>
> SHAP values do **not** reflect raw unscaled packet units (e.g., raw bytes or seconds) directly, but rather the relative sensitivity of the model to perturbations in the normalized feature space.

---

## 3. Comprehensive 46-Feature Mapping Table

| Index | Feature Name | Protocol / Layer | Physical Network Meaning | Preprocessing Scaling |
| :---: | :--- | :---: | :--- | :---: |
| **0** | `flow_duration` | Transport | Total duration of the bidirectional network flow (seconds) | StandardScaler |
| **1** | `Header_Length` | Network / Transport | Total length of IP/TCP/UDP packet headers (bytes) | StandardScaler |
| **2** | `Protocol Type` | Network Layer | Encoded IP protocol identifier (e.g., 6=TCP, 17=UDP, 1=ICMP) | StandardScaler |
| **3** | `Duration` | Transport | Effective active conversational duration | StandardScaler |
| **4** | `Rate` | Transport | Overall packet transmission rate (packets/sec) | StandardScaler |
| **5** | `Srate` | Transport | Source-to-destination packet rate | StandardScaler |
| **6** | `Drate` | Transport | Destination-to-source packet rate | StandardScaler |
| **7** | `fin_flag_number` | TCP Layer | Binary presence of TCP FIN flag in flow | StandardScaler |
| **8** | `syn_flag_number` | TCP Layer | Binary presence of TCP SYN flag in flow | StandardScaler |
| **9** | `rst_flag_number` | TCP Layer | Binary presence of TCP RST flag in flow | StandardScaler |
| **10** | `psh_flag_number` | TCP Layer | Binary presence of TCP PSH flag in flow | StandardScaler |
| **11** | `ack_flag_number` | TCP Layer | Binary presence of TCP ACK flag in flow | StandardScaler |
| **12** | `ece_flag_number` | TCP Layer | Binary presence of ECN-Echo flag in flow | StandardScaler |
| **13** | `cwr_flag_number` | TCP Layer | Binary presence of Congestion Window Reduced flag | StandardScaler |
| **14** | `ack_count` | TCP Layer | Cumulative count of ACK packets observed | StandardScaler |
| **15** | `syn_count` | TCP Layer | Cumulative count of SYN packets observed | StandardScaler |
| **16** | `fin_count` | TCP Layer | Cumulative count of FIN packets observed | StandardScaler |
| **17** | `urg_count` | TCP Layer | Cumulative count of URG packets observed | StandardScaler |
| **18** | `rst_count` | TCP Layer | Cumulative count of RST packets observed | StandardScaler |
| **19** | `HTTP` | Application Layer | Binary indicator of HTTP port traffic | StandardScaler |
| **20** | `HTTPS` | Application Layer | Binary indicator of HTTPS (TLS/SSL 443) traffic | StandardScaler |
| **21** | `DNS` | Application Layer | Binary indicator of DNS (UDP 53) traffic | StandardScaler |
| **22** | `Telnet` | Application Layer | Binary indicator of Telnet (TCP 23) traffic | StandardScaler |
| **23** | `SMTP` | Application Layer | Binary indicator of SMTP (TCP 25) traffic | StandardScaler |
| **24** | `SSH` | Application Layer | Binary indicator of SSH (TCP 22) traffic | StandardScaler |
| **25** | `IRC` | Application Layer | Binary indicator of IRC protocol traffic | StandardScaler |
| **26** | `TCP` | Transport Layer | Binary protocol indicator for TCP | StandardScaler |
| **27** | `UDP` | Transport Layer | Binary protocol indicator for UDP | StandardScaler |
| **28** | `DHCP` | Application Layer | Binary protocol indicator for DHCP | StandardScaler |
| **29** | `ARP` | Link Layer | Binary protocol indicator for Address Resolution Protocol | StandardScaler |
| **30** | `ICMP` | Network Layer | Binary protocol indicator for ICMP control messages | StandardScaler |
| **31** | `IPv` | Network Layer | IP version identifier | StandardScaler |
| **32** | `LLC` | Link Layer | Logical Link Control frame indicator | StandardScaler |
| **33** | `Tot sum` | Statistical | Sum of packet length values in flow | StandardScaler |
| **34** | `Min` | Statistical | Minimum packet length observed in flow (bytes) | StandardScaler |
| **35** | `Max` | Statistical | Maximum packet length observed in flow (bytes) | StandardScaler |
| **36** | `AVG` | Statistical | Mean packet length observed across flow (bytes) | StandardScaler |
| **37** | `Std` | Statistical | Standard deviation of packet lengths in flow | StandardScaler |
| **38** | `Tot size` | Statistical | Aggregate volume of transmitted payload and headers | StandardScaler |
| **39** | `IAT` | Temporal | Mean Inter-Arrival Time between consecutive packets | StandardScaler |
| **40** | `Number` | Statistical | Total number of packets in the flow window | StandardScaler |
| **41** | `Magnitue` | Statistical | Geometric mean / Euclidean magnitude of flow features | StandardScaler |
| **42** | `Radius` | Statistical | Variance radius of packet size distributions | StandardScaler |
| **43** | `Covariance` | Statistical | Covariance across forward/backward packet streams | StandardScaler |
| **44** | `Variance` | Statistical | Second central moment of packet lengths | StandardScaler |
| **45** | `Weight` | Temporal | Exponential decay weight factor for recent packets | StandardScaler |
