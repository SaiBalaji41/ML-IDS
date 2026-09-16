# CICIoT2023 Dataset Setup and Exploratory Analysis Report

**Project:** ML-Powered Intrusion Detection System (IDS) for Secure Network Monitoring  
**Phase:** Phase 3 — CICIoT2023 Dataset Setup & Analysis  
**Analysis Date:** 2026-09-16  
**Status:** Complete  

---

## 1. Executive Summary
The CICIoT2023 dataset placed in `data/raw/` has been thoroughly analyzed. The dataset contains **7,845,673 total network traffic flow records** across 3 pre-partitioned CSV files:
- **`train.csv`:** 5,491,971 records (70.00% partition, 1.512 GB)
- **`validation.csv`:** 1,176,851 records (15.00% partition, 331.8 MB)
- **`test.csv`:** 1,176,851 records (15.00% partition, 331.8 MB)
- **Total Storage Size:** ~2.176 GB (2,319,327,087 bytes)

---

## 2. Actual Dataset Representation
> [!IMPORTANT]
> **Data Representation Finding:**  
> The raw data in `data/raw/` consists of **tabular statistical network flow and protocol features**.  
> - It is **NOT** raw packet payload byte data (no hex, ASCII, or byte payload streams).  
> - It is **NOT** raw packet capture (PCAP) files.  
> - All 46 input features are pre-computed flow summary metrics, packet size distributions, inter-arrival times, protocol indicators, and TCP flag counts.

---

## 3. Dataset Schema & Column Specifications
The dataset comprises **47 total columns**: **46 numeric input features** and **1 categorical target label** (`label`).

| # | Column Name | Type | Description / Role |
|---|---|---|---|
| 1 | `flow_duration` | Continuous (Float) | Duration of the bidirectional network flow (seconds) |
| 2 | `Header_Length` | Continuous (Float) | Total length of protocol headers (bytes) |
| 3 | `Protocol Type` | Discrete (Float) | Transport/network protocol numerical code |
| 4 | `Duration` | Continuous (Float) | Connection active time |
| 5 | `Rate` | Continuous (Float) | Overall packet transmission rate (packets/sec) |
| 6 | `Srate` | Continuous (Float) | Source-to-destination packet rate |
| 7 | `Drate` | Continuous (Float) | Destination-to-source packet rate |
| 8–14 | `fin_flag_number`, `syn_flag_number`, `rst_flag_number`, `psh_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number` | Binary (0.0 / 1.0) | TCP control flag presence indicators |
| 15–19 | `ack_count`, `syn_count`, `fin_count`, `urg_count`, `rst_count` | Continuous (Float) | Accumulated TCP flag frequencies |
| 20–33 | `HTTP`, `HTTPS`, `DNS`, `Telnet`, `SMTP`, `SSH`, `IRC`, `TCP`, `UDP`, `DHCP`, `ARP`, `ICMP`, `IPv`, `LLC` | Binary (0.0 / 1.0) | Protocol and application layer service indicators |
| 34–39 | `Tot sum`, `Min`, `Max`, `AVG`, `Std`, `Tot size` | Continuous (Float) | Statistical packet size aggregations and moments |
| 40 | `IAT` | Continuous (Float) | Packet Inter-Arrival Time |
| 41 | `Number` | Continuous (Float) | Aggregated packet count in flow window |
| 42 | `Magnitue` | Continuous (Float) | Geometric magnitude across flow attributes |
| 43 | `Radius` | Continuous (Float) | Dispersion radius of flow vectors |
| 44 | `Covariance` | Continuous (Float) | Inter-feature covariance estimate |
| 45 | `Variance` | Continuous (Float) | Packet length variance |
| 46 | `Weight` | Continuous (Float) | Flow weighting factor |
| 47 | `label` | Categorical (String) | **Target Ground Truth Class** (34 unique classes) |

---

## 4. Complete 34-Class Distribution
The dataset contains **34 distinct traffic classes** across Benign traffic and 33 attack variants:

| Class Name | Train Count | Validation Count | Test Count | Total Count | Percentage |
|:---|:---:|:---:|:---:|:---:|:---:|
| `DDoS-ICMP_Flood` | 848,088 | 182,011 | 180,447 | 1,210,546 | 15.4295% |
| `DDoS-UDP_Flood` | 637,558 | 136,466 | 136,717 | 910,741 | 11.6082% |
| `DDoS-TCP_Flood` | 528,499 | 113,888 | 113,735 | 756,122 | 9.6374% |
| `DDoS-PSHACK_Flood` | 481,254 | 102,985 | 103,326 | 687,565 | 8.7636% |
| `DDoS-SYN_Flood` | 478,653 | 102,644 | 102,208 | 683,505 | 8.7119% |
| `DDoS-RSTFINFlood` | 475,441 | 101,563 | 101,819 | 678,823 | 8.6522% |
| `DDoS-SynonymousIP_Flood` | 422,083 | 90,795 | 90,480 | 603,358 | 7.6903% |
| `DoS-UDP_Flood` | 390,422 | 83,446 | 83,627 | 557,495 | 7.1058% |
| `DoS-TCP_Flood` | 314,174 | 67,056 | 67,697 | 448,927 | 5.7220% |
| `DoS-SYN_Flood` | 237,573 | 51,115 | 51,116 | 339,804 | 4.3311% |
| `BenignTraffic` | 129,538 | 27,519 | 27,709 | 184,766 | 2.3550% |
| `Mirai-greeth_flood` | 116,133 | 24,910 | 24,934 | 165,977 | 2.1155% |
| `Mirai-udpplain` | 104,814 | 22,314 | 22,536 | 149,664 | 1.9076% |
| `Mirai-greip_flood` | 88,821 | 18,867 | 19,279 | 126,967 | 1.6183% |
| `DDoS-ICMP_Fragmentation` | 53,046 | 11,430 | 11,402 | 75,878 | 0.9671% |
| `MITM-ArpSpoofing` | 36,316 | 7,741 | 7,840 | 51,897 | 0.6615% |
| `DDoS-UDP_Fragmentation` | 34,169 | 7,221 | 7,224 | 48,614 | 0.6196% |
| `DDoS-ACK_Fragmentation` | 33,581 | 7,194 | 7,292 | 48,067 | 0.6127% |
| `DNS_Spoofing` | 21,214 | 4,565 | 4,570 | 30,349 | 0.3868% |
| `Recon-HostDiscovery` | 15,737 | 3,482 | 3,331 | 22,550 | 0.2874% |
| `Recon-OSScan` | 11,587 | 2,500 | 2,433 | 16,520 | 0.2106% |
| `Recon-PortScan` | 9,648 | 2,083 | 2,082 | 13,813 | 0.1761% |
| `DoS-HTTP_Flood` | 8,487 | 1,833 | 1,805 | 12,125 | 0.1545% |
| `VulnerabilityScan` | 4,396 | 905 | 913 | 6,214 | 0.0792% |
| `DDoS-HTTP_Flood` | 3,371 | 706 | 709 | 4,786 | 0.0610% |
| `DDoS-SlowLoris` | 2,757 | 620 | 622 | 3,999 | 0.0510% |
| `DictionaryBruteForce` | 1,541 | 290 | 319 | 2,150 | 0.0274% |
| `BrowserHijacking` | 665 | 158 | 134 | 957 | 0.0122% |
| `SqlInjection` | 590 | 148 | 148 | 886 | 0.0113% |
| `CommandInjection` | 620 | 132 | 119 | 871 | 0.0111% |
| `XSS` | 414 | 91 | 103 | 608 | 0.0077% |
| `Backdoor_Malware` | 392 | 93 | 89 | 574 | 0.0073% |
| `Recon-PingSweep` | 249 | 45 | 53 | 347 | 0.0044% |
| `Uploading_Attack` | 140 | 35 | 33 | 208 | 0.0027% |

---

## 5. Macro Traffic Category Breakdown
When aggregated into higher-level attack taxonomies:
1. **DDoS Attacks:** 5,038,799 samples (64.22%)
2. **DoS Attacks:** 1,358,351 samples (17.31%)
3. **Mirai IoT Botnet:** 442,608 samples (5.64%)
4. **Benign Traffic:** 184,766 samples (2.36%)
5. **Spoofing & MITM:** 82,246 samples (1.05%)
6. **Reconnaissance & Scanning:** 59,444 samples (0.76%)
7. **Web-Based Application Attacks:** 3,530 samples (0.045%)
8. **Brute Force & Malware:** 2,724 samples (0.035%)

---

## 6. Data Integrity & Leakage Verification
- **Missing Values:** Zero empty or null values detected across sampled partitions.
- **Infinite Values:** Zero infinite values detected.
- **Data Leakage Columns:** 
  - No static record identifiers or MAC addresses.
  - No source/destination IP addresses.
  - No source/destination port numbers.
  - No absolute timestamps.
  - All 46 features represent legitimate behavioral network flow dynamics.

---

## 7. Artifacts Created in Phase 3
- `notebooks/01_dataset_analysis.ipynb`
- `docs/dataset_analysis_report.md`
- `results/dataset_analysis/dataset_overview.json`
- `results/dataset_analysis/column_summary.csv`
- `results/dataset_analysis/class_distribution_total.csv`
- `results/dataset_analysis/class_distribution_train.csv`
- `results/dataset_analysis/class_distribution_validation.csv`
- `results/dataset_analysis/class_distribution_test.csv`
- `results/graphs/dataset_analysis/class_distribution_all.png`
- `results/graphs/dataset_analysis/class_distribution_top15.png`
- `results/graphs/dataset_analysis/split_distribution.png`
- `results/graphs/dataset_analysis/attack_categories.png`
