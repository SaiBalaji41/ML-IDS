# ML-Powered Intrusion Detection System for Secure Network Monitoring

**Sentinel** is a local IDS dashboard with a trained 1D-CNN + BiLSTM classifier, XGBoost baseline, actual model inference, CSV flow analysis, local SHAP explanations and bounded Scapy packet observation.

The interface uses white and cool-gray surfaces, dark typography, blue primary actions, **30 px button height**, **16 px (rounded-2xl) cards**, consistent padding, thin borders and **no box shadows**. It has no CDN, external font or chart-service dependency.

## Run the delivered system

The Mac environment is already installed in `.venv-macos`:

```bash
./start.sh
```

Open **http://127.0.0.1:8765**. On macOS you can also double-click `Start-IDS.command`. The server listens on loopback only. Port 8080 on the development machine was already occupied by nginx.

For a clean Python 3.12 installation:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# macOS only: brew install libomp
python -m pip install -r requirements.txt
python dashboard/server.py --port 8765
```

On Windows use `py -3.12 -m venv .venv`, `.venv\Scripts\python -m pip install -r requirements.txt`, then `start.bat`. Live capture needs the platform's capture permissions and, on Windows, Npcap. The supplied old `.venv` is a Windows environment and cannot run on macOS.

## What is working

- **Overview:** run batches or continuous held-out replay with the selected model; inspect predictions, confidence, mismatches, session statistics and export events. Replay is labeled explicitly and does not invent network addresses.
- **Traffic analyzer:** choose a real held-out example, submit all 46 flow features, or upload a CSV with up to 500 rows. Input validation rejects missing, unknown, nonnumeric and nonfinite features. JSON batch predictions can be downloaded.
- **Model performance:** new training history, full test metrics, binary confusion matrix, all 34 class metrics and historical baseline comparisons. The complete 34 × 34 matrix is in the evaluation JSON.
- **Explainability:** local SHAP permutation attribution computed from the selected model, using 16 training background rows. Positive and negative probability contributions and additivity residual are shown. One permutation is approximate; attribution is not proof of an attack.
- **Deployment:** model/data readiness, interface discovery, bounded Scapy capture (60 seconds or 1,000 packets), and packet metadata. Capture returns explicit OS permission errors when access is unavailable.
- **Streaming CLI:** held-out replay, validated Kafka messages and PCAP payload inference using a matching byte checkpoint. Kafka messages are written to local JSONL; the optional broker is not needed for the dashboard.

```bash
.venv-macos/bin/python scripts/stream_ids.py --mode replay --samples 20
.venv-macos/bin/python -m pytest -q
.venv-macos/bin/python scripts/verify_delivery.py
```

## Dataset verification and representation

`/Users/harsha/Downloads/CICIOT23` contains three CSVs: train, validation and test. SHA-256 checks confirm they are identical to this project's `data/raw` partitions. Each has **46 numeric flow features plus `label`**, not packet payload bytes. The source audit is recorded in `output/delivery/dataset_source_audit.json`.

The available processed data contains 5,491,971 training flows, 1,176,851 validation flows and 1,176,851 test flows, with 34 labels including benign traffic.

The running hybrid processes ordered flow features as `(46, 1)`. Its BiLSTM sequence represents feature positions, **not a captured packet-time sequence**. Raw bytes must never be fed into this model. The provided PDF is a reference: it recommends additional production services and emphasizes feature consistency. It is not a mandate to claim those services are deployed.

## Completed training run

The new model was trained from scratch for 25 epochs on a seed-42 stratified sample of **200,000 training flows** with **40,000 validation flows**. The saved train-only scaler is followed by a fixed signed `log1p` transform, applied identically at inference. The architecture uses two Conv1D layers, max pooling, a bidirectional LSTM, dense layers and dropout. Early stopping/checkpoint selection uses validation loss; the full held-out test set is evaluated after checkpoint selection.

| Metric | New CNN-BiLSTM |
|---|---:|
| Test flows | 1,176,851 |
| Multiclass accuracy | 86.68% |
| Macro precision | 59.13% |
| Macro recall | 59.77% |
| Macro F1 | 58.34% |
| Attack-vs-benign precision | 99.71% |
| Attack-vs-benign recall | 99.49% |
| Attack-vs-benign F1 | 99.60% |

There were **3,304 false positives among 27,709 benign flows (11.92%)** and **5,822 false negatives among 1,149,142 attack flows (0.51%)**. High aggregate attack recall does not establish production readiness. Minority-class performance remains limited. The supplied split's original capture/session independence has not been established.

The previous hybrid reported 62.47% accuracy and 34.65% macro F1. All original checkpoints and research metrics remain preserved; the new model is in `models/deployed`, with its hash, input transform and labels in `metadata.json`. New evidence is in `output/delivery`.

To reproduce the new training run in the original project with the full processed partitions:

```bash
.venv-macos/bin/python scripts/train_delivery.py --epochs 25 --train-samples 200000 --validation-samples 40000
```

Restart the server after training to load the new checkpoint. The handoff ZIP includes the full test partition for inference; attach the original train/validation partitions before retraining from that ZIP.

## Raw-payload extension: data still required

The raw-byte preprocessing and training/inference code is implemented, but **no labeled PCAP/raw-byte dataset was supplied**. Consequently no packet-byte checkpoint is claimed as trained, and live packet observations are not labeled as attacks by the flow model.

Prepare a CSV manifest with `path,label,split`, where each capture contains one verified label and split is `train`, `validation` or `test`. Split independent source captures before making the manifest. Duplicate capture files are rejected. Mixed-label PCAPs require packet-level labeling before using this trainer.

```csv
path,label,split
captures/benign-train.pcap,BenignTraffic,train
captures/attack-train.pcap,Attack,train
captures/benign-val.pcap,BenignTraffic,validation
captures/attack-val.pcap,Attack,validation
captures/benign-test.pcap,BenignTraffic,test
captures/attack-test.pcap,Attack,test
```

```bash
python scripts/train_packet_bytes.py --manifest data/packet_manifest.csv --benign-label BenignTraffic
python scripts/stream_ids.py --mode pcap --pcap path/to/capture.pcap
```

The byte pipeline extracts Scapy `Raw` payload, truncates or right-pads to 256 bytes and normalizes by 255, consistently across training, PCAP and live inference. Payloadless packets are unclassified. After valid training, the capture service loads `models/packet_bytes`.

## Optional Apache Kafka

Docker was not running on the development machine, so a real broker round-trip has **not** been verified. Schema routing and model inference are covered by tests. To run the included local broker configuration:

```bash
docker compose -f compose.kafka.yaml up -d
python scripts/stream_ids.py --mode kafka --broker localhost:9092 --topic ids-flows
```

Publish JSON with `representation: "flow_features"`, `model: "cnn_bilstm"` and a `features` object containing all 46 fields. For a trained raw-byte model use `representation: "raw_payload_bytes"` and `payload_hex`. Consumer offsets are committed after a prediction or validation rejection is persisted to `runtime/events.jsonl`. Transient model/runtime failures do not commit the failed message. This is an at-least-once consumer, so retries can duplicate persisted events.

## Delivery artifacts

- `output/ML-IDS-Delivery.zip`: runnable inference source, deep-learning checkpoints, XGBoost, scaler, full held-out test partition and evidence. It includes the 21 inference/preprocessing/capture regression tests; all 86 tests, including historical artifact checks, run in the original project.
- `output/delivery/delivery_report.md`: exact delivered scope and limitations.
- `output/delivery/evaluation.json`: complete metrics, 34-class report and confusion matrices.
- `output/delivery/training_history.json`: all epoch metrics.
- `output/delivery/verification.json`: actual API calls and model outputs.
- `output/delivery/package_manifest.json`: per-file SHA-256 package inventory.
- `docs/HISTORICAL_RESEARCH_README.md`: earlier research snapshot, superseded for current runtime behavior.

The dashboard is an academic/local demonstration, not an authenticated production deployment or an intrusion prevention system. It does not block traffic or send external notifications.
