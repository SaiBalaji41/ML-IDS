# ML-Powered Intrusion Detection System for Secure Network Monitoring

The active project is **Sentinel: a PyTorch 1D-CNN + BiLSTM packet-payload IDS with a local Streamlit dashboard and byte-level SHAP explanations**. Inputs preserve their byte order as 1,024-byte one-dimensional signals. No packet-to-image conversion, NSL-KDD, TensorFlow or Kafka is used in this application.

**Current delivery status:** the application and training pipeline are implemented. The supplied `/Users/harsha/Downloads/CICIOT23` folder contains **three flow CSVs, with 46 statistics plus a label, and zero PCAP files**. Real CICIoT2023 byte-model training, research evaluation and trained-model monitoring remain pending raw labeled captures. The dashboard shows this state without substituting historical flow metrics. Synthetic fixtures exercise training/inference/SHAP only in automated tests and cannot load as a deployed model.

## Run locally

Open the [hosted Streamlit dashboard](https://sentinel-ml-ids.streamlit.app), or see [Cloud deployment](docs/CLOUD_DEPLOYMENT.md) for configuration and troubleshooting. It runs `cloud/app.py` from the `streamlit-cloud` branch with Python 3.12. The hosted entrypoint supports uploaded packet analysis and keeps capture and training controls local. Trained detection and SHAP remain pending labeled raw PCAPs and a verified checkpoint.

The Python 3.12 environment on this Mac is installed in `.venv-pytorch`:

```bash
./start.sh
```

Open **http://127.0.0.1:8501**. Double-clicking `Start-IDS.command` also starts the app. It listens on loopback only. The interface uses blue accents, white surfaces and dark text, **30 px buttons**, **16 px rounded cards**, consistent spacing, thin borders and **no box shadows**.

For a new installation:

```bash
python3.12 -m venv .venv-pytorch
source .venv-pytorch/bin/activate
python -m pip install -r requirements.txt
./start.sh
```

On Windows create `.venv-pytorch` with `py -3.12 -m venv .venv-pytorch`, install requirements using that environment's Python, then run `start.bat`. Python 3.12 is the verified runtime. `requirements.lock.txt` records the full tested environment; platform-specific wheels may differ.

## Dashboard workflow

1. **Overview:** model readiness, held-out payload replay with start/pause, alert counts, confidence review flags, detection feed and JSON export. Every simulated event is an actual saved-model prediction on a held-out payload. The dashboard does not generate synthetic attacks.
   The feed includes time, actual source IP, transport protocol, known ground truth, model prediction, confidence, severity and MATCH/MISMATCH/UNVERIFIED status. Search and filter the feed, export CSV/JSON, or select a payload for inspection and SHAP. Missing addresses are shown as unavailable. Protocol names come from packet headers; a port number alone is not labeled HTTP/TLS.
2. **Live capture & PCAP:** bounded passive Scapy capture on a selected interface, a bounded queue with an inference worker, explicit dropped-packet counters, stop controls and optional local PCAP recording. Without a trained model, packets are observed but not classified. Batch-analyze uploaded captures up to 32 MB and 5,000 packets. Ground truth is unknown for live and unlabeled uploads.
3. **Packet inspector:** paste hexadecimal bytes or select a packet from a PCAP/PCAPNG upload. Inspect normalization, padding and truncation immediately. Classification activates when a compatible trained checkpoint exists.
4. **Byte explanations:** SHAP expected gradients for the predicted-class probability, signed contributions at original byte offsets, a scrolling 1D hex strip, strongest bytes, background probability and sampling residual. Export all 1,024 contributions as JSON.
5. **Training & evaluation:** prepare captures, start training, follow learning curves, and view held-out metrics and confusion matrices. It also provides nested group cross-validation and capture-level bootstrap intervals. Refresh or select the output model directory after deployment training finishes.

Severity uses an explicit triage policy: low-confidence predictions are Review, benign predictions Info, DDoS/Mirai predictions Critical, recon predictions Medium, and other attack predictions High. It is not a learned or calibrated risk measurement. MATCH requires an actual known label; it is never assigned to live traffic without ground truth.

## Live capture permissions

The local macOS verification reached the capture device but was denied permission to open `/dev/bpf0`. Saved-PCAP ingestion, queuing, inference and recording passed software integration tests with Scapy's offline reader. A successful live-network capture is **not** claimed for this machine. Do not run the Streamlit server as root.

Use an existing authorized capture in the PCAP tab, or run only the bounded CLI with the OS privileges required for capture. For example, select your actual interface and, on macOS, run this from the project folder:

```bash
sudo .venv-pytorch/bin/python scripts/packet_pipeline.py capture --interface en0 --seconds 30 --observe-only --save-directory runtime/captures
```

The CLI does not need a model in `--observe-only` mode. To perform model inference, remove that option after real training is complete. The UI capture stops after its duration or packet limit; it never captures indefinitely. New captures have no attack labels and do not substitute for labeled CICIoT2023 training data.

## Nested validation and bootstrap

After preparing sufficient independent captures, run:

```bash
.venv-pytorch/bin/python scripts/packet_pipeline.py validate --outer-folds 5 --inner-folds 3 --bootstrap 100 --epochs 5
```

The train and validation partitions form the development pool. The deployment test partition is excluded. Five outer folds evaluate the selection procedure; three inner folds choose between two predefined CNN–BiLSTM architectures using mean macro F1. Fixed epochs per fit prevent outer-fold tuning or stopping. This default experiment trains 35 models and may be expensive. Every fold must retain every class; insufficient independent capture groups cause an explicit error, never a fallback to random packet splitting.

One hundred bootstrap draws resample whole capture groups from the out-of-fold predictions, reporting means, standard deviations and 95% percentile intervals. These intervals are conditional on the fitted out-of-fold predictions and do not include repeated model refitting. Outputs include `cross_validation.json`, `folds.json`, `out_of_fold.npz` and `status.json` in `output/pytorch/nested_validation`. They do not replace the deployment checkpoint. Until labeled PCAPs are provided, only test-fixture execution has been verified; there are no real CICIoT2023 validation results.

## Obtain the required data

The [official CICIoT2023 dataset page](https://www.unb.ca/cic/datasets/iotdataset-2023.html) distinguishes raw PCAP traffic from extracted CSV features. The official raw download currently requires registration. Download the **raw PCAP edition**, retain its provenance and verify the labels. The already downloaded CSVs cannot reconstruct discarded packet bytes.

The repository includes an empty `data/packet_manifest.csv` with the required header. Fill it using `docs/packet_manifest.example.csv` as a guide. Paths are relative to the manifest unless absolute. Every row needs `path,label,split,capture_id,dataset`; `dataset` must be `CICIoT2023`. `source_url` is optional provenance. Example filenames are placeholders, not supplied data.

If the manifest is missing, use **Create blank manifest** in Training & evaluation, or run `.venv-pytorch/bin/python scripts/packet_pipeline.py init-manifest`. Existing files are never overwritten. The dashboard checks the manifest and capture paths before enabling **Prepare dataset**. An empty manifest correctly remains unavailable for training until verified PCAP rows are added. The CLI now reports a clear setup error without a Python traceback when a manifest or capture is missing. Creating this CSV does not create or recover packet data.

All files from one capture session must share a `capture_id` and one split. Use independent capture sessions for train/validation/test and include every intended class in each. The default benign label is `BenignTraffic`; output classes are derived from the manifest, supporting binary or multiclass classification. Do not randomly split packet rows from one capture across partitions. Each manifest row assumes a verified homogeneous label: mixed-label captures need packet-level ground truth and separation into homogeneous files first, retaining the parent capture ID. Software validates schema, file identity and grouping; it cannot independently prove a declared capture's origin or labels.

```bash
.venv-pytorch/bin/python scripts/packet_pipeline.py audit --path /Users/harsha/Downloads/CICIOT23
.venv-pytorch/bin/python scripts/packet_pipeline.py prepare --manifest data/packet_manifest.csv --output data/packet_processed --max-per-capture 10000
.venv-pytorch/bin/python scripts/packet_pipeline.py train --data data/packet_processed --output models/pytorch_packet_ids --epochs 25 --batch-size 128
```

Choose fresh output directories for additional runs; existing artifacts are preserved. A convenience command combines preparation and training:

```bash
.venv-pytorch/bin/python scripts/train_packet_bytes.py --manifest data/packet_manifest.csv
```

Reservoir sampling scans each complete capture, avoiding first-N-only selection. The default limit is 10,000 payload candidates per capture. Default training uses 25 maximum epochs, batch size 128, seed 42, AdamW, class weights, validation-loss checkpoint selection, a learning-rate scheduler, gradient clipping and early stopping. The test partition is evaluated after checkpoint selection. Runtime depends on capture volume and CPU performance.

## Model and data contract

`packet_ids/preprocessing.py` is shared by preparation, inference and SHAP. It extracts TCP/UDP/ICMP transport payloads using Scapy, including decoded application layers such as DNS, excludes headers and link padding, keeps the first 1,024 bytes, right-pads shorter inputs with zero, then divides by 255. It skips empty payloads and IP fragments; it does not reassemble sessions. Inputs are `(batch, 1, 1024)` float32 tensors.

Architecture: Conv1D/ReLU/MaxPool → Conv1D/ReLU/MaxPool → 64 ordered byte-region features → bidirectional LSTM → dense classifier. The BiLSTM models regions **within one packet**, not temporal relationships between separate packets. The default network has 32/64 convolution channels and 48 hidden units per LSTM direction. Only PyTorch modules are used.

Capture files are SHA-256 checked. Identical model-visible payloads, including duplicates after padding/truncation, are excluded from later splits. Conflicting labels within a partition are removed; training selection does not depend on validation/test labels. Every split must retain all required classes. This evaluates unseen payloads, so metrics can differ from studies retaining duplicate traffic. Prepared arrays use uint8 storage and are verified by checksum before training.

Training writes `model.pt`, `metadata.json`, `evaluation.json`, `history.json`, `status.json`, training-only `shap_background.npy`, and up to 500 held-out payloads in `replay.npz`. The service validates input schema, framework, model checksum and label mapping. Confidence below 60% is flagged for review; it is not a calibrated risk score or an unknown-attack detector.

SHAP uses `GradientExplainer` expected gradients against up to 16 training references. Its target is the predicted class probability. Padding positions are held constant and contribute zero. Results are approximate; the residual reports the difference between predicted probability and baseline plus summed contributions. Increasing sample count can improve the estimate. SHAP explains model sensitivity, not causal proof of an attack.

**Payload-only limitation:** empty-payload attacks, including some floods/scans, cannot be scored by this representation. Header cues, timing, flow statistics, bytes beyond offset 1023, encrypted semantics and session reassembly are unavailable. Captures without sufficient usable bytes for a required class fail preparation. Detection performance must be established after real capture training.

## CLI inference and verification

Once a trained checkpoint exists:

```bash
.venv-pytorch/bin/python scripts/packet_pipeline.py predict --hex '00 7f ff'
.venv-pytorch/bin/python scripts/packet_pipeline.py explain --hex '00 7f ff' --samples 256
.venv-pytorch/bin/python scripts/packet_pipeline.py replay --count 10
```

Software checks:

```bash
.venv-pytorch/bin/python -m pytest packet_tests -q
.venv-pytorch/bin/python scripts/build_packet_delivery.py
```

Tests train a small network on synthetic temporary PCAPs and verify gradients, weight updates, checkpoint reload, inference, SHAP, data separation and Streamlit interactions. They are software tests, **not CICIoT2023 performance measurements**. No synthetic checkpoint is delivered as an IDS model.

The current package is `output/ML-IDS-PyTorch-Streamlit.zip`. Its readiness record is `output/pytorch/delivery_status.json`. It contains active source, launchers, dependency versions, tests and data instructions. Until real PCAP training is completed, it contains no production packet checkpoint.

## Historical flow implementation

The existing `dashboard/`, `src/`, `tests/`, TensorFlow checkpoints, flow metrics and `output/ML-IDS-Delivery.zip` belong to the earlier flow-feature project and remain preserved. They are outside the active raw-byte scope and are not bundled in the new packet delivery. `docs/LEGACY_FLOW_DELIVERY.md` describes the historical version; its raw-byte extension is superseded. Run it separately with `./start-flow-legacy.sh` and `.venv-macos` / `requirements-legacy.txt`. Its metrics do not apply to this packet model.
