# Delivery report — Sentinel ML-IDS

Completed on 18 September 2026. Open http://127.0.0.1:8765 or run `./start.sh`.

## Delivered and verified

- Blue professional light dashboard with dark text, cool neutral surfaces, restrained purple/amber accents, 30 px buttons, 16 px card radii, fine borders, consistent spacing and no shadows.
- A newly trained CNN-BiLSTM checkpoint, with the same preprocessing used by the dashboard and streaming CLI.
- Model inference on actual held-out flows; replay controls, flow JSON analysis, CSV batch analysis and downloadable predictions.
- Local SHAP explanations from the actual selected model, with a measured additivity residual.
- Training history, model comparison, binary confusion matrix and 34-class evaluation.
- Scapy interface discovery and bounded capture implementation. Raw-packet metadata is explicitly separate from flow predictions.
- Strict schema validation, request size bounds, same-origin POST handling and static path restrictions.
- 86 passing tests. End-to-end local API verification passes and is saved in `verification.json`.
- Standalone source-and-artifact ZIP with checksums and setup instructions.

## Training evidence

The checkpoint was trained from scratch for 25 epochs; validation selected epoch 23. Training used 200,000 stratified flows from the original 5,491,971 training partition and 40,000 validation flows. The network has 57,010 parameters. Input is the training-scaled 46-feature vector, compressed by signed log1p and reshaped to (46, 1). It is not a raw-byte model.

The entire held-out test partition of 1,176,851 flows was evaluated after checkpoint selection.

| Metric | Result |
|---|---:|
| Multiclass accuracy | 86.6789% |
| Macro precision | 59.1338% |
| Macro recall | 59.7732% |
| Macro F1 | 58.3385% |
| Binary attack precision | 99.7118% |
| Binary attack recall | 99.4934% |
| Binary attack F1 | 99.6025% |

| Actual / predicted | Benign | Attack |
|---|---:|---:|
| Benign | 24,405 | 3,304 |
| Attack | 5,822 | 1,143,320 |

False-positive rate on benign flows is 11.92%. This is a material limitation even though attack recall is high. The original split's capture/session independence has not been audited. Probabilities are not calibrated risk scores. Existing research comparisons are preserved and labeled separately.

Checkpoint SHA-256: `d166160bc093986f75e68e105d9cd861550ddb1d400bc93defc09bc2af5686e7`.

## Dataset reference checked

The user's `/Users/harsha/Downloads/CICIOT23` directory contains only train, validation and test CSVs. Each file's SHA-256 matches the corresponding project raw file. All 47 columns are the 46 flow features plus `label`. `dataset_source_audit.json` records the check.

The attached PDF recommends feature parity and an offline/online pipeline and describes optional production services. Its recommendations were treated as reference content. The user's CNN-BiLSTM and light UI requirements determine the delivered implementation.

## Remaining external prerequisites

1. **Raw-byte training:** needs verified labels associated with PCAP payloads. No such data is in the supplied directory. The byte preprocessor, manifest validator, trainer, PCAP inference and live inference are implemented; no trained byte model or live attack-detection accuracy is invented. Live capture without that model shows metadata only.
2. **Live capture permissions:** need access to the selected OS capture interface. The local interface list and capture error handling were checked; an authorized live capture with a matching byte model has not been demonstrated.
3. **Kafka broker:** the adapter and schema routing are implemented and tested. Docker was not running, so a real broker round-trip is unverified. The optional `compose.kafka.yaml` config and instructions are provided. Kafka output currently goes to JSONL, while the dashboard's detection feed is held-out replay.
4. **Production deployment:** authentication, TLS, alert delivery services, an exact CICIoT2023 live flow extractor and intrusion prevention are outside this local deliverable. Nothing in this package claims to block attacks or provide production protection.

## Handoff

The ZIP contains the 21 inference/preprocessing/capture regression tests; all 86 full-project tests passed in the original workspace. The extracted ZIP passed all 21 included regression tests. The live browser replay button also produced real predictions. It includes the full held-out test partition, deployed and historical deep-learning checkpoints, the XGBoost model, preprocessing assets, source, tests, and evaluation evidence. It excludes virtual environments, local traffic logs, full training/raw datasets and the large historical random forest binary. Use the original project for full training or attach the original train/validation files to the ZIP checkout. Dependencies are pinned in `requirements.txt` and the verified environment inventory is in `environment.json`.

Original research files and checkpoints remain intact. The current README supersedes the historical project's all-phases-complete claims.
