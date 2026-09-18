# Streamlit Cloud deployment

This project keeps its Streamlit interface. Vercel is not the deployment target because this app needs a persistent Streamlit server. The hosted edition supports PCAP uploads, packet-byte inspection, and model inference, replay, SHAP and evaluation when a verified checkpoint is deployed. Training and live capture remain in the local application.

## Deployment settings

Use Streamlit Community Cloud at https://share.streamlit.io and connect the existing private repository. Keep the repository private.

| Setting | Value |
| --- | --- |
| Repository | HarshavardhanVemali/ML-IDS |
| Branch | streamlit-cloud |
| Main file | cloud/app.py |
| Python version | 3.12 |
| Dependency file | cloud/requirements.txt |
| Secrets | None required for the current app |

The deployment branch changes the server binding to `0.0.0.0` for the hosting platform. The master branch and local launcher retain loopback binding. Community Cloud checks for dependencies beside the entrypoint; the cloud dependency file installs the official Linux CPU PyTorch wheel instead of GPU packages.

After deployment, verify Overview, PCAP analysis, Packet inspector and Model evaluation. Enter `00 7f ff` in Packet inspector and inspect the payload: it should report three original bytes and 1,021 padding bytes. No capture or training controls should be present in this hosted entrypoint.

The current repository has no trained packet checkpoint. The app must say the model awaits training, show unavailable evaluation metrics and disable predictions and SHAP. Do not treat the uploaded historical flow metrics as raw-byte model results.

## Data and model updates

Uploads are processed by the hosting service and kept in the session; export results before the session ends. Cloud storage is not a persistent training workspace. Use the local app to capture traffic, verify CICIoT2023 labels, prepare data and train.

After real training, deploy all verified packet model artifacts together at `models/pytorch_packet_ids`. The weights, metadata, replay and SHAP background checksums must match. Test fixtures are rejected. Model artifacts and held-out packet bytes require a deliberate data-sharing review before making them available to hosted users.

For subsequent code updates, commit and push master, merge master into streamlit-cloud, preserve that branch's cloud server binding, then push streamlit-cloud. Community Cloud rebuilds from the selected branch.

## Troubleshooting

If the repository is not listed, the Streamlit account needs access to the private GitHub repository. That account authorization is a deployment prerequisite; no application secret or model is needed just to open the dashboard.

Select Python 3.12 in Advanced settings before deploying. If installation fails, inspect the dependency log and ensure Community Cloud selected `cloud/requirements.txt`.

On this Mac, live capture previously failed because macOS denied BPF device access. A hosted cloud server cannot capture the user's local network. Saved PCAP upload and local capture are separate workflows.
