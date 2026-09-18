"""Local Streamlit UI for the required raw-byte PyTorch IDS workflow."""
import html
import io
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from packet_ids import BYTE_LENGTH
from packet_ids.preprocessing import encode_payload, extract_payload, parse_payload_hex
from packet_ids.service import PacketIDS
from packet_ids.explain import explain_payload
from streamlit_app.jobs import JobManager
from streamlit_app.monitoring import render_feed
from packet_ids.capture import CaptureMonitor
from packet_ids.events import inspect_pcap
from packet_ids.data import read_manifest, initialize_manifest

st.set_page_config(page_title='Sentinel | Packet IDS', page_icon=':material/shield:', layout='wide')
st.html('<style>' + (Path(__file__).parent / 'styles.css').read_text() + '</style>')


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


@st.cache_resource
def cached_service(directory, fingerprint):
    return PacketIDS(directory)


@st.cache_resource
def jobs():
    return JobManager(ROOT)


def selected_service():
    directory = Path(st.session_state.model_directory).expanduser()
    metadata = directory / 'metadata.json'
    fingerprint = metadata.stat().st_mtime_ns if metadata.exists() else 0
    try:
        return cached_service(str(directory), fingerprint), None
    except (FileNotFoundError, ValueError, RuntimeError, KeyError, OSError) as exc:
        return None, str(exc)


def header(eyebrow, title, subtitle):
    st.html(f'<div class="eyebrow">{html.escape(eyebrow)}</div>')
    st.title(title)
    st.caption(subtitle)


def plot_layout(fig, height=250):
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=20, b=35),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='#617590', size=11), legend=dict(orientation='h', y=1.15),
                      xaxis=dict(gridcolor='#edf1f7', zeroline=False),
                      yaxis=dict(gridcolor='#edf1f7', zeroline=False))
    return fig


def payload_chart(payload):
    signal = encode_payload(payload)
    fig = go.Figure(go.Scatter(x=np.arange(BYTE_LENGTH), y=signal.values, mode='lines',
                              line=dict(color='#4f7fda', width=1.4), name='Normalized byte',
                              hovertemplate='Byte %{x}<br>Value %{y:.4f}<extra></extra>'))
    if signal.used_length < BYTE_LENGTH:
        fig.add_vrect(x0=signal.used_length, x1=BYTE_LENGTH - 1, fillcolor='#dce3ed', opacity=.3,
                      line_width=0, annotation_text='Zero padding', annotation_position='top right')
    fig.update_xaxes(title='Byte offset (0–1023)', range=[0, BYTE_LENGTH - 1])
    fig.update_yaxes(title='Byte value / 255', range=[-.02, 1.02])
    st.plotly_chart(plot_layout(fig), width='stretch', key='payload_signal')
    st.caption(f'{signal.original_length:,} original bytes · {signal.used_length:,} retained · '
               f'{signal.padding_length:,} padded · shape (1, 1, 1024). '
               + ('Tail bytes are truncated.' if signal.truncated else 'Original byte order is preserved.'))


def set_payload(payload, description):
    st.session_state.packet_payload = payload
    st.session_state.packet_description = description
    st.session_state.pop('byte_explanation', None)
    st.session_state.pop('last_prediction', None)


def add_replay(service, count):
    events = service.replay(count)
    st.session_state.events = (st.session_state.events + events)[-500:]
    st.session_state.total_packets += len(events)
    st.session_state.attack_packets += sum(e['is_attack'] for e in events)
    st.session_state.review_packets += sum(e['review_required'] for e in events)
    st.session_state.batches = (st.session_state.batches + [{'time': time.strftime('%H:%M:%S'),
        'Benign': sum(not e['is_attack'] for e in events), 'Attack': sum(e['is_attack'] for e in events)}])[-30:]


for key, value in {'events': [], 'total_packets': 0, 'attack_packets': 0, 'review_packets': 0,
                   'streaming': False, 'batches': [], 'packet_payload': None,
                   'model_directory': str(ROOT / 'models/pytorch_packet_ids')}.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.title('sentinel')
    st.html('<div class="brand-note">PACKET INTELLIGENCE / ML-IDS</div>')
    page = st.radio('Workspace', ['Overview', 'Live capture & PCAP', 'Packet inspector', 'Byte explanations', 'Training & evaluation'],
                    label_visibility='collapsed', key='page')
    st.divider()
    st.caption('MODEL CONTRACT')
    st.markdown('**PyTorch CNN + BiLSTM**')
    st.caption('1,024 raw payload bytes · 1D signals')
    st.text_input('Model directory', key='model_directory')
    service, model_error = selected_service()
    identity = (st.session_state.model_directory, service.metadata['checkpoint_sha256'] if service else None)
    if identity != st.session_state.get('active_model_identity'):
        st.session_state.active_model_identity = identity
        for key in ('last_prediction', 'byte_explanation'):
            st.session_state.pop(key, None)
        st.session_state.pop('pcap_events', None)
        for key in ('events', 'batches'):
            st.session_state[key] = []
        for key in ('total_packets', 'attack_packets', 'review_packets'):
            st.session_state[key] = 0
        st.session_state.streaming = False
        if 'capture_monitor' in st.session_state:
            st.session_state.capture_monitor.stop()
    if service:
        st.success('Trained packet model ready')
        st.caption(f"{len(service.labels)} classes · {service.metadata['checkpoint_sha256'][:12]}")
    else:
        st.info('Packet model awaits training')
    st.html('<div class="sidebar-foot">Local Streamlit application<br>CICIoT2023 IoT captures only<br>No 2D packet-image conversion</div>')


def overview():
    header('NETWORK INTELLIGENCE', 'Packet security overview',
           'Inspect the model, replay held-out payloads, and follow evidence behind every alert.')
    st.html('<div class="hero"><div><span class="pill">PYTORCH · 1D-CNN + BiLSTM</span>'
            '<h2>Network intelligence,<br>one byte at a time.</h2>'
            '<p>Raw IoT packet payloads become 1,024-byte signals. The hybrid network classifies them, '
            'and SHAP explains the contribution of individual bytes.</p></div><div class="hero-mark">1024</div></div>')
    if not service:
        st.warning('The downloaded CICIOT23 folder contains flow CSVs, not raw packet captures. '
                   'The byte model has not been trained. No earlier flow-model metrics are shown as packet results.')
        a, b = st.columns([1.5, 1])
        with a, st.container(border=False, key="ids_card_downloads"):
            st.subheader('What is already downloaded?')
            audit = read_json(ROOT / 'output/pytorch/dataset_audit.json', {})
            st.write(f"{len(audit.get('csv_files', []))} CSV files · {len(audit.get('pcap_files', []))} PCAP files")
            st.caption('The CSVs contain 46 flow statistics plus a label. They cannot recover the original packet payloads.')
            st.code('CICIOT23/\n  train/train.csv\n  validation/validation.csv\n  test/test.csv', language='text')
        with b, st.container(border=False, key="ids_card_data_step"):
            st.subheader('Complete the data step')
            st.write('Download the raw PCAP edition, verify capture labels, and use Training & evaluation to prepare it.')
            st.link_button('Official CICIoT2023 dataset', 'https://www.unb.ca/cic/datasets/iotdataset-2023.html')
            st.caption('The official download currently requires registration. Preparation and training commands are included.')
    controls = st.columns([1, 1, 1, 2])
    with controls[0]:
        if st.button('Run 5 packets', type='primary', disabled=service is None, width='stretch'):
            try:
                add_replay(service, 5)
            except Exception as exc:
                st.error(str(exc))
    with controls[1]:
        if st.button('Pause simulation' if st.session_state.streaming else 'Start simulation',
                     disabled=service is None, width='stretch'):
            st.session_state.streaming = not st.session_state.streaming
            st.rerun()
    with controls[2]:
        if st.button('Clear session', width='stretch'):
            for key in ('events', 'batches'):
                st.session_state[key] = []
            for key in ('total_packets', 'attack_packets', 'review_packets'):
                st.session_state[key] = 0
            st.session_state.streaming = False
            st.rerun()
    with controls[3]:
        st.caption('Simulation replays real held-out payloads through the saved model. It does not generate attacks.')

    @st.fragment(run_every=2 if st.session_state.streaming and service else None)
    def feed():
        if st.session_state.streaming and service:
            try:
                add_replay(service, 5)
            except Exception as exc:
                st.session_state.streaming = False
                st.error(str(exc))
        columns = st.columns(4)
        evaluation = read_json(Path(st.session_state.model_directory) / 'evaluation.json', {}) if service else {}
        for col, label, value in zip(columns, ['Packets analyzed', 'Attack predictions', 'Needs review', 'Test accuracy'],
                                    [st.session_state.total_packets, st.session_state.attack_packets,
                                     st.session_state.review_packets,
                                     f"{evaluation['accuracy']:.2%}" if 'accuracy' in evaluation else '—']):
            col.metric(label, value)
        st.caption('Needs review means confidence below 60%. Confidence is not a calibrated security risk score.')
        left, right = st.columns([1.6, 1])
        with left, st.container(border=False, key="ids_card_traffic"):
            st.subheader('Traffic activity')
            if st.session_state.batches:
                frame = pd.DataFrame(st.session_state.batches)
                fig = go.Figure()
                for label, color in [('Benign', '#7297e4'), ('Attack', '#b19bd7')]:
                    fig.add_trace(go.Bar(x=np.arange(len(frame)), y=frame[label], name=label, marker_color=color))
                fig.update_layout(barmode='group')
                fig.update_xaxes(title='Replay batch')
                st.plotly_chart(plot_layout(fig), width='stretch')
            else:
                st.info('Run the held-out simulation after a verified packet model is available.')
        with right, st.container(border=False, key="ids_card_input"):
            st.subheader('Model input')
            st.markdown('**Raw payload → 1,024 bytes → 1D CNN → BiLSTM → prediction**')
            st.caption('Packets without an application/transport payload are unclassified. '
                       'The BiLSTM models byte-region order within a packet, not a sequence of separate packets.')
        with st.container(border=False, key="ids_card_feed"):
            st.subheader('Detection feed')
            render_feed(st.session_state.events, 'replay', set_payload)
    feed()



def monitoring_page():
    header('NETWORK MONITORING', 'Live capture and PCAP analysis',
           'Observe real packet metadata, run the loaded byte model, and inspect alerts.')
    if 'capture_monitor' not in st.session_state:
        st.session_state.capture_monitor = CaptureMonitor()
    monitor = st.session_state.capture_monitor
    mode = st.radio('Monitoring source', ['Live interface', 'PCAP file'], horizontal=True)
    if not service:
        st.info('Observation mode: packet addresses and protocols are available. Predictions and confidence need a trained packet model.')
    if mode == 'Live interface':
        with st.container(border=False, key='ids_card_live_controls'):
            try:
                interfaces = monitor.interfaces()
            except Exception as exc:
                st.error(f'Could not list capture interfaces: {exc}')
                interfaces = []
            interface = st.selectbox('Network interface', interfaces, index=None,
                                     placeholder='Select your IoT or test-network interface')
            a, b = st.columns(2)
            seconds = a.number_input('Capture duration in seconds', 1, 120, 30)
            limit = b.number_input('Maximum packets', 1, 10000, 1000)
            save = st.checkbox('Save captured packets locally as PCAP', value=False)
            status = monitor.snapshot()
            a, b = st.columns(2)
            if a.button('Start capture', type='primary', disabled=interface is None or status['running']):
                try:
                    monitor.start(interface, service, int(seconds), int(limit), ROOT / 'runtime/captures' if save else None)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            if b.button('Stop capture', disabled=not status['running']):
                monitor.stop()
                st.rerun()
            st.caption('Passive capture on the selected interface; maximum 120 seconds per run. '
                       'A bounded queue separates capture from inference. Ground truth is unknown for live traffic.')
        @st.fragment(run_every=1)
        def captured_feed():
            state = monitor.snapshot()
            cols = st.columns(4)
            for col, label, value in zip(cols, ['Packets observed', 'Packets processed', 'Queue drops', 'Queued'],
                                         [state['observed'], state['processed'], state['dropped'], state['queued']]):
                col.metric(label, value)
            st.caption('Capture running' if state['running'] else 'Capture stopped')
            if state['error']:
                st.error(state['error'])
                st.caption('If macOS denies BPF access, use an existing authorized PCAP and the PCAP file tab. '
                           'Run only the capture CLI with required OS privileges; do not run Streamlit as root.')
            if state['pcap_path'] and not state['running'] and state['observed']:
                st.caption(f"Saved capture: {state['pcap_path']}")
            render_feed(state['events'], 'live', set_payload)
        captured_feed()
    else:
        with st.container(border=False, key='ids_card_pcap_batch'):
            upload = st.file_uploader('Capture to analyze', type=['pcap', 'pcapng', 'cap', 'pcp'])
            limit = st.number_input('Packets to analyze', 1, 5000, 500)
            if st.button('Analyze capture', type='primary', disabled=upload is None):
                try:
                    with st.spinner('Parsing capture and processing payloads…'):
                        st.session_state.pcap_events = inspect_pcap(io.BytesIO(upload.getvalue()), service, int(limit))
                except Exception as exc:
                    st.error(f'Capture analysis failed: {exc}')
            st.caption('Up to 32 MB per upload. Labels are not inferred from filenames; uploaded captures show unknown ground truth.')
        render_feed(st.session_state.get('pcap_events', []), 'pcap', set_payload)


def inspector():
    header('RAW PACKET ANALYSIS', 'Packet inspector', 'Examine raw payloads and the exact 1,024-byte input sent to the model.')
    with st.container(border=False, key="ids_card_inspector"):
        mode = st.radio('Input source', ['Hex payload', 'PCAP file', 'Held-out sample'], horizontal=True)
        if mode == 'Hex payload':
            with st.form('payload_hex_form', border=False):
                raw = st.text_area('Payload bytes in hexadecimal', placeholder='00 7f ff …', height=130, max_chars=131072)
                if st.form_submit_button('Inspect payload', type='primary'):
                    try:
                        set_payload(parse_payload_hex(raw), 'User-provided payload; no ground-truth label')
                    except ValueError as exc:
                        st.error(str(exc))
        elif mode == 'PCAP file':
            upload = st.file_uploader('PCAP or PCAPNG capture (up to 32 MB)', type=['pcap', 'pcapng', 'pcp', 'cap'])
            if upload:
                try:
                    from scapy.utils import PcapReader
                    records = []
                    with PcapReader(io.BytesIO(upload.getvalue())) as packets:
                        for number, packet in enumerate(packets):
                            payload = extract_payload(packet)
                            if payload:
                                records.append((number, payload))
                            if len(records) >= 200:
                                break
                    if records:
                        choice = st.selectbox('Packet payload', range(len(records)),
                                              format_func=lambda i: f'Packet {records[i][0]} · {len(records[i][1]):,} bytes')
                        if st.button('Inspect selected packet', type='primary'):
                            set_payload(records[choice][1], f'{upload.name} · packet {records[choice][0]}')
                    else:
                        st.warning('No supported, nonfragmented TCP/UDP/ICMP payloads were found.')
                except Exception as exc:
                    st.error(f'Capture could not be parsed: {exc}')
        else:
            if not service:
                st.info('Held-out packet samples become available after packet-model training.')
            elif st.button('Load a held-out payload', type='primary'):
                event = service.replay(1)[0]
                set_payload(bytes.fromhex(event['payload_hex']), f"{event['sample_id']} · actual label: {event['actual_label']}")
    payload = st.session_state.packet_payload
    if payload is not None:
        st.caption(st.session_state.packet_description)
        with st.container(border=False, key="ids_card_signal"):
            st.subheader('One-dimensional byte signal')
            payload_chart(payload)
        if st.button('Classify packet', type='primary', disabled=service is None):
            try:
                st.session_state.last_prediction = service.predict(payload)
            except Exception as exc:
                st.error(str(exc))
        if 'last_prediction' in st.session_state:
            prediction = st.session_state.last_prediction
            columns = st.columns(3)
            columns[0].metric('Prediction', prediction['predicted_label'])
            columns[1].metric('Confidence', f"{prediction['confidence']:.2%}")
            columns[2].metric('Inference', f"{prediction['latency_ms']:.1f} ms")
            st.dataframe(pd.DataFrame(prediction['probabilities'].items(), columns=['Class', 'Probability']),
                         width='stretch', hide_index=True)
            st.caption('Open Byte explanations to inspect SHAP contributions for this same payload.')
        if not service:
            st.info('Preprocessing is available now. Classification requires the trained PyTorch byte checkpoint.')


def byte_explanations():
    header('EXPLAINABLE PACKET INTELLIGENCE', 'Byte explanations',
           'Per-byte SHAP contributions to this model’s predicted class probability.')
    if not service:
        st.info('Train the packet model on labeled CICIoT2023 PCAPs before generating byte explanations.')
        st.caption(model_error)
    payload = st.session_state.packet_payload
    if payload is None:
        st.info('Select a payload in Packet inspector first.')
        return
    st.caption(st.session_state.packet_description)
    samples = st.select_slider('Expected-gradient samples', options=[64, 128, 256, 512, 1024], value=256)
    if st.button('Explain packet with SHAP', type='primary', disabled=service is None):
        try:
            with st.spinner('Computing local byte contributions…'):
                st.session_state.byte_explanation = explain_payload(service, payload, samples=samples)
        except Exception as exc:
            st.error(str(exc))
    result = st.session_state.get('byte_explanation')
    if not result:
        return
    st.subheader(f"Predicted class: {result['target_label']}")
    metrics = st.columns(3)
    metrics[0].metric('Background probability', f"{result['base_probability']:.2%}")
    metrics[1].metric('Packet probability', f"{result['probability']:.2%}")
    metrics[2].metric('Sampling residual', f"{result['sampling_residual']:+.4f}")
    values = np.asarray(result['attributions'])
    used = result['used_bytes']
    with st.container(border=False, key="ids_card_shap"):
        st.subheader('Contribution at each byte offset')
        fig = go.Figure(go.Bar(x=np.arange(used), y=values[:used],
                              marker_color=np.where(values[:used] >= 0, '#5e89df', '#af8ac9'),
                              customdata=[f'{b:02x}' for b in payload[:used]],
                              hovertemplate='Byte %{x} · 0x%{customdata}<br>SHAP %{y:+.5f}<extra></extra>'))
        fig.update_xaxes(title='Original payload byte offset')
        fig.update_yaxes(title='Probability contribution')
        st.plotly_chart(plot_layout(fig, 300), width='stretch')
        st.caption('Blue increases the predicted class probability; purple decreases it. Zero padding is excluded from observed-byte attribution.')
        start = st.slider('Byte strip start offset', 0, max(1, used - 1), 0)
        end = min(used, start + 128)
        scale = max(float(np.abs(values[:used]).max()), 1e-12)
        cells = []
        for offset in range(start, end):
            value = values[offset]
            color = '73,119,212' if value >= 0 else '161,111,186'
            alpha = .06 + .55 * abs(value) / scale
            cells.append(f'<span title="Byte {offset}; contribution {value:+.6f}" '
                         f'style="background:rgba({color},{alpha:.3f})">{payload[offset]:02x}</span>')
        st.html('<div class="byte-strip">' + ''.join(cells) + '</div>')
        st.caption(f'One-dimensional hex strip, offsets {start}–{max(start, end - 1)}. Hover to see the byte index and signed contribution.')
    st.dataframe(pd.DataFrame(result['top_bytes']), width='stretch', hide_index=True)
    st.caption(f"{result['method']} · {result['background_rows']} training background packets · {result['samples']} samples. {result['note']}")
    if abs(result['sampling_residual']) > .05:
        st.warning('Approximation error exceeds 0.05 probability units. Increase SHAP samples before interpreting individual contributions.')
    st.download_button('Export byte explanation', json.dumps(result, indent=2), 'byte-shap-explanation.json', 'application/json')


def training_page():
    header('REPRODUCIBLE TRAINING', 'Training & evaluation',
           'Prepare labeled CICIoT2023 captures, train the PyTorch hybrid, and evaluate unseen capture groups.')
    template = ('path,label,split,capture_id,dataset,source_url\n'
                'captures/benign_train.pcap,BenignTraffic,train,benign-session-1,CICIoT2023,\n'
                'captures/attack_train.pcap,DDoS-UDP_Flood,train,attack-session-1,CICIoT2023,\n'
                'captures/benign_val.pcap,BenignTraffic,validation,benign-session-2,CICIoT2023,\n'
                'captures/attack_val.pcap,DDoS-UDP_Flood,validation,attack-session-2,CICIoT2023,\n'
                'captures/benign_test.pcap,BenignTraffic,test,benign-session-3,CICIoT2023,\n'
                'captures/attack_test.pcap,DDoS-UDP_Flood,test,attack-session-3,CICIoT2023,\n')
    left, right = st.columns(2)
    job_state = jobs().status()
    with left, st.container(border=False, key="ids_card_prepare"):
        st.subheader('1. Prepare packet data')
        st.caption('Each capture must have a verified homogeneous label. All files from the same capture session use the same capture_id and split.')
        st.download_button('Download manifest template', template, 'packet_manifest.csv', 'text/csv')
        manifest = st.text_input('Manifest path', value=str(ROOT / 'data/packet_manifest.csv'))
        manifest_ready = False
        try:
            capture_rows = read_manifest(manifest, verify_sha256=False)
            manifest_ready = True
            st.caption(f'{len(capture_rows)} capture paths checked. Preparation verifies file checksums and usable payloads.')
        except (OSError, ValueError) as exc:
            st.info(str(exc))
        if not Path(manifest).expanduser().exists():
            if st.button('Create blank manifest', disabled=job_state['running']):
                try:
                    initialize_manifest(manifest)
                    st.rerun()
                except OSError as exc:
                    st.error(f'Manifest could not be created: {exc}')
        data_dir = st.text_input('Prepared data directory', value=str(ROOT / 'data/packet_processed'))
        benign_label = st.text_input('Benign label', value='BenignTraffic')
        sample_limit = st.number_input('Maximum unique candidates per capture', min_value=100, max_value=100000, value=10000, step=100)
        if st.button('Prepare dataset', disabled=not manifest_ready or job_state['running']):
            try:
                jobs().start('Preparing PCAP payloads', ['prepare', '--manifest', manifest, '--output', data_dir,
                                                       '--max-per-capture', str(sample_limit), '--benign-label', benign_label])
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with right, st.container(border=False, key="ids_card_train"):
        st.subheader('2. Train PyTorch CNN–BiLSTM')
        epochs = st.number_input('Maximum epochs', min_value=1, max_value=200, value=25)
        batch = st.selectbox('Batch size', [32, 64, 128, 256], index=2)
        output = st.text_input('New model output directory', value=st.session_state.model_directory)
        ready = (Path(data_dir) / 'metadata.json').exists()
        if st.button('Start training', type='primary', disabled=not ready or job_state['running']):
            try:
                jobs().start('Training PyTorch model', ['train', '--data', data_dir, '--output', output,
                                                      '--epochs', str(epochs), '--batch-size', str(batch)])
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        st.caption('Training fits only on the training split; validation selects the checkpoint. Test evaluation runs once after selection. Existing models are never overwritten.')
        if not ready:
            st.info('Prepare raw packet data first. Flow CSV partitions do not satisfy this input contract.')

    @st.fragment(run_every=3)
    def progress():
        state = jobs().status()
        if state['name']:
            if state['running']:
                st.info(f"{state['name']} is running in the background.")
            elif state['exit_code'] == 0:
                st.success(f"{state['name']} completed. Select the output model directory in the sidebar to load it.")
            else:
                st.error(f"{state['name']} failed. Details are in the job log.")
            with st.expander('Job log', expanded=state['exit_code'] not in (None, 0)):
                st.code(state['log_tail'], language='text')
        history = read_json(Path(output) / 'history.json', [])
        if history:
            frame = pd.DataFrame(history)
            st.subheader('Learning curves')
            plots = st.columns(2)
            for column, keys, title in [(plots[0], ['accuracy', 'val_accuracy'], 'Accuracy'), (plots[1], ['loss', 'val_loss'], 'Loss')]:
                fig = go.Figure()
                for key, color in zip(keys, ['#5e89df', '#af8ac9']):
                    fig.add_trace(go.Scatter(x=frame['epoch'], y=frame[key], mode='lines', name=key, line_color=color))
                fig.update_xaxes(title='Epoch'); fig.update_yaxes(title=title)
                column.plotly_chart(plot_layout(fig), width='stretch')
    progress()
    if service:
        evaluation = read_json(service.directory / 'evaluation.json', {})
        if evaluation:
            st.subheader('Held-out payload evaluation')
            cols = st.columns(4)
            for col, key, title in zip(cols, ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1'],
                                       ['Accuracy', 'Macro precision', 'Macro recall', 'Macro F1']):
                col.metric(title, f"{evaluation[key]:.2%}")
            st.caption(f"{evaluation['sample_count']:,} unique held-out payloads. False-positive rate {evaluation['false_positive_rate']:.2%}; false-negative rate {evaluation['false_negative_rate']:.2%}.")
            cm = np.asarray(evaluation['confusion_matrix'])
            fig = go.Figure(go.Heatmap(z=cm, x=evaluation['labels'], y=evaluation['labels'], colorscale='Blues',
                                      hovertemplate='Actual %{y}<br>Predicted %{x}<br>Packets %{z}<extra></extra>'))
            fig.update_xaxes(title='Predicted class'); fig.update_yaxes(title='Actual class')
            st.plotly_chart(plot_layout(fig, 380), width='stretch')
            st.caption('This is an evaluation confusion matrix, not a conversion of packet inputs into images.')
            frame = pd.DataFrame(evaluation['classwise']).T
            st.dataframe(frame, width='stretch')
            st.download_button('Export evaluation JSON', json.dumps(evaluation, indent=2), 'packet-evaluation.json', 'application/json')

    st.subheader('Nested validation and bootstrap intervals')
    st.caption('Development captures are divided into outer evaluation folds and inner tuning folds. '
               'The deployment test partition stays reserved. Each bootstrap draw resamples whole captures.')
    with st.container(border=False, key='ids_card_nested'):
        cv_output = st.text_input('Validation output directory', value=str(ROOT / 'output/pytorch/nested_validation'))
        a, b, c = st.columns(3)
        outer = a.number_input('Outer folds', 2, 10, 5)
        inner = b.number_input('Inner folds', 2, 5, 3)
        bootstrap = c.number_input('Bootstrap iterations', 20, 5000, 100)
        cv_epochs = st.number_input('Epochs per validation fit', 1, 200, 5)
        st.caption(f'{outer * (inner * 2 + 1)} model fits across two predefined architectures. '
                   'Each fold needs every class and independent capture groups. Fixed epochs avoid tuning on an outer evaluation fold.')
        if st.button('Run nested validation', disabled=not ready or job_state['running']):
            try:
                jobs().start('Nested capture validation', ['validate', '--data', data_dir, '--output', cv_output,
                    '--outer-folds', str(outer), '--inner-folds', str(inner), '--bootstrap', str(bootstrap),
                    '--epochs', str(cv_epochs), '--batch-size', str(batch)])
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        @st.fragment(run_every=3)
        def validation_progress():
            state = read_json(Path(cv_output) / 'status.json', {})
            if state:
                st.caption(f"Validation state: {state.get('state')} · {state.get('completed_fits', 0)}/{state.get('total_fits', '?')} fits")
                if state.get('error'):
                    st.error(state['error'])
            result = read_json(Path(cv_output) / 'cross_validation.json', {})
            if result and result.get('artifact_kind') == 'ciciot2023_nested_validation':
                st.dataframe(pd.DataFrame([{'Fold': f['fold'], **f['metrics']} for f in result['folds']]), hide_index=True, width='stretch')
                st.dataframe(pd.DataFrame(result['bootstrap']['metrics']).T.rename_axis('Metric'), width='stretch')
                st.caption('Bootstrap columns show mean, standard deviation and 95% percentile bounds. ' + result['bootstrap']['note'])
                st.download_button('Export nested validation', json.dumps(result, indent=2), 'nested-validation.json', 'application/json')
            elif result:
                st.info('This validation artifact is a software test fixture, not a CICIoT2023 result.')
        validation_progress()
    with st.expander('Architecture and scope'):
        st.code('1024 uint8 bytes\n  -> right-pad / truncate / divide by 255\n  -> tensor [batch, 1, 1024]\n  -> Conv1D + ReLU + MaxPool\n  -> Conv1D + ReLU + MaxPool\n  -> ordered 64-region feature sequence\n  -> bidirectional LSTM\n  -> dense classifier\n  -> class probability + byte SHAP', language='text')
        st.caption('Only CICIoT2023 raw IoT payloads are accepted for preparation. NSL-KDD and other legacy tabular datasets are outside the active project. Payloadless traffic is not scored.')

{'Overview': overview, 'Live capture & PCAP': monitoring_page, 'Packet inspector': inspector, 'Byte explanations': byte_explanations,
 'Training & evaluation': training_page}[page]()
