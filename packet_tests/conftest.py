"""Small synthetic PCAP fixtures for software verification, never deployment."""
import csv
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_captures(directory, duplicate=False):
    from scapy.all import Ether, IP, UDP, Raw, wrpcap
    rows = []
    for split_index, split in enumerate(('train', 'validation', 'test')):
        for label_index, label in enumerate(('BenignTraffic', 'DDoS-UDP_Flood')):
            name = f'{split}-{label_index}.pcap'
            packets = []
            for index in range(6):
                # Unique visible bytes for each capture; classes have distinct
                # fixture patterns. This is NOT real CICIoT2023 traffic.
                payload = bytes([30 + label_index * 180]) * 64 + bytes([split_index, index, label_index])
                packets.append(Ether(src='00:11:22:33:44:55', dst='66:77:88:99:aa:bb')/IP(id=split_index * 100 + index)/UDP(sport=2000, dport=3000)/Raw(payload))
            if duplicate and split != 'train':
                payload = bytes([30 + label_index * 180]) * 64 + bytes([0, 0, label_index])
                packets.append(Ether(src='00:11:22:33:44:55', dst='66:77:88:99:aa:bb')/IP()/UDP(sport=2000, dport=3000)/Raw(payload))
            wrpcap(str(directory / name), packets)
            rows.append({'path': name, 'label': label, 'split': split,
                         'capture_id': f'{split}-{label_index}', 'dataset': 'CICIoT2023'})
    manifest = directory / 'manifest.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest, rows


@pytest.fixture(scope='session')
def trained_fixture(tmp_path_factory):
    from packet_ids.data import prepare_dataset
    from packet_ids.model import ModelConfig
    from packet_ids.training import train_dataset
    root = tmp_path_factory.mktemp('synthetic-packet-test-only')
    manifest, _ = make_captures(root)
    prepared = root / 'prepared'
    metadata = prepare_dataset(manifest, prepared)
    metadata['test_fixture'] = True
    metadata['limitations'] = 'SYNTHETIC TEST FIXTURE. Not CICIoT2023 observations or research results.'
    (prepared / 'metadata.json').write_text(json.dumps(metadata))
    model = root / 'model'
    result, evaluation = train_dataset(prepared, model, epochs=2, batch_size=4,
                                       config=ModelConfig(2, conv_channels=4, lstm_hidden=4, dropout=0))
    return model, prepared, result, evaluation
