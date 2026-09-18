import csv
import hashlib
import json
import shutil
import numpy as np
import pytest
import torch
from packet_ids.preprocessing import encode_payload, extract_payload, parse_payload_hex, representation_hash
from packet_ids.model import ModelConfig, PacketCNNBiLSTM
from packet_ids.data import prepare_dataset, load_prepared, read_manifest, initialize_manifest
from packet_ids.service import PacketIDS
from conftest import make_captures


def test_byte_contract():
    signal = encode_payload(bytes([0, 127, 255]))
    assert signal.values.dtype == np.float32 and signal.values.shape == (1024,)
    np.testing.assert_allclose(signal.values[:3], [0, 127/255, 1])
    assert not signal.values[3:].any() and signal.padding_length == 1021
    long = bytes(range(256)) * 5
    truncated = encode_payload(long)
    assert truncated.truncated and truncated.used_length == 1024 and truncated.original_length == 1280
    np.testing.assert_allclose(truncated.values, np.frombuffer(long[:1024], np.uint8)/255)
    assert representation_hash(b'abc') == representation_hash(b'abc\x00')
    assert representation_hash(long) == representation_hash(long[:1024])


@pytest.mark.parametrize('payload', [b'', [], np.zeros(46)])
def test_no_flow_or_empty_inputs(payload):
    with pytest.raises((ValueError, TypeError)):
        encode_payload(payload)


def test_hex_validation():
    assert parse_payload_hex('00 7f ff') == b'\x00\x7f\xff'
    for invalid in ('', 'a', 'hello', 'aa' * 65537):
        with pytest.raises(ValueError):
            parse_payload_hex(invalid)


def test_capture_payload_extraction():
    from scapy.all import Ether, IP, IPv6, UDP, TCP, Raw, DNS, DNSQR, Padding
    dns = DNS(id=321, qd=DNSQR(qname='example.test'))
    packet = Ether(bytes(Ether()/IP()/UDP(sport=12000, dport=53)/dns))
    assert extract_payload(packet) == bytes(dns)  # DNS has no Raw layer.
    assert extract_payload(IP()/TCP()) == b''
    assert extract_payload(IP(flags='MF')/UDP()/Raw(b'fragment')) == b''
    assert extract_payload(IPv6()/UDP()/Raw(b'v6 payload')) == b'v6 payload'
    assert extract_payload(IP()/UDP()/Raw(b'abc')/Padding(b'\x00'*10)) == b'abc'


def test_architecture_and_gradient():
    torch.set_num_threads(2)
    model = PacketCNNBiLSTM(ModelConfig(3, conv_channels=4, lstm_hidden=4))
    inputs = torch.rand(2, 1, 1024, requires_grad=True)
    logits = model(inputs)
    assert logits.shape == (2, 3) and model.sequence.bidirectional
    assert not any(isinstance(m, torch.nn.Conv2d) for m in model.modules())
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
    assert inputs.grad.isfinite().all() and inputs.grad.abs().sum() > 0
    assert model.encoder[0].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='1024'):
        model(torch.zeros(2, 1, 46))
    with pytest.raises(ValueError):
        ModelConfig(2, byte_length=256)


def test_manifest_group_and_schema_guards(tmp_path):
    manifest, rows = make_captures(tmp_path)
    rows[2]['capture_id'] = rows[0]['capture_id']
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match='spans splits'):
        read_manifest(manifest)
    rows[2]['capture_id'] = 'validation-0'
    rows[0]['path'] = 'train.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match='Flow CSVs'):
        read_manifest(manifest)


def test_manifest_setup_and_missing_data_errors(tmp_path):
    manifest = tmp_path / 'data/packet_manifest.csv'
    with pytest.raises(FileNotFoundError, match='init-manifest'):
        read_manifest(manifest)
    initialize_manifest(manifest)
    original = manifest.read_bytes()
    with pytest.raises(ValueError, match='contains no captures'):
        read_manifest(manifest)
    with pytest.raises(FileExistsError):
        initialize_manifest(manifest)
    assert manifest.read_bytes() == original
    captures = tmp_path / 'captures'
    captures.mkdir()
    good_manifest, _ = make_captures(captures)
    rows = read_manifest(good_manifest, verify_sha256=False)
    assert rows and all(row['sha256'] is None for row in rows)
    assert all(row['sha256'] for row in read_manifest(good_manifest))


def test_prepare_cli_missing_manifest_is_actionable_without_traceback(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    cli = Path(__file__).resolve().parents[1] / 'scripts/packet_pipeline.py'
    output = tmp_path / 'processed'
    result = subprocess.run([sys.executable, str(cli), 'prepare', '--manifest',
                             str(tmp_path / 'missing.csv'), '--output', str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'Manifest not found' in result.stderr and 'init-manifest' in result.stderr
    assert 'Traceback' not in result.stderr and not output.exists()


def test_preparation_keeps_capture_splits_and_removes_duplicate_bytes(tmp_path):
    manifest, _ = make_captures(tmp_path, duplicate=True)
    prepared = tmp_path / 'prepared'
    metadata = prepare_dataset(manifest, prepared)
    data, _ = load_prepared(prepared)
    assert metadata['removed_samples']['test']['duplicate'] == 2
    assert metadata['removed_samples']['validation']['duplicate'] == 2
    hashes = {s: {hashlib.sha256(v.tobytes()).hexdigest() for v in data[s]['X']} for s in data}
    assert hashes['train'].isdisjoint(hashes['test']) and hashes['train'].isdisjoint(hashes['validation'])
    for split in data:
        assert all(str(identifier).startswith(split) for identifier in data[split]['sample_ids'])
    with pytest.raises(ValueError, match='preserve'):
        prepare_dataset(manifest, prepared)
    with (prepared / 'train.npz').open('ab') as stream:
        stream.write(b'tamper')
    with pytest.raises(ValueError, match='checksum'):
        load_prepared(prepared)


def test_real_training_reload_and_replay_on_test_only_fixture(trained_fixture):
    model_dir, _, metadata, evaluation = trained_fixture
    assert metadata['artifact_kind'] == 'test_fixture_only' and metadata['epochs_completed'] == 2
    assert evaluation['sample_count'] == 12
    assert np.asarray(evaluation['confusion_matrix']).sum() == 12
    checkpoint = torch.load(model_dir / 'model.pt', weights_only=True)
    torch.manual_seed(metadata['seed'])
    original = PacketCNNBiLSTM(ModelConfig(**metadata['config']))
    assert not torch.equal(original.encoder[0].weight, checkpoint['state_dict']['encoder.0.weight'])
    with pytest.raises(ValueError, match='test-only'):
        PacketIDS(model_dir)
    engine = PacketIDS(model_dir, allow_test_fixture=True)
    payload = bytes([30])*64 + b'\x02\x00\x00'
    prediction = engine.predict(payload)
    assert sum(prediction['probabilities'].values()) == pytest.approx(1, abs=1e-6)
    assert prediction['used_bytes'] == 67 and prediction['padding_bytes'] == 957
    replay = engine.replay(3, seed=42)
    assert len(replay) == 3 and all(e['sample_id'].startswith('test-') for e in replay)
    assert all(e['source'] == 'held_out_payload_replay' for e in replay)
    assert engine.predict(payload)['probabilities'] == prediction['probabilities']


def test_checkpoint_tampering_rejected(trained_fixture, tmp_path):
    model_dir, *_ = trained_fixture
    copy = tmp_path / 'copy'
    shutil.copytree(model_dir, copy)
    with (copy / 'model.pt').open('ab') as stream:
        stream.write(b'tamper')
    with pytest.raises(ValueError, match='checksum'):
        PacketIDS(copy, allow_test_fixture=True)


def test_actual_shap_returns_one_value_per_byte(trained_fixture):
    from packet_ids.explain import explain_payload
    model_dir, *_ = trained_fixture
    engine = PacketIDS(model_dir, allow_test_fixture=True)
    payload = bytes([30])*64 + b'\x02\x00\x00'
    result = explain_payload(engine, payload, samples=64)
    values = np.asarray(result['attributions'])
    assert values.shape == (1024,) and np.isfinite(values).all()
    assert (values[67:] == 0).all() and np.abs(values[:67]).sum() > 0
    assert abs(result['sampling_residual']) < .05
    assert all(0 <= row['offset'] < 67 and row['hex'] == f"{payload[row['offset']]:02x}" for row in result['top_bytes'])
    assert result['background_source'] == 'training partition only'


def test_no_checkpoint_no_predictions(tmp_path):
    with pytest.raises(FileNotFoundError, match='No trained PyTorch'):
        PacketIDS(tmp_path)
