import io
from queue import Queue
import numpy as np
import pytest
from scapy.all import Ether, IP, TCP, UDP, Raw, wrpcap, rdpcap
from packet_ids.capture import CaptureMonitor
from packet_ids.events import packet_context, packet_event, decorate_event, inspect_pcap
from packet_ids.service import PacketIDS
from streamlit_app.monitoring import event_rows


def test_real_metadata_unknown_truth_and_model_predictions(trained_fixture):
    engine = PacketIDS(trained_fixture[0], allow_test_fixture=True)
    packet = Ether(src='00:11:22:33:44:55', dst='66:77:88:99:aa:bb')/IP(src='192.0.2.5', dst='198.51.100.9')/TCP(sport=4242, dport=443)/Raw(b'payload bytes')
    packet.time = 1234567890
    event = packet_event(packet, engine, 'test-packet')
    assert event['source_ip'] == '192.0.2.5' and event['destination_ip'] == '198.51.100.9'
    assert event['protocol'] == 'TCP'  # Port 443 does not prove HTTP or TLS.
    assert event['captured_at'] == 1234567890 and event['source_port'] == 4242
    assert event['status'] == 'UNVERIFIED' and event['actual_label'] is None
    assert event['confidence'] == engine.predict(b'payload bytes')['confidence']
    assert decorate_event({**event, 'actual_label': event['predicted_label']})['status'] == 'MATCH'
    assert decorate_event({**event, 'actual_label': 'different-class'})['status'] == 'MISMATCH'
    row = event_rows([event])[0]
    assert row['Source IP'] == '192.0.2.5' and row['True classification'] == 'Unknown'


def test_unscored_packets_and_policy_are_not_fabricated():
    no_model = packet_event(IP()/UDP()/Raw(b'abc'))
    assert no_model['confidence'] is None and no_model['status'] == 'UNCLASSIFIED'
    empty = packet_event(IP()/TCP())
    assert empty['severity'] == 'Unscored' and 'payload' in empty['reason']
    base = {'predicted_label': 'DDoS-UDP_Flood', 'actual_label': None, 'is_attack': True, 'review_required': False}
    assert decorate_event(base)['severity'] == 'Critical'
    assert decorate_event({**base, 'review_required': True})['severity'] == 'Review'


def test_capture_worker_roundtrip_uses_actual_scapy_offline_reader(tmp_path, monkeypatch, trained_fixture):
    import scapy.all
    capture = tmp_path / 'fixture.pcap'
    packets = [Ether(src='00:11:22:33:44:55', dst='66:77:88:99:aa:bb')/IP(src='192.0.2.8', dst='198.51.100.2')/UDP(sport=50, dport=60)/Raw(bytes([i+1])*32) for i in range(8)]
    wrpcap(str(capture), packets)
    actual_sniffer = scapy.all.AsyncSniffer
    def offline_sniffer(**kwargs):
        kwargs.pop('iface'); kwargs.pop('promisc')
        return actual_sniffer(offline=str(capture), **kwargs)
    monkeypatch.setattr(scapy.all, 'AsyncSniffer', offline_sniffer)
    monkeypatch.setattr(CaptureMonitor, 'interfaces', staticmethod(lambda: ['offline-fixture']))
    engine = PacketIDS(trained_fixture[0], allow_test_fixture=True)
    monitor = CaptureMonitor()
    monitor.start('offline-fixture', engine, seconds=1, save_directory=tmp_path/'saved')
    monitor.worker.join(timeout=10)
    state = monitor.snapshot()
    assert not state['running'] and state['error'] is None
    assert state['observed'] == state['processed'] == 8 and state['dropped'] == 0
    assert all(e['status'] == 'UNVERIFIED' and e['source_ip'] == '192.0.2.8' for e in state['events'])
    assert len(rdpcap(state['pcap_path'])) == 8
    batch = inspect_pcap(io.BytesIO(capture.read_bytes()), engine, limit=3)
    assert len(batch) == 3 and all(e['actual_label'] is None for e in batch)
    assert batch[0]['confidence'] == state['events'][0]['confidence']


def test_queue_overflow_is_counted_and_not_silent():
    monitor = CaptureMonitor()
    monitor.queue = Queue(maxsize=1)
    monitor.run_id = 'fixture'
    monitor._enqueue(IP()/UDP()/Raw(b'one'))
    monitor._enqueue(IP()/UDP()/Raw(b'two'))
    assert monitor.snapshot()['observed'] == 2 and monitor.snapshot()['dropped'] == 1


def test_replay_retains_capture_provenance(trained_fixture):
    engine = PacketIDS(trained_fixture[0], allow_test_fixture=True)
    events = engine.replay(3, seed=1)
    assert all(e['source_ip'] == '127.0.0.1' and e['protocol'] == 'UDP' for e in events)
    assert all(e['source_port'] == 2000 and e['capture_id'].startswith('test-') for e in events)
    assert all(e['status'] in ('MATCH', 'MISMATCH') for e in events)
