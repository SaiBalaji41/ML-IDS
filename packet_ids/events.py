"""Packet provenance and alert policy; none of these fields are model inputs."""
import time
from packet_ids.preprocessing import extract_payload

CONTEXT_ARRAYS = {
    'source_ips': 'source_ip', 'destination_ips': 'destination_ip',
    'protocols': 'protocol', 'source_ports': 'source_port',
    'destination_ports': 'destination_port', 'captured_ats': 'captured_at',
    'capture_ids': 'capture_id',
}


def packet_context(packet):
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    from scapy.layers.inet6 import IPv6
    from scapy.layers.l2 import ARP
    ip = packet.getlayer(IP) or packet.getlayer(IPv6)
    transport = ip.payload if ip is not None else None
    while transport is not None and type(transport).__name__.startswith('IPv6ExtHdr'):
        transport = transport.payload
    protocol = ('TCP' if isinstance(transport, TCP) else 'UDP' if isinstance(transport, UDP)
                else 'ICMP' if isinstance(transport, ICMP) else 'ARP' if packet.haslayer(ARP)
                else 'IPv6' if isinstance(ip, IPv6) else 'IP' if ip is not None else 'Other')
    return {'source_ip': str(ip.src) if ip is not None else None,
            'destination_ip': str(ip.dst) if ip is not None else None,
            'protocol': protocol,
            'source_port': int(transport.sport) if isinstance(transport, (TCP, UDP)) else None,
            'destination_port': int(transport.dport) if isinstance(transport, (TCP, UDP)) else None,
            'captured_at': float(packet.time), 'wire_length': len(packet)}


def decorate_event(event):
    """Severity is an explicit triage rule, not a learned risk measurement."""
    event = dict(event)
    predicted, actual = event.get('predicted_label'), event.get('actual_label')
    if predicted is None:
        event.update(status='UNCLASSIFIED', severity='Unscored', review_required=False)
    else:
        event['status'] = 'UNVERIFIED' if actual is None else 'MATCH' if actual == predicted else 'MISMATCH'
        if event.get('review_required'):
            severity = 'Review'
        elif not event['is_attack']:
            severity = 'Info'
        elif predicted.lower().startswith(('ddos', 'mirai')):
            severity = 'Critical'
        elif predicted.lower().startswith(('recon', 'vulnerability')):
            severity = 'Medium'
        else:
            severity = 'High'
        event['severity'] = severity
    event['severity_policy'] = 'triage-v1: low-confidence Review; benign Info; DDoS/Mirai Critical; recon Medium; other attacks High'
    return event


def packet_event(packet, engine=None, sample_id=None, source='scapy_live', actual_label=None):
    payload = extract_payload(packet)
    event = {**packet_context(packet), 'timestamp': time.time(), 'sample_id': sample_id,
             'source': source, 'actual_label': actual_label, 'predicted_label': None,
             'confidence': None, 'is_attack': None, 'used_bytes': min(len(payload), 1024),
             'payload_hex': payload[:1024].hex(), 'original_payload_length': len(payload),
             'input_was_truncated': len(payload) > 1024}
    if not payload:
        event['reason'] = 'No supported nonfragmented transport payload'
    elif engine is None:
        event['reason'] = 'No trained PyTorch packet model loaded'
    else:
        try:
            event.update(engine.predict(payload))
        except Exception as exc:
            event['reason'] = f'Inference failed: {exc}'
    return decorate_event(event)

def inspect_pcap(capture, engine=None, limit=500):
    """Classify a bounded capture, preserving unknown ground truth as unknown."""
    import scapy.layers.l2  # Register Ethernet dissectors for linktype 1
    from scapy.utils import PcapReader
    if not 1 <= limit <= 10000:
        raise ValueError('Capture limit must be 1 to 10000 packets.')
    events = []
    with PcapReader(capture) as packets:
        for number, packet in enumerate(packets):
            events.append(packet_event(packet, engine, f'pcap:{number}', source='uploaded_pcap'))
            if len(events) >= limit:
                break
    return events
