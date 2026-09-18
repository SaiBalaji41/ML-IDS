"""The single byte representation used by preparation, inference and explanation."""
from dataclasses import dataclass
import hashlib
import numpy as np
from packet_ids import BYTE_LENGTH


@dataclass(frozen=True)
class PayloadSignal:
    values: np.ndarray
    original_length: int
    used_length: int

    @property
    def padding_length(self):
        return BYTE_LENGTH - self.used_length

    @property
    def truncated(self):
        return self.original_length > BYTE_LENGTH


def encode_payload(payload: bytes) -> PayloadSignal:
    """Keep the first 1,024 payload bytes, right-pad with zero, normalize to [0, 1]."""
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError('Payload must be raw bytes, not a flow-feature vector or image.')
    if not len(payload):
        raise ValueError('This packet has no payload and cannot be classified by the payload model.')
    used = min(len(payload), BYTE_LENGTH)
    values = np.zeros(BYTE_LENGTH, dtype=np.float32)
    values[:used] = np.frombuffer(bytes(payload[:used]), dtype=np.uint8) / np.float32(255)
    return PayloadSignal(values, len(payload), used)


def padded_bytes(payload: bytes) -> np.ndarray:
    encode_payload(payload)  # Empty/type validation matches inference.
    return np.frombuffer(bytes(payload[:BYTE_LENGTH]).ljust(BYTE_LENGTH, b'\x00'), dtype=np.uint8).copy()


def parse_payload_hex(text: str) -> bytes:
    if not isinstance(text, str) or len(text) > 131072:
        raise ValueError('Provide a hexadecimal payload of at most 65,536 bytes.')
    try:
        result = bytes.fromhex(text)
    except ValueError as exc:
        raise ValueError('Payload must contain complete hexadecimal byte pairs.') from exc
    if not result:
        raise ValueError('Provide at least one payload byte.')
    return result


def extract_payload(packet) -> bytes:
    """Extract transport/application payload, excluding Ethernet/IP/TCP/UDP headers.

    Scapy may decode DNS/HTTP into structured layers; bytes(transport.payload)
    preserves those original bytes even when a Raw layer is absent. No packet
    reassembly is performed. Fragmented IP packets and payloadless packets skip.
    """
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    from scapy.layers.inet6 import IPv6, IPv6ExtHdrFragment
    from scapy.packet import Padding
    ip = packet.getlayer(IP) or packet.getlayer(IPv6)
    if ip is None or packet.haslayer(IPv6ExtHdrFragment):
        return b''
    if packet.haslayer(IP) and (int(ip.frag) or int(ip.flags) & 1):
        return b''
    transport = ip.payload
    # Walk only IPv6 extension headers; do not mistake an encapsulated packet
    # inside an ICMP error or tunnel for the outer packet's transport payload.
    while type(transport).__name__.startswith('IPv6ExtHdr'):
        transport = transport.payload
    if isinstance(transport, (TCP, UDP, ICMP)):
        payload = bytes(transport.payload)
        padding = transport.payload.getlayer(Padding)
        if padding is not None:
            payload = payload[:-len(bytes(padding))] if len(bytes(padding)) else payload
        if isinstance(transport, UDP) and transport.len is not None:
            payload = payload[:max(0, int(transport.len) - 8)]
        return payload
    return b''


def representation_hash(payload: bytes) -> str:
    """Hash the model-visible bytes, so padded/truncated duplicates cannot leak."""
    return hashlib.sha256(padded_bytes(payload).tobytes()).hexdigest()
