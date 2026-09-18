"""Shared raw-payload representation for training, PCAP and live inference."""
import numpy as np


def payload_to_array(payload: bytes, length=256):
    if not isinstance(length,int) or length < 8:raise ValueError('Byte length must be an integer >= 8.')
    if not isinstance(payload,(bytes,bytearray)):raise ValueError('Payload must be bytes.')
    out=np.zeros(length,dtype=np.float32)
    chunk=np.frombuffer(payload[:length],dtype=np.uint8)
    out[:len(chunk)]=chunk.astype(np.float32)/255.0
    return out


def parse_hex(value):
    if not isinstance(value,str) or not value.strip():raise ValueError('Provide non-empty hexadecimal payload bytes.')
    if len(value)>131072:raise ValueError('Payload is too large.')
    try:return bytes.fromhex(value)
    except ValueError:raise ValueError('Payload contains invalid hexadecimal bytes.') from None
