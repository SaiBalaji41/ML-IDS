"""Load the strict byte checkpoint and share model inference across UI and CLI."""
import hashlib
import json
from pathlib import Path
import threading
import time
import numpy as np
import torch
from packet_ids import BYTE_LENGTH, SCHEMA_VERSION
from packet_ids.model import ModelConfig, PacketCNNBiLSTM
from packet_ids.preprocessing import encode_payload
from packet_ids.events import CONTEXT_ARRAYS, decorate_event


class PacketIDS:
    def __init__(self, model_dir, allow_test_fixture=False):
        self.directory = Path(model_dir)
        meta_path = self.directory / 'metadata.json'
        if not meta_path.is_file():
            raise FileNotFoundError('No trained PyTorch packet checkpoint. Prepare labeled CICIoT2023 PCAPs and run training first.')
        self.metadata = json.loads(meta_path.read_text())
        m = self.metadata
        if m.get('dataset') != 'CICIoT2023' or m.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Checkpoint does not match CICIoT2023 / 1,024-byte payload inputs.')
        if m.get('framework') != 'PyTorch' or m.get('byte_length') != BYTE_LENGTH:
            raise ValueError('A PyTorch 1,024-byte checkpoint is required; flow or TensorFlow models cannot be substituted.')
        if m.get('artifact_kind') != 'trained_ciciot2023_payload_model' and not allow_test_fixture:
            raise ValueError('A test-only checkpoint cannot be used as the deployed IDS.')
        path = self.directory / 'model.pt'
        if hashlib.sha256(path.read_bytes()).hexdigest() != m['checkpoint_sha256']:
            raise ValueError('Model checkpoint checksum mismatch.')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unsupported checkpoint representation.')
        if checkpoint['config'] != m['config']:
            raise ValueError('Checkpoint and metadata model configuration differ.')
        for name in ('shap_background.npy', 'replay.npz'):
            if hashlib.sha256((self.directory / name).read_bytes()).hexdigest() != m['supporting_sha256'][name]:
                raise ValueError(f'{name} checksum mismatch.')
        self.model = PacketCNNBiLSTM(ModelConfig(**checkpoint['config']))
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model.eval()
        self.labels = m['labels']
        if len(self.labels) != self.model.config.num_classes or len(set(self.labels)) != len(self.labels) or m['benign_label'] not in self.labels:
            raise ValueError('Invalid checkpoint label mapping.')
        self.lock = threading.RLock()
        torch.set_num_threads(min(4, torch.get_num_threads()))

    def predict(self, payload: bytes):
        signal = encode_payload(payload)
        started = time.perf_counter()
        with self.lock, torch.inference_mode():
            inputs = torch.from_numpy(signal.values.copy())[None, None, :]
            probabilities = self.model(inputs).softmax(1)[0].numpy()
        if not np.isfinite(probabilities).all():
            raise RuntimeError('Model produced invalid probabilities.')
        winner = int(np.argmax(probabilities))
        label = self.labels[winner]
        confidence = float(probabilities[winner])
        return {'predicted_label': label, 'class_index': winner, 'is_attack': label != self.metadata['benign_label'],
                'confidence': confidence, 'latency_ms': (time.perf_counter() - started) * 1000,
                'review_required': confidence < 0.6, 'input_length': signal.original_length,
                'used_bytes': signal.used_length, 'padding_bytes': signal.padding_length,
                'truncated': signal.truncated, 'model_sha256': self.metadata['checkpoint_sha256'],
                'probabilities': {name: float(value) for name, value in zip(self.labels, probabilities)}}

    def replay(self, count=5, seed=None):
        if not 1 <= count <= 50:
            raise ValueError('Replay count must be 1 to 50.')
        with np.load(self.directory / 'replay.npz', allow_pickle=False) as data:
            chosen = np.random.default_rng(seed).choice(len(data['y']), min(count, len(data['y'])), replace=False)
            results = []
            for index in chosen:
                actual_length = int(data['lengths'][index])
                payload = bytes(data['X'][index][:min(actual_length, BYTE_LENGTH)])
                event = self.predict(payload)
                event.update({'sample_id': str(data['sample_ids'][index]), 'source': 'held_out_payload_replay',
                              'original_payload_length': actual_length,
                              'input_was_truncated': actual_length > BYTE_LENGTH,
                              'actual_label': self.labels[int(data['y'][index])],
                              'payload_hex': payload.hex(), 'timestamp': time.time()})
                for key, field in CONTEXT_ARRAYS.items():
                    value = data[key][index].item() if key in data else None
                    event[field] = None if value in ('', -1) else value
                results.append(decorate_event(event))
            return results
