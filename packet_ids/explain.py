"""Per-byte SHAP expected-gradient attribution of the selected model probability."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from packet_ids import BYTE_LENGTH
from packet_ids.preprocessing import encode_payload


class ClassProbability(nn.Module):
    def __init__(self, model, class_index):
        super().__init__()
        self.model, self.class_index = model, class_index

    def forward(self, inputs):
        return self.model(inputs).softmax(1)[:, self.class_index:self.class_index + 1]


def explain_payload(service, payload, samples=256, seed=42):
    import shap
    if not 32 <= samples <= 2048:
        raise ValueError('SHAP sample count must be 32 to 2048.')
    signal = encode_payload(payload)
    prediction = service.predict(payload)
    background = np.load(Path(service.directory) / 'shap_background.npy', allow_pickle=False)
    if background.dtype != np.uint8 or background.ndim != 2 or background.shape[1] != BYTE_LENGTH:
        raise ValueError('Invalid training background for byte SHAP.')
    background = background[:16].astype(np.float32) / 255
    # Padding positions contain no observed bytes; do not attribute background
    # length differences to nonexistent bytes in the inspected packet.
    background[:, signal.used_length:] = 0
    x = torch.from_numpy(signal.values.copy())[None, None, :]
    bg = torch.from_numpy(background.copy())[:, None, :]
    wrapped = ClassProbability(service.model, prediction['class_index']).eval()
    with service.lock, torch.backends.mkldnn.flags(enabled=False):
        with torch.no_grad():
            base_value = float(wrapped(bg).mean())
        explainer = shap.GradientExplainer(wrapped, bg, batch_size=32)
        values = explainer.shap_values(x, nsamples=samples, rseed=seed)
    if isinstance(values, list):
        values = values[0]
    attribution = np.asarray(values).reshape(-1)
    if attribution.size != BYTE_LENGTH or not np.isfinite(attribution).all():
        raise RuntimeError('SHAP returned an invalid byte attribution vector.')
    # Constant zero padding should have exactly zero attribution by construction.
    attribution[signal.used_length:] = 0
    residual = prediction['confidence'] - base_value - float(attribution.sum())
    strongest = np.argsort(np.abs(attribution[:signal.used_length]))[::-1][:20]
    return {'method': 'SHAP GradientExplainer / expected gradients', 'approximate': True,
            'target_label': prediction['predicted_label'], 'probability': prediction['confidence'],
            'base_probability': base_value, 'sampling_residual': residual, 'samples': samples,
            'background_rows': len(background), 'background_source': 'training partition only',
            'attributions': attribution.tolist(), 'used_bytes': signal.used_length,
            'padding_bytes': signal.padding_length, 'model_sha256': service.metadata['checkpoint_sha256'],
            'top_bytes': [{'offset': int(i), 'hex': f'{payload[i]:02x}', 'value': int(payload[i]),
                           'contribution': float(attribution[i])} for i in strongest],
            'note': 'Signed contributions explain this model probability, not causal proof of an attack. Sampling residual measures approximation error.'}
