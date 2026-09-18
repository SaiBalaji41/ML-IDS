"""Reproducible PyTorch training with capture-separated validation and held-out test."""
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from packet_ids import BYTE_LENGTH, SCHEMA_VERSION
from packet_ids.data import load_prepared
from packet_ids.model import ModelConfig, PacketCNNBiLSTM
from packet_ids.events import CONTEXT_ARRAYS


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def probabilities(model, byte_rows, batch_size=128):
    model.eval()
    result = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(byte_rows), batch_size):
            inputs = torch.from_numpy(byte_rows[start:start + batch_size].astype(np.float32) / 255.0)
            logits = model(inputs.unsqueeze(1).to(device))
            result.append(logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(result)


def train_dataset(data_dir, output_dir, epochs=25, batch_size=128, learning_rate=0.001,
                  patience=5, seed=42, device='cpu', config=None):
    if epochs < 1 or batch_size < 1 or patience < 1 or learning_rate <= 0:
        raise ValueError('Epochs, batch size, patience and learning rate must be positive.')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # A distinct run directory prevents partially overwriting an active checkpoint.
    if (output / 'model.pt').exists() or (output / 'metadata.json').exists():
        raise ValueError('Output already contains a model. Choose a new run directory to preserve it.')
    started = time.time()
    write_json(output / 'status.json', {'state': 'loading', 'started_at': started})
    data, metadata = load_prepared(data_dir)
    if metadata.get('test_fixture'):
        # Test fixtures can exercise the pipeline, but cannot become a deployment.
        artifact_kind = 'test_fixture_only'
    else:
        artifact_kind = 'trained_ciciot2023_payload_model'
    labels = metadata['labels']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)
    config = config or ModelConfig(num_classes=len(labels))
    if config.num_classes != len(labels):
        raise ValueError('Model output dimension must match the dataset label mapping.')
    model = PacketCNNBiLSTM(config).to(device)
    train = data['train']
    train_x = torch.from_numpy(train['X'].astype(np.float32) / 255).unsqueeze(1)
    train_y = torch.from_numpy(train['y'].astype(np.int64))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size,
                        shuffle=True, generator=generator, num_workers=0)
    counts = np.bincount(train['y'], minlength=len(labels))
    class_weights = np.sqrt(len(train['y']) / (len(labels) * counts))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    validation_loss = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    best_loss, best_epoch, stale = float('inf'), 0, 0
    history = []
    best_path = output / 'best_weights.pt'
    validation = data['validation']
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, correct = 0., 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(targets)
            correct += int((logits.argmax(1) == targets).sum())
        model.eval()
        val_sum, val_correct = 0., 0
        with torch.inference_mode():
            for start in range(0, len(validation['y']), batch_size):
                x = torch.from_numpy(validation['X'][start:start + batch_size].astype(np.float32) / 255).unsqueeze(1).to(device)
                y = torch.from_numpy(validation['y'][start:start + batch_size].astype(np.int64)).to(device)
                logits = model(x)
                val_sum += float(validation_loss(logits, y))
                val_correct += int((logits.argmax(1) == y).sum())
        val_loss = val_sum / len(validation['y'])
        row = {'epoch': epoch, 'loss': loss_sum / len(train_y), 'accuracy': correct / len(train_y),
               'val_loss': val_loss, 'val_accuracy': val_correct / len(validation['y']),
               'learning_rate': optimizer.param_groups[0]['lr']}
        history.append(row)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, best_path)
        else:
            stale += 1
        scheduler.step(val_loss)
        write_json(output / 'history.json', history)
        write_json(output / 'status.json', {'state': 'training', 'epoch': epoch, 'max_epochs': epochs,
                                          'best_epoch': best_epoch, 'elapsed_seconds': time.time() - started, 'latest': row})
        print(json.dumps(row), flush=True)
        if stale >= patience:
            break
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    write_json(output / 'status.json', {'state': 'evaluating', 'best_epoch': best_epoch})
    test = data['test']
    before = time.perf_counter()
    probs = probabilities(model, test['X'], batch_size)
    elapsed = time.perf_counter() - before
    predicted = probs.argmax(axis=1)
    truth = test['y']
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=np.arange(len(labels)), average='macro', zero_division=0)
    benign_index = labels.index(metadata['benign_label'])
    binary = confusion_matrix(truth != benign_index, predicted != benign_index, labels=[False, True])
    evaluation = {'sample_count': len(truth), 'accuracy': float(accuracy_score(truth, predicted)),
                  'macro_precision': float(precision), 'macro_recall': float(recall), 'macro_f1': float(f1),
                  'confusion_matrix': confusion_matrix(truth, predicted, labels=np.arange(len(labels))).tolist(),
                  'binary_confusion_matrix': binary.tolist(), 'labels': labels,
                  'false_positive_rate': float(binary[0, 1] / max(1, binary[0].sum())),
                  'false_negative_rate': float(binary[1, 0] / max(1, binary[1].sum())),
                  'classwise': classification_report(truth, predicted, labels=np.arange(len(labels)),
                                                    target_names=labels, output_dict=True, zero_division=0),
                  'inference_seconds': elapsed, 'mean_batch_latency_ms_per_packet': elapsed * 1000 / len(truth),
                  'evaluation_unit': 'unique payloads from held-out capture groups'}
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'config': config.to_dict(), 'schema_version': SCHEMA_VERSION}, output / 'model.pt')
    digest = hashlib.sha256((output / 'model.pt').read_bytes()).hexdigest()
    background_ids = np.random.default_rng(seed).choice(len(train_y), size=min(16, len(train_y)), replace=False)
    np.save(output / 'shap_background.npy', train['X'][background_ids])
    replay_ids = np.random.default_rng(seed).choice(len(truth), size=min(500, len(truth)), replace=False)
    replay_keys = ('X', 'y', 'lengths', 'sample_ids', *CONTEXT_ARRAYS)
    np.savez_compressed(output / 'replay.npz', **{k: test[k][replay_ids] for k in replay_keys if k in test})
    supporting_sha256 = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                         for name in ('shap_background.npy', 'replay.npz')}
    result = {'dataset': 'CICIoT2023', 'artifact_kind': artifact_kind, 'schema_version': SCHEMA_VERSION,
              'framework': 'PyTorch', 'torch_version': str(torch.__version__), 'representation': 'raw_transport_payload_bytes',
              'byte_length': BYTE_LENGTH, 'normalization': 'uint8 / 255', 'padding': 'right zero',
              'truncation': 'first 1024 bytes', 'labels': labels, 'benign_label': metadata['benign_label'],
              'config': config.to_dict(), 'seed': seed, 'epochs_completed': len(history), 'best_epoch': best_epoch,
              'parameter_count': sum(v.numel() for v in model.parameters()), 'sample_counts': metadata['sample_counts'],
              'class_weights': class_weights.tolist(), 'training_seconds': time.time() - started,
              'checkpoint_sha256': digest, 'prepared_dataset': str(Path(data_dir).resolve()),
              'supporting_sha256': supporting_sha256,
              'source_manifest': metadata['sources'], 'split_sha256': metadata['split_sha256'],
              'note': 'Metrics apply to supplied capture-separated payload data only. Confidence is uncalibrated.'}
    write_json(output / 'evaluation.json', evaluation)
    write_json(output / 'metadata.json', result)
    write_json(output / 'status.json', {'state': 'complete', 'epochs_completed': len(history),
                                      'best_epoch': best_epoch, 'artifact_kind': artifact_kind})
    return result, evaluation
