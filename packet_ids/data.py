"""Capture-group splitting and preparation of verified CICIoT2023 payload data."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from packet_ids import BYTE_LENGTH, SCHEMA_VERSION
from packet_ids.preprocessing import extract_payload, padded_bytes, representation_hash
from packet_ids.events import packet_context, CONTEXT_ARRAYS

SPLITS = ('train', 'validation', 'test')
REQUIRED_COLUMNS = {'path', 'label', 'split', 'capture_id', 'dataset'}
MANIFEST_HEADER = 'path,label,split,capture_id,dataset,source_url\n'


def initialize_manifest(path):
    """Create an empty, editable manifest without inventing capture labels."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(MANIFEST_HEADER)
    return path


def read_manifest(path, *, verify_sha256=True):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f'Manifest not found: {path}. Create it with the dashboard "Create blank manifest" '
            'button or the init-manifest command, then add verified labeled CICIoT2023 PCAP paths. '
            'See docs/packet_manifest.example.csv. Flow CSVs cannot supply packet payloads.')
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED_COLUMNS <= set(reader.fieldnames or []):
            raise ValueError('Manifest columns required: path,label,split,capture_id,dataset. Optional: source_url.')
        rows = [row for row in reader if any(str(value or '').strip() for value in row.values())]
    if not rows:
        raise ValueError('Manifest contains no captures. Add verified labeled CICIoT2023 PCAP paths, '
                         'labels, capture IDs and train/validation/test splits. '
                         'The downloaded flow CSVs cannot replace raw packet captures.')
    groups, seen_files = {}, set()
    resolved = []
    for row in rows:
        if (row.get('dataset') or '').strip() != 'CICIoT2023':
            raise ValueError('Only verified CICIoT2023 packet captures are in scope.')
        split, label, group = ((row.get(k) or '').strip() for k in ('split', 'label', 'capture_id'))
        if split not in SPLITS or not label or not group:
            raise ValueError('Each capture requires a label, capture_id and train/validation/test split.')
        if group in groups and groups[group] != split:
            raise ValueError(f'Capture group {group!r} spans splits. Split entire capture sessions, not packets.')
        groups[group] = split
        capture = (path.parent / Path((row.get('path') or '').strip()).expanduser()).resolve()
        if capture.suffix.lower() not in ('.pcap', '.pcapng', '.cap', '.pcp'):
            raise ValueError('Expected PCAP/PCAPNG files. Flow CSVs cannot be converted into raw payloads.')
        if not capture.is_file():
            raise FileNotFoundError(f'Capture not found: {capture}')
        if verify_sha256:
            with capture.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        else:
            digest = None  # UI preflight must not re-hash large captures on every rerun.
        identity = digest if verify_sha256 else str(capture)
        if identity in seen_files:
            raise ValueError('Duplicate capture file detected in manifest.')
        seen_files.add(identity)
        resolved.append({**row, 'path': str(capture), 'label': label, 'split': split,
                         'capture_id': group, 'sha256': digest})
    return resolved


def prepare_dataset(manifest, output_dir, max_packets_per_capture=10000, seed=42,
                    benign_label='BenignTraffic'):
    """Reservoir-sample payloads within already-disjoint capture groups.

    Labels are supplied by verified capture provenance. A mixed-label capture
    must be labeled/separated first. Model-identical payloads in multiple splits
    are removed from later splits. Contradictory labels within a split are
    """
    import scapy.layers.l2  # Register Ethernet dissectors for linktype 1
    from scapy.utils import PcapReader
    if max_packets_per_capture < 1:
        raise ValueError('Packet sample limit must be positive.')
    output = Path(output_dir)
    if any((output / name).exists() for name in ('metadata.json', 'train.npz', 'validation.npz', 'test.npz')):
        raise ValueError('Output already contains prepared data. Choose a new directory to preserve it.')
    rows = read_manifest(manifest)
    rng = np.random.default_rng(seed)
    samples = {split: [] for split in SPLITS}
    sources = []
    for source in rows:
        reservoir, eligible, skipped = [], 0, 0
        with PcapReader(source['path']) as packets:
            for index, packet in enumerate(packets):
                payload = extract_payload(packet)
                if not payload:
                    skipped += 1
                    continue
                eligible += 1
                item = (padded_bytes(payload), len(payload), source['label'],
                        f"{source['capture_id']}:{index}",
                        {**packet_context(packet), 'capture_id': source['capture_id']}, representation_hash(payload))
                if len(reservoir) < max_packets_per_capture:
                    reservoir.append(item)
                else:
                    slot = int(rng.integers(eligible))
                    if slot < max_packets_per_capture:
                        reservoir[slot] = item
        samples[source['split']].extend(reservoir)
        sources.append({**source, 'eligible_payloads': eligible, 'skipped_payloadless_or_fragmented': skipped,
                        'sampled_payloads': len(reservoir)})
    # Earlier splits never depend on later labels. Reserve even ambiguous
    # hashes so model-visible bytes cannot reappear in an evaluation split.
    accepted, seen, removed = {}, set(), {}
    for split in SPLITS:
        hash_labels = {}
        for item in samples[split]:
            hash_labels.setdefault(item[-1], set()).add(item[2])
        conflicting = {h for h, labels in hash_labels.items() if len(labels) > 1}
        accepted[split] = []
        removed[split] = {'conflicting': 0, 'duplicate': 0}
        for item in samples[split]:
            digest = item[-1]
            if digest in conflicting:
                removed[split]['conflicting'] += 1
            elif digest in seen:
                removed[split]['duplicate'] += 1
            else:
                accepted[split].append(item)
                seen.add(digest)
        seen.update(hash_labels)
    labels = sorted({row['label'] for row in rows})
    if len(labels) < 2 or benign_label not in labels:
        raise ValueError('At least one benign and one attack class are required.')
    for split, items in accepted.items():
        missing = set(labels) - {item[2] for item in items}
        if missing:
            raise ValueError(f'{split} lacks usable, nonduplicate payloads for {sorted(missing)}. '
                             'Supply additional independent captures; payloadless attacks cannot train this model.')
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for split, items in accepted.items():
        path = output / f'{split}.npz'
        np.savez_compressed(path, X=np.stack([v[0] for v in items]),
                            y=np.asarray([labels.index(v[2]) for v in items], dtype=np.int64),
                            lengths=np.asarray([v[1] for v in items], dtype=np.int32),
                            sample_ids=np.asarray([v[3] for v in items]),
                            **{key: np.asarray([v[4][field] if v[4][field] is not None else
                                               (-1 if key.endswith('ports') else '') for v in items])
                               for key, field in CONTEXT_ARRAYS.items()})
        with path.open('rb') as stream:
            artifacts[split] = hashlib.file_digest(stream, 'sha256').hexdigest()
    metadata = {'dataset': 'CICIoT2023', 'schema_version': SCHEMA_VERSION,
                'representation': 'raw_transport_payload_bytes', 'byte_length': BYTE_LENGTH,
                'normalization': 'uint8 / 255', 'padding': 'right zero', 'truncation': 'first 1024 bytes',
                'labels': labels, 'benign_label': benign_label, 'seed': seed,
                'split_unit': 'capture_id', 'split_sha256': artifacts,
                'sample_counts': {s: len(v) for s, v in accepted.items()},
                'class_counts': {s: {label: sum(v[2] == label for v in items) for label in labels}
                                 for s, items in accepted.items()},
                'removed_samples': removed, 'sources': sources,
                'label_policy': 'Verified homogeneous capture labels supplied in manifest; no labels inferred from payload content.',
                'limitations': 'Capture provenance and homogeneous labels must be verified by the data owner. No payloads are reconstructed from flow CSVs.'}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def load_prepared(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    if metadata.get('schema_version') != SCHEMA_VERSION or metadata.get('dataset') != 'CICIoT2023':
        raise ValueError('Dataset is not the required CICIoT2023 1,024-byte representation.')
    labels = metadata['labels']
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError('Invalid class labels.')
    splits = {}
    for split in SPLITS:
        path = directory / f'{split}.npz'
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != metadata['split_sha256'][split]:
            raise ValueError(f'{split} data differs from the prepared dataset checksum.')
        with np.load(path, allow_pickle=False) as data:
            values = {key: data[key] for key in data.files}
        x, y = values['X'], values['y']
        if x.dtype != np.uint8 or x.ndim != 2 or x.shape[1] != BYTE_LENGTH:
            raise ValueError('Prepared inputs must be uint8 byte sequences (N, 1024).')
        if y.shape != (len(x),) or not len(x) or not np.isin(y, np.arange(len(labels))).all():
            raise ValueError('Invalid labels in prepared dataset.')
        if not np.isin(np.arange(len(labels)), y).all():
            raise ValueError(f'{split} must contain every required class.')
        lengths, ids = values['lengths'], values['sample_ids']
        if lengths.shape != (len(x),) or not np.issubdtype(lengths.dtype, np.integer) or (lengths <= 0).any():
            raise ValueError('Invalid original payload lengths.')
        if ids.shape != (len(x),) or len(set(ids.tolist())) != len(x):
            raise ValueError('Invalid or duplicate sample identifiers.')
        splits[split] = values
    return splits, metadata
