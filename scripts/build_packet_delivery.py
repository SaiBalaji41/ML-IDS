"""Package only the active PyTorch/Streamlit project and its truthful readiness."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = []
    for folder in ('packet_ids', 'streamlit_app', 'packet_tests', '.streamlit'):
        files.extend(p for p in (ROOT / folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix in ('.py', '.css', '.toml'))
    files.extend(ROOT / name for name in ('README.md', 'requirements.txt', 'requirements.lock.txt',
        'start.sh', 'start.bat', 'Start-IDS.command', 'scripts/packet_pipeline.py',
        'scripts/train_packet_bytes.py', 'scripts/build_packet_delivery.py', 'docs/packet_manifest.example.csv',
        'data/packet_manifest.csv'))
    for name in ('dataset_audit.json', 'delivery_status.json', 'test_results.xml'):
        path = ROOT / 'output/pytorch' / name
        if path.is_file():
            files.append(path)
    guide = ROOT / 'output/ML-IDS-Run-Guide.docx'
    if guide.is_file():
        files.append(guide)
    model_path = ROOT / 'models/pytorch_packet_ids'
    if (model_path / 'metadata.json').is_file():
        from packet_ids.service import PacketIDS
        PacketIDS(model_path)  # Refuse test fixtures and incompatible models.
        files.extend(model_path / name for name in ('model.pt', 'metadata.json', 'evaluation.json',
                     'history.json', 'status.json', 'shap_background.npy', 'replay.npz'))
    files = sorted(set(files))
    manifest = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    directory = ROOT / 'output/pytorch'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'package_manifest.json').write_text(json.dumps(manifest, indent=2))
    target = ROOT / 'output/ML-IDS-PyTorch-Streamlit.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path('ML-IDS-PyTorch-Streamlit') / path.relative_to(ROOT))
        archive.write(directory / 'package_manifest.json', 'ML-IDS-PyTorch-Streamlit/output/pytorch/package_manifest.json')
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (target.with_suffix('.zip.sha256')).write_text(f'{digest}  {target.name}\n')
    print(json.dumps({'package': str(target), 'bytes': target.stat().st_size, 'sha256': digest,
                      'contains_trained_packet_model': (model_path / 'metadata.json').is_file()}, indent=2))


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT))
    main()
