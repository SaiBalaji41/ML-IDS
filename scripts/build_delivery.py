"""Package runnable inference source, checkpoints and held-out data for handoff."""
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]


def main():
    target=ROOT/'output/ML-IDS-Delivery.zip'
    files=[]
    for folder in ['dashboard','src','scripts','configs','docs']:
        files += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.html','.css','.js','.md','.yaml']]
    for name in ['README.md','requirements.txt','requirements.lock.txt','start.sh','start.bat','Start-IDS.command','compose.kafka.yaml','.env.example','.gitignore',
                 'tests/conftest.py','tests/__init__.py','tests/test_delivery.py','tests/test_preprocessing.py','tests/test_packet_sniffer.py']:
        if (ROOT/name).exists():files.append(ROOT/name)
    for folder in ['models/deployed','models/cnn_bilstm','models/cnn_1d','models/bilstm','models/preprocessing','output/delivery']:
        files += [p for p in (ROOT/folder).rglob('*') if p.is_file() and p.name not in ['package_manifest.json']]
    files += [ROOT/'models/xgboost.pkl',ROOT/'models/xgboost_metadata.json',ROOT/'models/random_forest_metadata.json']
    files += list((ROOT/'results/metrics').glob('*.json'))
    files += [ROOT/'data/processed/test/test.npz',ROOT/'data/processed/label_mapping.json',ROOT/'data/processed/preprocessing_metadata.json']
    manifest=[]
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(files)):
            relative=str(p.relative_to(ROOT));z.write(p,'ML-IDS/'+relative)
            with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            manifest.append({'path':relative,'bytes':p.stat().st_size,'sha256':digest})
        manifest_text=json.dumps({'files':manifest,'note':'Inference package includes the full held-out test set. Full raw/train/validation datasets, the 1.5 GB random forest checkpoint, virtual environments and local network logs are excluded. Use the original project folder for full retraining.'},indent=2)
        z.writestr('ML-IDS/output/delivery/package_manifest.json',manifest_text)
    (ROOT/'output/delivery/package_manifest.json').write_text(manifest_text)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
    print(f'Created {target} ({target.stat().st_size/1024**2:.1f} MB; {len(manifest)} files)')

if __name__=='__main__':main()
