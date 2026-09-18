"""Required project workflow: raw CICIoT2023 -> 1024 bytes -> PyTorch -> SHAP."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ml-ids-matplotlib')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    audit = commands.add_parser('audit')
    audit.add_argument('--path', type=Path, default=Path('/Users/harsha/Downloads/CICIOT23'))
    initialize = commands.add_parser('init-manifest', help='Create an empty manifest for verified PCAP paths')
    initialize.add_argument('--manifest', type=Path, default=ROOT / 'data/packet_manifest.csv')
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--manifest', type=Path, required=True)
    prepare.add_argument('--output', type=Path, default=ROOT / 'data/packet_processed')
    prepare.add_argument('--max-per-capture', type=int, default=10000)
    prepare.add_argument('--benign-label', default='BenignTraffic')
    train = commands.add_parser('train')
    train.add_argument('--data', type=Path, default=ROOT / 'data/packet_processed')
    train.add_argument('--output', type=Path, default=ROOT / 'models/pytorch_packet_ids')
    train.add_argument('--epochs', type=int, default=25)
    train.add_argument('--batch-size', type=int, default=128)
    train.add_argument('--patience', type=int, default=5)
    validation = commands.add_parser('validate')
    validation.add_argument('--data', type=Path, default=ROOT / 'data/packet_processed')
    validation.add_argument('--output', type=Path, default=ROOT / 'output/pytorch/nested_validation')
    validation.add_argument('--outer-folds', type=int, default=5)
    validation.add_argument('--inner-folds', type=int, default=3)
    validation.add_argument('--bootstrap', type=int, default=100)
    validation.add_argument('--epochs', type=int, default=5)
    validation.add_argument('--batch-size', type=int, default=128)
    capture = commands.add_parser('capture')
    capture.add_argument('--interface', required=True)
    capture.add_argument('--seconds', type=int, default=30)
    capture.add_argument('--limit', type=int, default=1000)
    capture.add_argument('--save-directory', type=Path)
    capture.add_argument('--model', type=Path, default=ROOT / 'models/pytorch_packet_ids')
    capture.add_argument('--observe-only', action='store_true')
    pcap = commands.add_parser('pcap')
    pcap.add_argument('--path', type=Path, required=True)
    pcap.add_argument('--limit', type=int, default=500)
    pcap.add_argument('--model', type=Path, default=ROOT / 'models/pytorch_packet_ids')
    pcap.add_argument('--observe-only', action='store_true')
    for name in ('predict', 'explain', 'replay'):
        item = commands.add_parser(name)
        item.add_argument('--model', type=Path, default=ROOT / 'models/pytorch_packet_ids')
        if name == 'replay':
            item.add_argument('--count', type=int, default=10)
        else:
            item.add_argument('--hex', required=True, dest='payload_hex')
        if name == 'explain':
            item.add_argument('--samples', type=int, default=256)
    args = parser.parse_args()
    if args.command == 'audit':
        if not args.path.is_dir():
            parser.error(f'Dataset folder does not exist: {args.path}')
        captures = sorted(str(p) for p in args.path.rglob('*') if p.suffix.lower() in ('.pcap', '.pcapng', '.pcp', '.cap'))
        csvs = sorted(str(p) for p in args.path.rglob('*.csv'))
        import csv
        csv_schemas = []
        for name in csvs:
            with Path(name).open(newline='', encoding='utf-8-sig') as stream:
                columns = next(csv.reader(stream), [])
            csv_schemas.append({'path': name, 'bytes': Path(name).stat().st_size,
                                'columns': columns, 'feature_count_excluding_label': len([c for c in columns if c != 'label'])})
        result = {'folder': str(args.path), 'pcap_files': captures, 'csv_files': csvs,
                  'csv_schemas': csv_schemas,
                  'raw_payload_data_available': bool(captures),
                  'note': 'Flow CSVs do not contain recoverable packet payload bytes.'}
        from packet_ids.training import write_json
        write_json(ROOT / 'output/pytorch/dataset_audit.json', result)
    elif args.command == 'init-manifest':
        from packet_ids.data import initialize_manifest
        try:
            path = initialize_manifest(args.manifest)
        except OSError as exc:
            parser.exit(2, f'Manifest was not created: {exc}\nExisting files are never overwritten.\n')
        result = {'manifest': str(path), 'ready_for_training': False,
                  'next_step': 'Add verified labeled CICIoT2023 PCAP paths; see docs/packet_manifest.example.csv.'}
    elif args.command == 'prepare':
        from packet_ids.data import prepare_dataset
        try:
            result = prepare_dataset(args.manifest, args.output, args.max_per_capture, benign_label=args.benign_label)
        except (OSError, ValueError) as exc:
            parser.exit(2, f'Dataset preparation cannot start: {exc}\n')
    elif args.command == 'train':
        from packet_ids.training import train_dataset, write_json
        try:
            metadata, evaluation = train_dataset(args.data, args.output, args.epochs,
                                                args.batch_size, patience=args.patience)
            result = {'metadata': metadata, 'evaluation': evaluation}
        except Exception as exc:
            if not (args.output / 'metadata.json').exists():
                write_json(args.output / 'status.json', {'state': 'failed', 'error': str(exc)})
            raise
    elif args.command == 'validate':
        from packet_ids.validation import run_nested_validation
        from packet_ids.training import write_json
        try:
            result = run_nested_validation(args.data, args.output, args.outer_folds, args.inner_folds,
                                            args.bootstrap, args.epochs, args.batch_size)
        except Exception as exc:
            if not (args.output / 'cross_validation.json').exists():
                write_json(args.output / 'status.json', {'state': 'failed', 'error': str(exc)})
            raise
    elif args.command in ('capture', 'pcap'):
        from packet_ids.service import PacketIDS
        engine = None if args.observe_only else PacketIDS(args.model)
        if args.command == 'pcap':
            from packet_ids.events import inspect_pcap
            result = inspect_pcap(str(args.path), engine, args.limit)
        else:
            from packet_ids.capture import CaptureMonitor
            import time
            monitor = CaptureMonitor()
            monitor.start(args.interface, engine, args.seconds, args.limit, args.save_directory)
            try:
                while monitor.snapshot()['running']:
                    time.sleep(.1)
            except KeyboardInterrupt:
                monitor.stop()
                if monitor.worker:
                    monitor.worker.join(timeout=3)
            result = monitor.snapshot()
            if result['error']:
                print(json.dumps(result, indent=2))
                raise SystemExit(1)
    else:
        from packet_ids.service import PacketIDS
        from packet_ids.preprocessing import parse_payload_hex
        engine = PacketIDS(args.model)
        if args.command == 'replay':
            result = engine.replay(args.count, seed=42)
        elif args.command == 'predict':
            result = engine.predict(parse_payload_hex(args.payload_hex))
        else:
            from packet_ids.explain import explain_payload
            result = explain_payload(engine, parse_payload_hex(args.payload_hex), samples=args.samples)
    print(json.dumps(result, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
