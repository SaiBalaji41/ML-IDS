"""Convenience entry point for the required 1024-byte PyTorch training pipeline."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/packet_processed')
    parser.add_argument('--output', type=Path, default=ROOT / 'models/pytorch_packet_ids')
    parser.add_argument('--max-per-capture', type=int, default=10000)
    parser.add_argument('--benign-label', default='BenignTraffic')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    from packet_ids.data import prepare_dataset
    from packet_ids.training import train_dataset
    prepare_dataset(args.manifest, args.data, args.max_per_capture, benign_label=args.benign_label)
    train_dataset(args.data, args.output, args.epochs, args.batch_size)


if __name__ == '__main__':
    main()
