"""
ML-IDS Master Pipeline Execution Entrypoint.
Coordinates end-to-end evaluation, SHAP explainability, model comparisons, and real-time streaming inference.

Usage:
    python scripts/run_pipeline.py --phase evaluate
    python scripts/run_pipeline.py --phase explain
    python scripts/run_pipeline.py --phase realtime
    python scripts/run_pipeline.py --phase all
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_evaluation():
    print("\n" + "=" * 60)
    print("EXECUTING CONSOLIDATED MODEL EVALUATION & COMPARISON")
    print("=" * 60)
    from scripts.generate_model_comparisons import main as eval_main
    eval_main()


def run_explainability():
    print("\n" + "=" * 60)
    print("EXECUTING EXPLAINABLE AI (SHAP) ATTRIBUTION GENERATOR")
    print("=" * 60)
    from scripts.generate_shap_explanations import main as shap_main
    shap_main()


def run_realtime_simulation(samples: int = 15):
    print("\n" + "=" * 60)
    print("EXECUTING REAL-TIME PACKET SNIFFING & FLOW STREAM SIMULATION")
    print("=" * 60)
    import joblib
    from src.realtime.packet_sniffer import PacketSniffer

    model_path = PROJECT_ROOT / "models" / "xgboost.pkl"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "models" / "random_forest.pkl"

    if model_path.exists():
        loaded = joblib.load(model_path)
        model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
        sniffer = PacketSniffer(model=model)
        sniffer.load_artifacts(
            scaler_path=PROJECT_ROOT / "models" / "preprocessing" / "scaler.pkl",
            label_mapping_path=PROJECT_ROOT / "data" / "processed" / "label_mapping.json",
        )

        def alert_callback(event):
            sev_badge = f"[{event['severity']}]"
            stat = "[PASS]" if event.get("is_correct") else "[MISMATCH]"
            print(f"[{time.strftime('%H:%M:%S')}] {sev_badge:<10} True: {event['true_label']:<22} -> Pred: {event['predicted_label']:<22} (Conf: {event['confidence']*100:.1f}%) {stat}")

        print("Simulating live high-speed network event stream from held-out test partition:")
        print("-" * 85)
        events = sniffer.simulate_stream(
            test_npz_path=PROJECT_ROOT / "data" / "processed" / "test" / "test.npz",
            max_events=samples,
            callback=alert_callback,
        )
        print("-" * 85)
        print(f"Processed {len(events)} streaming events with real-time classification.")
    else:
        print(f"No trained model found at {model_path} for real-time simulation.")


def run_dashboard(port: int = 8080):
    print("\n" + "=" * 60)
    print("LAUNCHING INTERACTIVE ML-IDS SOC DASHBOARD")
    print("=" * 60)
    from dashboard.server import run_server
    run_server(port=port)


def main():
    parser = argparse.ArgumentParser(
        description="ML-Powered Intrusion Detection System (IDS) Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to pipeline configuration YAML file",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["evaluate", "explain", "realtime", "dashboard", "all"],
        default="all",
        help="Pipeline phase to execute",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=15,
        help="Number of network events to simulate in real-time phase",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run the dashboard server on",
    )

    args = parser.parse_args()
    print("=" * 60)
    print("ML-POWERED INTRUSION DETECTION SYSTEM (IDS) MASTER PIPELINE")
    print(f"Config: {args.config} | Phase: {args.phase}")
    print("=" * 60)

    if args.phase in ["evaluate", "all"]:
        run_evaluation()

    if args.phase in ["explain", "all"]:
        run_explainability()

    if args.phase in ["realtime", "all"]:
        run_realtime_simulation(samples=args.samples)

    if args.phase == "dashboard":
        run_dashboard(port=args.port)

    print("\n" + "=" * 60)
    print("ML-IDS PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
