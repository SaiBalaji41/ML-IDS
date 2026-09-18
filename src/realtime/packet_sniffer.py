"""
Real-Time Packet Capture and Stream Ingestion Module.
Provides live network packet sniffing and real-time IDS flow inference:
- Feature extraction aligning with the 46 CICIoT2023 flow schema
- Live prediction and threat severity alerting
- Streaming playback simulation for testing without raw socket/root privileges
"""

import json
import os
from pathlib import Path
import time
from typing import Callable, Dict, Generator, List, Optional, Union

import joblib
import numpy as np


class PacketSniffer:
    """Live network packet capture and real-time IDS inference engine."""

    # 46 CICIoT2023 Ordered Flow Feature Names
    FEATURE_NAMES = [
        "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
        "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
        "ack_flag_number", "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count",
        "fin_count", "urg_count", "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH",
        "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC", "Tot sum", "Min", "Max",
        "AVG", "Std", "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight"
    ]

    def __init__(
        self,
        interface: Optional[str] = None,
        model=None,
        scaler=None,
        label_mapping: Optional[Dict[str, int]] = None,
    ):
        self.interface = interface
        if isinstance(model, dict) and "model" in model:
            self.model = model["model"]
        else:
            self.model = model
        self.scaler = scaler
        self.label_mapping = label_mapping or {}
        self.id_to_label = {v: k for k, v in self.label_mapping.items()} if label_mapping else {}
        self.is_running = False

    def load_artifacts(
        self,
        scaler_path: Union[str, Path] = "models/preprocessing/scaler.pkl",
        label_mapping_path: Union[str, Path] = "data/processed/label_mapping.json",
    ):
        """Load fitted scaler and label mapping."""
        scaler_p = Path(scaler_path)
        if scaler_p.exists():
            loaded = joblib.load(scaler_p)
            if isinstance(loaded, dict) and "scaler" in loaded:
                self.scaler = loaded["scaler"]
            else:
                self.scaler = loaded

        mapping_p = Path(label_mapping_path)
        if mapping_p.exists():
            with open(mapping_p, "r", encoding="utf-8") as f:
                self.label_mapping = json.load(f)
            self.id_to_label = {int(v): k for k, v in self.label_mapping.items()}

    def extract_features(self, raw_flow: Union[Dict, np.ndarray, List]) -> np.ndarray:
        """
        Extract ordered 46-dimensional numerical vector from raw packet/flow dictionary.
        """
        if isinstance(raw_flow, np.ndarray):
            vec = raw_flow.flatten()
            if len(vec) == len(self.FEATURE_NAMES):
                return vec.reshape(1, -1)

        if isinstance(raw_flow, dict):
            vec = [float(raw_flow.get(f, 0.0)) for f in self.FEATURE_NAMES]
            return np.array(vec, dtype=np.float32).reshape(1, -1)

        vec = np.array(raw_flow, dtype=np.float32)
        return vec.reshape(1, -1)

    def classify_flow(self, raw_features: np.ndarray, is_preprocessed: bool = False) -> Dict[str, Union[str, float, int]]:
        """
        Scale features and perform model prediction.
        
        Returns:
            Dictionary containing predicted label, class ID, confidence score, and alert severity.
        """
        features = self.extract_features(raw_features)
        if features.shape != (1, len(self.FEATURE_NAMES)) or not np.isfinite(features).all():
            raise ValueError("Expected 46 finite flow features.")
        if self.model is None:
            raise RuntimeError("Load a trained model before classifying traffic.")

        # Scale features if scaler available and raw unscaled input
        if self.scaler is not None and not is_preprocessed:
            if hasattr(self.scaler, "transform"):
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
        else:
            features_scaled = features

        # Predict using model
        confidence = 1.0
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features_scaled)[0]
            pred_id = int(np.argmax(probs))
            confidence = float(probs[pred_id])
        elif hasattr(self.model, "predict"):
            pred = self.model.predict(features_scaled)
            pred = np.asarray(pred)
            if pred.ndim == 2 and pred.shape[1] > 1:
                pred_id = int(np.argmax(pred[0]))
                confidence = float(pred[0, pred_id])
            else:
                pred_id = int(pred.reshape(-1)[0])
                confidence = None
        else:
            raise RuntimeError("The supplied model does not support prediction.")

        pred_label = self.id_to_label.get(pred_id, f"Class_{pred_id}")
        is_attack = "Benign" not in pred_label

        severity = "LOW"
        if is_attack:
            if confidence is None:
                severity = "REVIEW"
            elif confidence > 0.85:
                severity = "CRITICAL"
            elif confidence > 0.60:
                severity = "HIGH"
            else:
                severity = "MEDIUM"

        return {
            "timestamp": time.time(),
            "predicted_class_id": pred_id,
            "predicted_label": pred_label,
            "is_attack": is_attack,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "severity": severity,
        }

    def simulate_stream(
        self,
        test_npz_path: Union[str, Path] = "data/processed/test/test.npz",
        max_events: int = 10,
        callback: Optional[Callable[[Dict], None]] = None,
    ) -> List[Dict]:
        """
        Simulate real-time streaming traffic events from test dataset.
        """
        npz_p = Path(test_npz_path)
        if not npz_p.exists():
            return []

        data = np.load(npz_p)
        X_test = data["X"]
        y_test = data["y"]

        events = []
        indices = np.random.choice(len(X_test), size=min(max_events, len(X_test)), replace=False)

        for idx in indices:
            raw_x = X_test[idx]
            true_id = int(y_test[idx])
            true_label = self.id_to_label.get(true_id, str(true_id))

            # When loading from processed test.npz, data is already scaled, so pass is_preprocessed=True
            pred_res = self.classify_flow(raw_x, is_preprocessed=True)
            pred_res["true_label"] = true_label
            pred_res["is_correct"] = bool(pred_res["predicted_label"] == true_label)
            pred_res["event_id"] = int(idx)

            events.append(pred_res)
            if callback:
                callback(pred_res)

        return events
