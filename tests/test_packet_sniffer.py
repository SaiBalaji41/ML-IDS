"""
Unit tests for Real-Time Packet Sniffer and Stream Ingestion module.
"""

from pathlib import Path
import numpy as np
import pytest

from src.realtime.packet_sniffer import PacketSniffer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MockModel:
    """Lightweight mock model for sniffer unit tests."""
    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        probs = np.zeros((len(X), 34), dtype=float)
        probs[:, 0] = 0.95
        probs[:, 1] = 0.05
        return probs


def test_packet_sniffer_init():
    """Verify default initialization and feature names length."""
    sniffer = PacketSniffer()
    assert len(sniffer.FEATURE_NAMES) == 46
    assert sniffer.is_running is False
    assert sniffer.model is None


def test_packet_sniffer_feature_extraction():
    """Verify feature extraction from dict, list, and ndarray."""
    sniffer = PacketSniffer()
    
    # 1. From dictionary
    sample_dict = {"flow_duration": 1.25, "Rate": 100.5, "syn_flag_number": 1.0}
    vec_dict = sniffer.extract_features(sample_dict)
    assert vec_dict.shape == (1, 46)
    assert vec_dict[0, 0] == 1.25
    assert vec_dict[0, 4] == 100.5

    # 2. From list
    sample_list = [0.0] * 46
    sample_list[0] = 3.5
    vec_list = sniffer.extract_features(sample_list)
    assert vec_list.shape == (1, 46)
    assert vec_list[0, 0] == 3.5

    # 3. From numpy array
    sample_arr = np.ones((1, 46), dtype=np.float32)
    vec_arr = sniffer.extract_features(sample_arr)
    assert vec_arr.shape == (1, 46)
    assert np.allclose(vec_arr, 1.0)


def test_packet_sniffer_classify_flow():
    """Verify end-to-end flow classification with mock model."""
    mock_model = MockModel()
    label_mapping = {"Backdoor_Malware": 0, "BenignTraffic": 1}
    sniffer = PacketSniffer(model=mock_model, label_mapping=label_mapping)
    sniffer.id_to_label = {0: "Backdoor_Malware", 1: "BenignTraffic"}

    raw_features = np.zeros(46)
    res = sniffer.classify_flow(raw_features, is_preprocessed=True)

    assert res["predicted_label"] == "Backdoor_Malware"
    assert res["predicted_class_id"] == 0
    assert res["confidence"] == 0.95
    assert res["severity"] == "CRITICAL"


def test_packet_sniffer_stream_simulation():
    """Verify stream simulation generator."""
    mock_model = MockModel()
    sniffer = PacketSniffer(model=mock_model)
    sniffer.id_to_label = {0: "Backdoor_Malware", 1: "BenignTraffic"}

    test_npz = PROJECT_ROOT / "data" / "processed" / "test" / "test.npz"
    if test_npz.exists():
        events = sniffer.simulate_stream(test_npz_path=test_npz, max_events=3, callback=None)
        assert len(events) == 3
        assert "predicted_label" in events[0]
        assert "severity" in events[0]
