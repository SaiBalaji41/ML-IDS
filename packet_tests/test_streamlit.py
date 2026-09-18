from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'streamlit_app/app.py'


def test_untrained_ui_and_interactive_byte_preprocessing(tmp_path):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert any('byte model has not been trained' in warning.value for warning in app.warning)
    assert next(b for b in app.button if b.label == 'Run 5 packets').disabled
    assert app.metric[-1].value == '—'
    app.radio(key='page').set_value('Packet inspector').run()
    app.text_area[0].set_value('00 7f ff')
    next(b for b in app.button if b.label == 'Inspect payload').click().run()
    assert not app.exception
    assert app.session_state.packet_payload == b'\x00\x7f\xff'
    assert next(b for b in app.button if b.label == 'Classify packet').disabled
    app.radio(key='page').set_value('Byte explanations').run()
    assert not app.exception
    assert next(b for b in app.button if b.label == 'Explain packet with SHAP').disabled
    app.radio(key='page').set_value('Training & evaluation').run()
    assert not app.exception
    assert next(b for b in app.button if b.label == 'Prepare dataset').disabled
    manifest = tmp_path / 'new-manifest.csv'
    next(t for t in app.text_input if t.label == 'Manifest path').set_value(str(manifest)).run()
    assert next(b for b in app.button if b.label == 'Prepare dataset').disabled
    next(b for b in app.button if b.label == 'Create blank manifest').click().run()
    assert manifest.is_file() and not app.exception
    assert next(b for b in app.button if b.label == 'Prepare dataset').disabled
    assert any('contains no captures' in message.value for message in app.info)
    assert next(b for b in app.button if b.label == 'Start training').disabled
    assert next(b for b in app.button if b.label == 'Run nested validation').disabled
    app.radio(key='page').set_value('Live capture & PCAP').run()
    assert not app.exception
    assert next(b for b in app.button if b.label == 'Start capture').disabled
    next(r for r in app.radio if r.label == 'Monitoring source').set_value('PCAP file').run()
    assert not app.exception
    assert next(b for b in app.button if b.label == 'Analyze capture').disabled


def test_bad_hex_shows_actionable_error():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.radio(key='page').set_value('Packet inspector').run()
    app.text_area[0].set_value('this is not hexadecimal')
    next(b for b in app.button if b.label == 'Inspect payload').click().run()
    assert not app.exception
    assert any('hexadecimal byte pairs' in error.value for error in app.error)


def test_model_ui_replay_classification_shap_and_evaluation(trained_fixture, monkeypatch):
    # Explicit test injection only. The shipped app's strict loader rejects
    # this synthetic checkpoint, covered by the pipeline test.
    import packet_ids.service
    model_dir, *_ = trained_fixture
    engine = packet_ids.service.PacketIDS(model_dir, allow_test_fixture=True)
    monkeypatch.setattr(packet_ids.service, 'PacketIDS', lambda directory: engine)
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state.model_directory = str(model_dir)
    app.run()
    assert not app.exception
    next(b for b in app.button if b.label == 'Run 5 packets').click().run()
    assert not app.exception and app.metric[0].value == '5'
    assert len(app.session_state.events) == 5
    app.radio(key='page').set_value('Packet inspector').run()
    app.text_area[0].set_value('1e ' * 67)
    next(b for b in app.button if b.label == 'Inspect payload').click().run()
    next(b for b in app.button if b.label == 'Classify packet').click().run()
    assert not app.exception and app.session_state.last_prediction['used_bytes'] == 67
    app.radio(key='page').set_value('Byte explanations').run()
    app.select_slider[0].set_value(64)
    next(b for b in app.button if b.label == 'Explain packet with SHAP').click().run()
    assert not app.exception and len(app.session_state.byte_explanation['attributions']) == 1024
    app.radio(key='page').set_value('Training & evaluation').run()
    assert not app.exception
    assert any(metric.label == 'Macro F1' for metric in app.metric)
