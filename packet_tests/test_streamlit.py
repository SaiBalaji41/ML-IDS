from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'streamlit_app/app.py'


def test_untrained_ui_and_interactive_byte_preprocessing(tmp_path):
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state['model_directory'] = str(tmp_path / 'untrained_model')
    app.run()
    assert not app.exception
    assert any('byte model has not been trained' in warning.value for warning in app.warning)
    assert next(b for b in app.button if b.label == 'Run 5 packets').disabled
    assert app.metric[0].value in ('0', 0, '30', 30)

    # Test Traffic analyzer tab
    app.radio(key='page').set_value('Traffic analyzer').run()
    assert not app.exception
    app.text_area[0].set_value('00 7f ff')
    next(b for b in app.button if b.label == 'Inspect payload').click().run()
    assert not app.exception
    assert app.session_state.packet_payload == b'\x00\x7f\xff'
    assert next(b for b in app.button if b.label == 'Classify packet').disabled

    # Test Model performance tab
    app.radio(key='page').set_value('Model performance').run()
    assert not app.exception


def test_bad_hex_shows_actionable_error():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.radio(key='page').set_value('Traffic analyzer').run()
    app.text_area[0].set_value('this is not hexadecimal')
    next(b for b in app.button if b.label == 'Inspect payload').click().run()
    assert not app.exception
    assert any('hexadecimal byte pairs' in error.value for error in app.error)


def test_model_ui_replay_classification_and_evaluation(trained_fixture, monkeypatch):
    import packet_ids.service
    model_dir, *_ = trained_fixture
    engine = packet_ids.service.PacketIDS(model_dir, allow_test_fixture=True)
    monkeypatch.setattr(packet_ids.service, 'PacketIDS', lambda directory: engine)
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state.model_directory = str(model_dir)
    app.run()
    assert not app.exception

    # Run replay simulation
    next(b for b in app.button if b.label == 'Run 5 packets').click().run()
    assert not app.exception
    assert len(app.session_state.events) == 5

    # Switch to Traffic analyzer
    app.radio(key='page').set_value('Traffic analyzer').run()
    app.text_area[0].set_value('1e ' * 67)
    next(b for b in app.button if b.label == 'Inspect payload').click().run()
    next(b for b in app.button if b.label == 'Classify packet').click().run()
    assert not app.exception and app.session_state.last_prediction['used_bytes'] == 67

    # Switch to Model performance
    app.radio(key='page').set_value('Model performance').run()
    assert not app.exception
    assert any(metric.label == 'Macro F1' for metric in app.metric)
