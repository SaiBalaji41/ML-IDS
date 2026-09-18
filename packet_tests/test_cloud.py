from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'cloud/app.py'


def test_cloud_uploads_and_bytes_work_without_exposing_local_admin_controls():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert not any(t.label == 'Model directory' for t in app.text_input)
    app.radio(key='page').set_value('PCAP analysis').run()
    assert not app.exception
    assert any(b.label == 'Analyze capture' for b in app.button)
    assert not any(b.label in ('Start capture', 'Stop capture') for b in app.button)
    assert 'capture_monitor' not in app.session_state
    app.radio(key='page').set_value('Packet inspector').run()
    app.text_area[0].set_value('00 7f ff')
    next(b for b in app.button if b.label == 'Inspect payload').click().run()
    assert not app.exception and app.session_state.packet_payload == b'\x00\x7f\xff'
    app.radio(key='page').set_value('Model evaluation').run()
    assert not app.exception
    assert any('No trained packet model' in item.value for item in app.info)
    assert not any(b.label in ('Start training', 'Prepare dataset', 'Create blank manifest',
                              'Run nested validation') for b in app.button)
    app.radio(key='page').set_value('Byte explanations').run()
    assert not app.exception
    assert next(b for b in app.button if b.label == 'Explain packet with SHAP').disabled
