"""Regression tests for real inference, strict input parity and safe HTTP handling."""
import io
import json
from pathlib import Path
import numpy as np
import pytest
from src.preprocessing.packet_bytes import payload_to_array,parse_hex
from src.realtime.engine import IDSEngine
from src.realtime.capture import CaptureService
from scripts.stream_ids import process_message
from scripts.train_packet_bytes_legacy import load_manifest
from dashboard import server

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def engine():
    e=IDSEngine(ROOT);e.load_pool();return e


def test_padding_truncation_normalization():
    assert np.array_equal(payload_to_array(bytes([0,255,128]),8),[0,1,np.float32(128/255),0,0,0,0,0])
    assert np.all(payload_to_array(b'\xff'*20,8)==1)
    assert parse_hex('00 ff')==b'\x00\xff'
    for invalid in ['', 'xx', 'f']:
        with pytest.raises(ValueError):parse_hex(invalid)


def test_complete_schema_and_nonfinite_rejected(engine):
    row=engine.presets()[0]['features']
    assert engine.raw_vector(row).shape==(1,46)
    for bad in [{}, {**row,'unknown':2},{**row,engine.features[0]:float('nan')},{**row,engine.features[0]:'no'}]:
        with pytest.raises(ValueError):engine.raw_vector(bad)


def test_scale_roundtrip(engine):
    p=engine.presets()[0]
    i=int(np.where(engine.indices==p['sample_id'])[0][0])
    assert np.allclose(engine.raw_vector(p['features'])[0],engine.pool[i],atol=2e-5)


def test_real_hybrid_probabilities_and_replay(engine):
    p=engine.probabilities(engine.pool[:3])
    assert p.shape==(3,34)
    assert np.allclose(p.sum(axis=1),1,atol=1e-5)
    events=engine.replay(3)
    assert len(events)==3 and all(e['model']=='cnn_bilstm' for e in events)
    assert all(e['source']=='held_out_replay' and 'src_ip' not in e for e in events)
    for count in [-1,0,51]:
        with pytest.raises(ValueError):engine.replay(count)
    with pytest.raises(ValueError):engine.probabilities(engine.pool[:1],'imaginary_model')


def test_missing_packet_checkpoint_is_not_flow_inference(tmp_path):
    with pytest.raises(RuntimeError,match='Packet-byte detection needs'):
        CaptureService(tmp_path).classify_hex('00010203')


def test_kafka_schema_routing(engine,tmp_path):
    c=CaptureService(tmp_path)
    with pytest.raises(ValueError):process_message({'features':{}},engine,c)
    with pytest.raises(RuntimeError):process_message({'representation':'raw_payload_bytes','payload_hex':'00ff'},engine,c)
    p=engine.presets()[0]
    event=process_message({'representation':'flow_features','features':p['features']},engine,c)
    assert event['predicted_label'] in engine.labels


def make_handler(path,method='GET',data=None,headers=None):
    handler=object.__new__(server.DashboardRequestHandler)
    handler.path=path;handler.command=method;handler.requestline=f'{method} {path} HTTP/1.1'
    handler.headers=headers or {};handler.result=None
    handler.wfile=io.BytesIO()
    handler.send_json=lambda value,status=200:setattr(handler,'result',(status,value))
    handler.send_bytes=lambda value,content_type,status=200,**kw:setattr(handler,'result',(status,value))
    handler.send_response=lambda code,message=None:setattr(handler,'result',(code,message))
    handler.send_header=lambda k,v:None
    handler.end_headers=lambda:None
    handler.log_message=lambda *args:None
    if data is not None:
        raw=json.dumps(data).encode();handler.rfile=io.BytesIO(raw);handler.headers['Content-Length']=str(len(raw))
    return handler


def test_static_path_traversal_denied():
    for path in ['/../../README.md','/%2e%2e/README.md','/results/../../README.md','/server.py','/artifacts/../../models/xgboost.pkl']:
        h=make_handler(path);h.do_GET();assert h.result[0]==404


def test_post_invalid_json_origin_and_size():
    h=make_handler('/api/classify',data={});h.rfile=io.BytesIO(b'{oops');h.do_POST();assert h.result[0]==400
    h=make_handler('/api/classify',data={},headers={'Origin':'http://elsewhere.example','Host':'127.0.0.1:8765'});h.do_POST();assert h.result[0]==403
    h=make_handler('/api/classify',headers={'Content-Length':str(5*1024*1024)});h.do_POST();assert h.result[0]==413


def test_csv_validation_and_batch_prediction(engine,monkeypatch):
    import csv
    monkeypatch.setattr(server,'ENGINE',engine)
    row=engine.presets()[0]['features'];out=io.StringIO();writer=csv.DictWriter(out,fieldnames=engine.features);writer.writeheader();writer.writerow(row)
    h=make_handler('/api/analyze-csv',data={'csv':out.getvalue()});h.do_POST()
    assert h.result[0]==200 and h.result[1]['rows']==1
    for text in ['', 'wrong,column\n1,2', ','.join(engine.features)+'\n', out.getvalue().replace(str(row['flow_duration']),'nan')]:
        h=make_handler('/api/analyze-csv',data={'csv':text});h.do_POST();assert h.result[0]==400


def test_manifest_duplicate_capture_rejected(tmp_path):
    from scapy.all import Ether,IP,UDP,Raw,wrpcap
    wrpcap(str(tmp_path/'a.pcap'),[Ether(src='00:11:22:33:44:55', dst='66:77:88:99:aa:bb')/IP(src='1.1.1.1', dst='2.2.2.2')/UDP()/Raw(b'hello')])
    (tmp_path/'manifest.csv').write_text('path,label,split\na.pcap,BenignTraffic,train\na.pcap,Attack,test\n')
    with pytest.raises(ValueError,match='Duplicate capture'):load_manifest(tmp_path/'manifest.csv')


def test_shap_is_additive_on_actual_checkpoint(engine):
    p=engine.presets()[0]
    result=engine.explain({'features':p['features'],'model':'cnn_bilstm'})
    assert len(result['features'])==46
    assert abs(result['additivity_residual'])<1e-4
    assert all(np.isfinite(f['contribution']) for f in result['features'])


def test_preprocessor_uses_training_imputation_only():
    import pandas as pd
    from src.preprocessing.preprocessor import DataPreprocessor
    p=DataPreprocessor();p.fit_transform(pd.DataFrame({'a':[0.,2.,float('nan')],'b':[1.,1.,1.]}))
    medians=p.medians.copy()
    missing=p.transform(pd.DataFrame({'b':[1.],'a':[float('nan')]}))
    assert np.allclose(missing,0)
    p.transform(pd.DataFrame({'a':[1000.],'b':[1.]}))
    assert p.medians.equals(medians)
    with pytest.raises(ValueError):p.transform(pd.DataFrame({'c':[1.]}))


def test_legacy_sniffer_does_not_invent_prediction():
    from src.realtime.packet_sniffer import PacketSniffer
    with pytest.raises(RuntimeError):PacketSniffer().classify_flow(np.zeros(46))
