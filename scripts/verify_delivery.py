"""Exercise the running local API and save reproducible delivery evidence."""
import argparse
import csv
import io
import json
from pathlib import Path
import urllib.request
import urllib.error
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8765');args=p.parse_args()
    evidence={}
    def request(path,data=None,expected=200):
        req=urllib.request.Request(args.url+path,data=json.dumps(data).encode() if data is not None else None,
                                   headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=180) as r:
                status=r.status;body=r.read();content=r.headers.get('Content-Type','')
        except urllib.error.HTTPError as e:status=e.code;body=e.read();content=e.headers.get('Content-Type','')
        assert status==expected,(path,status,body[:200])
        return json.loads(body) if 'json' in content else body
    stats=request('/api/stats');assert stats['model_ready'] and stats['training']['state']=='complete' and not stats['errors'],stats
    evidence['stats']=stats
    presets=request('/api/presets');assert len(presets)>=3 and len(presets[0]['features'])==46
    examples=[]
    for sample in presets:
        result=request('/api/classify',{'model':'cnn_bilstm','features':sample['features']})
        examples.append({'ground_truth':sample['name'],'sample_id':sample['sample_id'],'result':result})
    evidence['examples']=examples
    replay=request('/api/replay?count=10');assert len(replay['events'])==10
    evidence['replay']=replay
    xgb=request('/api/classify',{'model':'xgboost','features':presets[0]['features']});evidence['xgboost']=xgb
    template=request('/api/template').decode();batch=request('/api/analyze-csv',{'csv':template});assert batch['rows']==1
    evidence['csv']=batch
    explanation=request('/api/explain',{'features':presets[0]['features'],'model':'cnn_bilstm'})
    assert abs(explanation['additivity_residual'])<1e-4
    evidence['explanation']=explanation
    assert request('/api/classify',{'features':{}},400)['error']
    assert request('/api/classify',{'model':'missing','features':presets[0]['features']},400)['error']
    request('/api/replay?count=-10',expected=400)
    request('/server.py',expected=404);request('/results/../../README.md',expected=404)
    request('/api/packet',{'payload_hex':'bad-hex'},400)
    packet=request('/api/packet',{'payload_hex':'00ff'},503);assert 'Packet-byte detection needs' in packet['error']
    capture=request('/api/capture');assert isinstance(capture['interfaces'],list)
    evidence['capture_observation_available']=True
    evidence['packet_byte_detection']='Needs labeled PCAP training data; raw bytes correctly rejected by the flow model.'
    evidence['checks']='API, checkpoint readiness, actual predictions, replay, CSV, local SHAP, schema errors, traversal protection, byte-model gating, capture interface discovery.'
    (ROOT/'output/delivery/verification.json').write_text(json.dumps(evidence,indent=2))
    (ROOT/'output/delivery/example_flows.csv').write_text(template)
    (ROOT/'output/delivery/example_flow.json').write_text(json.dumps(presets[0]['features'],indent=2))
    print(json.dumps({'status':'passed','accuracy':stats['accuracy'],'sample_count':stats['test_samples'],'examples':examples},indent=2))

if __name__=='__main__':main()
