"""Local IDS dashboard: model inference, CSV analysis, SHAP and packet monitoring."""
import argparse
import csv
import io
import json
import mimetypes
from pathlib import Path
import sys
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))
from src.realtime.engine import IDSEngine, read_json

CACHE={'model_xgb':None,'model_rf':None,'scaler':None,'label_mapping':{},'id_to_label':{},'feature_names':[],'metrics_comparison':[]}
ENGINE=None
CAPTURE=None
MAX_BODY=4*1024*1024


def model_comparison():
    rows=[]
    for key,title in [('cnn_bilstm','CNN-BiLSTM'),('xgboost','XGBoost'),('random_forest','Random Forest'),('cnn_1d','1D-CNN'),('bilstm','BiLSTM')]:
        deployed=key=='cnn_bilstm' and (PROJECT_ROOT/'models/deployed/metadata.json').exists()
        metric=read_json(PROJECT_ROOT/('output/delivery/evaluation.json' if deployed else f'results/metrics/{key}_test.json'),{})
        meta=read_json(PROJECT_ROOT/('models/deployed/metadata.json' if deployed else f'models/{key}/metadata.json'),{})
        if not meta: meta=read_json(PROJECT_ROOT/f'models/{key}_metadata.json',{})
        rows.append({'key':key,'name':title,'metrics':metric,'metadata':meta,'source':'verified_delivery' if deployed else 'saved_research',
                     'available':key!='random_forest'})
    return rows


def load_backend_assets():
    global ENGINE,CAPTURE
    if ENGINE is None:
        ENGINE=IDSEngine(PROJECT_ROOT)
        ENGINE.load_pool()
        for key in ['cnn_bilstm','xgboost']:
            try: ENGINE.load_model(key)
            except Exception as e: ENGINE.errors[key]=str(e)
        from src.realtime.capture import CaptureService
        CAPTURE=CaptureService(PROJECT_ROOT)
    CACHE.update({'scaler':ENGINE.scaler,'label_mapping':ENGINE.mapping,'id_to_label':dict(enumerate(ENGINE.labels)),
                  'feature_names':ENGINE.features,'model_xgb':ENGINE.models.get('xgboost'),'metrics_comparison':model_comparison()})


class DashboardRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed=urlparse(self.path); path=unquote(parsed.path); query=parse_qs(parsed.query)
        try:
            if path.startswith('/api/'):
                self.handle_api_get(path,query);return
            if path.startswith('/results/'):
                base=PROJECT_ROOT/'results'; relative=path[len('/results/'):]
                if Path(relative).suffix.lower() not in ['.png','.json']: self.send_json({'error':'Not found'},404);return
            elif path.startswith('/artifacts/'):
                base=PROJECT_ROOT/'output/delivery';relative=path[len('/artifacts/'):]
                if relative not in ['evaluation.json','training_history.json','training_status.json','training_curves.png','intrusion_confusion_matrix.png','delivery_report.md']:
                    self.send_json({'error':'Not found'},404);return
            else:
                base=PROJECT_ROOT/'dashboard'; relative='index.html' if path=='/' else path.lstrip('/')
                if relative not in ['index.html','styles.css','app.js']: self.send_json({'error':'Not found'},404);return
            file=(base/relative).resolve()
            if not file.is_relative_to(base.resolve()): self.send_json({'error':'Not found'},404);return
            self.serve_file(file)
        except ValueError as e:self.send_json({'error':str(e)},400)
        except Exception as e:self.send_json({'error':str(e)},503)

    def handle_api_get(self,path,query):
        if path in ['/api/stats','/api/health']: self.send_json(ENGINE.stats())
        elif path=='/api/models':self.send_json({'comparison':model_comparison()})
        elif path=='/api/presets':self.send_json(ENGINE.presets())
        elif path=='/api/schema':self.send_json({'features':ENGINE.features,'labels':ENGINE.labels})
        elif path in ['/api/replay','/api/simulate']:
            events=ENGINE.replay(int(query.get('count',['10'])[0]),query.get('model',['cnn_bilstm'])[0])
            self.send_json({'source':'held_out_replay','events':events})
        elif path=='/api/evaluation':
            key=query.get('model',['cnn_bilstm'])[0]
            row=next((r for r in model_comparison() if r['key']==key),None)
            if row is None:raise ValueError('Unknown model.')
            self.send_json(row)
        elif path=='/api/training':self.send_json({'status':read_json(PROJECT_ROOT/'output/delivery/training_status.json',{}),'history':read_json(PROJECT_ROOT/'output/delivery/training_history.json',[])})
        elif path=='/api/capture':self.send_json(CAPTURE.status())
        elif path=='/api/template':
            row=ENGINE.presets()[0]
            out=io.StringIO();writer=csv.DictWriter(out,fieldnames=ENGINE.features);writer.writeheader();writer.writerow(row['features'])
            self.send_bytes(out.getvalue().encode(),'text/csv',filename='ciciot2023-flow-template.csv')
        else:self.send_json({'error':'Endpoint not found'},404)

    def do_POST(self):
        # Local UI only; reject cross-origin submissions to inference/capture endpoints.
        origin=self.headers.get('Origin')
        if origin and origin != 'http://'+self.headers.get('Host',''):
            self.send_json({'error':'Cross-origin requests are not allowed.'},403);return
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=MAX_BODY: self.send_json({'error':'Request body must be 1 byte to 4 MB.'},413);return
            data=json.loads(self.rfile.read(size))
            if not isinstance(data,dict):raise ValueError('Request must be a JSON object.')
            path=urlparse(self.path).path
            if path=='/api/classify':result=ENGINE.classify(data)
            elif path=='/api/explain':result=ENGINE.explain(data)
            elif path=='/api/analyze-csv':result=self.analyze_csv(data)
            elif path=='/api/capture/start':result=CAPTURE.start(data.get('interface') or None)
            elif path=='/api/capture/stop':result=CAPTURE.stop()
            elif path=='/api/packet':result=CAPTURE.classify_hex(data.get('payload_hex',''))
            else:self.send_json({'error':'Endpoint not found'},404);return
            self.send_json(result)
        except (ValueError,TypeError,KeyError) as e:self.send_json({'error':str(e)},400)
        except Exception as e:self.send_json({'error':str(e)},503)

    def analyze_csv(self,data):
        text=data.get('csv','')
        if not isinstance(text,str):raise ValueError('csv must be text.')
        reader=csv.DictReader(io.StringIO(text.lstrip('\ufeff')))
        if not reader.fieldnames or len(set(reader.fieldnames))!=len(reader.fieldnames):raise ValueError('Missing or duplicate CSV column names.')
        if set(reader.fieldnames) != set(ENGINE.features):raise ValueError('CSV must contain exactly the 46 feature columns. Download the template for the schema.')
        vectors=[]
        for i,row in enumerate(reader):
            if i>=500:raise ValueError('Upload at most 500 rows per batch.')
            try:vectors.append(ENGINE.raw_vector(row)[0])
            except ValueError as e:raise ValueError(f'CSV row {i+2}: {e}') from e
        if not vectors:raise ValueError('CSV has no data rows.')
        import numpy as np
        import time
        key=data.get('model','cnn_bilstm');start=time.perf_counter();probs=ENGINE.probabilities(np.asarray(vectors),key)
        elapsed=(time.perf_counter()-start)*1000/len(vectors)
        events=[{'row':i+2,**ENGINE.verdict(p,key,elapsed)} for i,p in enumerate(probs)]
        return {'source':'uploaded_csv','rows':len(events),'attacks':sum(e['is_attack'] for e in events),'events':events}

    def serve_file(self,path):
        if not path.is_file():self.send_json({'error':'Not found'},404);return
        self.send_bytes(path.read_bytes(),mimetypes.guess_type(str(path))[0] or 'application/octet-stream')

    def send_bytes(self,body,content_type,status=200,filename=None):
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'")
        if filename:self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
        self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass

    def send_json(self,data,status=200):
        self.send_bytes(json.dumps(data,allow_nan=False).encode(),'application/json; charset=utf-8',status)

    def log_message(self,*args):pass


def run_server(port=8080):
    load_backend_assets()
    server=ThreadingHTTPServer(('127.0.0.1',port),DashboardRequestHandler)
    print(f'ML-IDS dashboard: http://127.0.0.1:{port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        if CAPTURE:CAPTURE.stop()
        server.server_close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8080)
    run_server(p.parse_args().port)
