"""One validated inference path for dashboard, CSV and Kafka flow ingestion."""
import json
import os
from pathlib import Path
import threading
import time
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


class IDSEngine:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.models = {}
        self.errors = {}
        self.mapping = read_json(self.root/'data/processed/label_mapping.json', {})
        self.labels = [name for name, _ in sorted(self.mapping.items(), key=lambda x:x[1])]
        self.metadata = read_json(self.root/'data/processed/preprocessing_metadata.json', {})
        self.features = self.metadata.get('feature_names', [])
        artifact = joblib.load(self.root/'models/preprocessing/scaler.pkl')
        self.scaler = artifact.get('scaler') if isinstance(artifact,dict) else artifact
        self.rng = np.random.default_rng(42)
        self.pool = self.targets = self.indices = None
        self.explainer_background = None
        self.last_latency = None

    def load_pool(self):
        if self.pool is not None: return
        with np.load(self.root/'data/processed/test/test.npz') as d:
            x,y = d['X'],d['y']
            self.test_count = len(y)
            self.indices = np.sort(self.rng.choice(len(y),size=min(10000,len(y)),replace=False))
            self.pool,self.targets = x[self.indices],y[self.indices]

    def load_model(self, key):
        if key not in ['cnn_bilstm','xgboost','cnn_1d','bilstm','packet_bytes']:
            raise ValueError('Unknown model. Choose CNN-BiLSTM, XGBoost, 1D-CNN or BiLSTM.')
        with self.lock:
            if key in self.models: return self.models[key]
            if key == 'xgboost':
                artifact = joblib.load(self.root/'models/xgboost.pkl')
                model = artifact.get('model') if isinstance(artifact,dict) else artifact
            else:
                import tensorflow as tf
                folder = 'packet_bytes' if key == 'packet_bytes' else key
                if key == 'cnn_bilstm' and (self.root/'models/deployed/metadata.json').exists(): folder='deployed'
                path = self.root/f'models/{folder}/best_model.keras'
                if not path.exists(): raise RuntimeError(f'No trained {key} checkpoint is available.')
                model = tf.keras.models.load_model(path,compile=False)
                if key == 'cnn_bilstm':
                    self.hybrid_metadata = read_json(self.root/f'models/{folder}/metadata.json',{})
            self.models[key] = model
            return model

    def probabilities(self, x, key='cnn_bilstm'):
        x = np.asarray(x,dtype='float32')
        if x.ndim != 2 or x.shape[1] != len(self.features) or not np.isfinite(x).all():
            raise ValueError(f'Expected finite values with shape (rows, {len(self.features)}).')
        with self.lock:
            model = self.load_model(key)
            if key == 'xgboost': return np.asarray(model.predict_proba(x))
            if key == 'packet_bytes': raise ValueError('Use packet inference for byte models.')
            if key == 'cnn_bilstm' and self.hybrid_metadata.get('input_transform') == 'signed_log1p_standardized':
                x = np.sign(x)*np.log1p(np.abs(x))
            probs = model(x[...,None],training=False).numpy()
            if probs.shape != (len(x),len(self.labels)) or not np.isfinite(probs).all():
                raise RuntimeError('Invalid model output.')
            return probs

    def raw_vector(self, features):
        if not isinstance(features,dict): raise ValueError('features must be an object containing the complete flow schema.')
        missing = [n for n in self.features if n not in features]
        unknown = [n for n in features if n not in self.features]
        if missing: raise ValueError('Missing required features: '+', '.join(missing))
        if unknown: raise ValueError('Unknown features: '+', '.join(unknown))
        try: x = np.asarray([[float(features[n]) for n in self.features]],dtype='float64')
        except (ValueError,TypeError): raise ValueError('All flow features must be numeric.') from None
        if not np.isfinite(x).all(): raise ValueError('All flow features must be finite numbers.')
        scaled = self.scaler.transform(x).astype('float32')
        if not np.isfinite(scaled).all(): raise ValueError('Feature values exceed the supported numeric range.')
        return scaled

    def verdict(self, probs, key, latency):
        pred = int(np.argmax(probs)); label=self.labels[pred]; confidence=float(probs[pred])
        attack = label != 'BenignTraffic'
        return {'predicted_label':label,'predicted_class_id':pred,'is_attack':attack,'confidence':round(confidence*100,2),
                'severity':'REVIEW' if confidence < .6 else ('ALERT' if attack else 'BENIGN'),
                'model':key,'latency_ms':round(latency,3),
                'top_classes':[{'label':self.labels[int(i)],'prob':round(float(probs[i])*100,2)} for i in np.argsort(probs)[::-1][:3]]}

    def classify(self, data):
        key=data.get('model','cnn_bilstm')
        x=self.raw_vector(data.get('features'))
        start=time.perf_counter();probs=self.probabilities(x,key)[0]
        latency=(time.perf_counter()-start)*1000
        result=self.verdict(probs,key,latency)
        result['source']='submitted_flow'
        return result

    def replay(self, count=10, key='cnn_bilstm'):
        if not 1 <= count <= 50: raise ValueError('count must be between 1 and 50.')
        self.load_pool()
        with self.lock: ids=self.rng.choice(len(self.pool),min(count,len(self.pool)),replace=False)
        start=time.perf_counter(); probs=self.probabilities(self.pool[ids],key)
        latency=(time.perf_counter()-start)*1000/len(ids);self.last_latency=latency
        events=[]
        for i,p in zip(ids,probs):
            event=self.verdict(p,key,latency)
            event.update({'event_id':int(self.indices[i]),'timestamp':time.strftime('%H:%M:%S'),
                          'true_label':self.labels[int(self.targets[i])],
                          'is_correct':int(np.argmax(p))==int(self.targets[i]),'source':'held_out_replay'})
            events.append(event)
        return events

    def presets(self):
        self.load_pool()
        names=['BenignTraffic','DDoS-ICMP_Flood','Mirai-greeth_flood','Recon-PortScan']
        result=[]
        for name in names:
            ids=np.flatnonzero(self.targets==self.mapping[name])
            if not len(ids): continue
            i=ids[0]; raw=self.scaler.inverse_transform(self.pool[i:i+1])[0]
            result.append({'name':name,'description':'Actual held-out flow; the label is ground truth, not a guaranteed prediction.',
                           'sample_id':int(self.indices[i]),'features':{n:float(v) for n,v in zip(self.features,raw)}})
        return result

    def explain(self,data):
        import shap
        key=data.get('model','cnn_bilstm')
        x=self.raw_vector(data.get('features'))
        p=self.probabilities(x,key)[0]; target=int(np.argmax(p))
        with self.lock:
            if self.explainer_background is None:
                path=self.root/'output/delivery/shap_background.npy'
                if path.exists(): bg=np.load(path)
                else:
                    with np.load(self.root/'data/processed/train/train.npz') as d:
                        train=d['X']; ids=np.random.default_rng(42).choice(len(train),size=16,replace=False);bg=train[ids]
                self.explainer_background=bg[:16]
            def predict_class(rows):
                return np.concatenate([self.probabilities(rows[i:i+512],key)[:,target] for i in range(0,len(rows),512)])
            exp=shap.Explainer(predict_class,self.explainer_background,algorithm='permutation',feature_names=self.features,seed=42)
            values=exp(x,max_evals=2*len(self.features)+1)
        vals=np.asarray(values.values[0]);base=float(np.asarray(values.base_values).flatten()[0])
        return {'model':key,'label':self.labels[target],'probability':float(p[target]),'base_value':base,
                'method':'SHAP permutation (one permutation; approximate, 16 training background rows)',
                'additivity_residual':float(p[target]-base-vals.sum()),
                'features':[{'feature':self.features[int(i)],'contribution':float(vals[i]),'value':float(data['features'][self.features[int(i)]])} for i in np.argsort(np.abs(vals))[::-1]],
                'note':'Attribution explains the model output, not proof of an attack.'}

    def stats(self):
        self.load_pool()
        delivery=read_json(self.root/'models/deployed/metadata.json',{})
        metrics=read_json(self.root/'output/delivery/evaluation.json',{}) if delivery else read_json(self.root/'results/metrics/cnn_bilstm_test.json',{})
        return {'dataset':'CICIoT2023','feature_count':len(self.features),'class_count':len(self.labels),
                'test_samples':self.test_count,'train_samples':self.metadata.get('split_metadata',{}).get('n_train'),
                'active_model':'CNN-BiLSTM','model_ready':'cnn_bilstm' in self.models,
                'accuracy':metrics.get('accuracy'),'macro_f1':metrics.get('macro_f1'),
                'mean_latency_ms':self.last_latency,'representation':'46 ordered flow features',
                'source':'held_out_replay','training':read_json(self.root/'output/delivery/training_status.json',{}),
                'trained_packet_model':(self.root/'models/packet_bytes/metadata.json').exists(),
                'errors':self.errors}
