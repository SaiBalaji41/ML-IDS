"""Train and verify a deployable CNN-BiLSTM on the available CICIoT2023 flows.

Original research artifacts are preserved. New evidence lives in output/delivery.
The test set is evaluated once after validation-based checkpoint selection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ml-ids-matplotlib')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

OUT = ROOT / 'output/delivery'
MODEL = ROOT / 'models/deployed'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def load_split(split, file):
    with np.load(ROOT / f'data/processed/{split}/{file}.npz') as d:
        return d['X'].astype('float32'), d['y'].astype('int32')


def select(x, y, count, seed):
    if count and len(y) > count:
        ids, _ = train_test_split(np.arange(len(y)), train_size=count, stratify=y, random_state=seed)
        return x[ids], y[ids]
    return x, y


def transform(x):
    # Signed log compression preserves order while controlling heavy-tailed values.
    return (np.sign(x) * np.log1p(np.abs(x))).astype('float32')[..., None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--train-samples', type=int, default=200000)
    p.add_argument('--validation-samples', type=int, default=40000)
    p.add_argument('--batch-size', type=int, default=512)
    args = p.parse_args()
    if args.epochs < 1 or args.train_samples < 0 or args.validation_samples < 0:
        p.error('Epochs must be positive and sample limits nonnegative (0 means all).')
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    mapping = json.loads((ROOT / 'data/processed/label_mapping.json').read_text())
    labels = [k for k,v in sorted(mapping.items(), key=lambda kv:kv[1])]
    meta = json.loads((ROOT / 'data/processed/preprocessing_metadata.json').read_text())
    start = time.time()
    write_json(OUT / 'training_status.json', {'state':'loading','started_at':start,'max_epochs':args.epochs})
    x,y = load_split('train','train')
    total_train = len(y)
    x,y = select(x,y,args.train_samples,42)
    xv,yv = load_split('validation','val')
    total_val = len(yv)
    xv,yv = select(xv,yv,args.validation_samples,43)
    x,xv = transform(x),transform(xv)
    print(f'Train: {len(y):,}/{total_train:,}; validation: {len(yv):,}/{total_val:,}',flush=True)
    inp = tf.keras.Input((x.shape[1],1), name='ordered_flow_features')
    z = tf.keras.layers.Conv1D(48,3,padding='same',activation='relu')(inp)
    z = tf.keras.layers.Conv1D(48,3,padding='same',activation='relu')(z)
    z = tf.keras.layers.MaxPooling1D(2)(z)
    z = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(48))(z)
    z = tf.keras.layers.Dense(96,activation='relu')(z)
    z = tf.keras.layers.Dropout(.15)(z)
    out = tf.keras.layers.Dense(len(labels),activation='softmax')(z)
    model = tf.keras.Model(inp,out,name='CICIoT2023_CNN_BiLSTM')
    model.compile(optimizer=tf.keras.optimizers.Adam(.001,clipnorm=1.0),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    counts = np.bincount(y,minlength=len(labels))
    weights = np.clip(np.sqrt(len(y)/(len(labels)*np.maximum(counts,1))),.5,8.)
    history = []
    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            row = {'epoch':epoch+1, **{k:float(v) for k,v in logs.items()}}
            history.append(row)
            write_json(OUT/'training_history.json',history)
            write_json(OUT/'training_status.json',{'state':'training','epoch':epoch+1,'max_epochs':args.epochs,'train_samples':len(y),'validation_samples':len(yv),'elapsed_seconds':round(time.time()-start,2),'latest':row})
            print(json.dumps(row),flush=True)
    best = MODEL/'best_model.keras'
    model.fit(x,y,validation_data=(xv,yv),epochs=args.epochs,batch_size=args.batch_size,verbose=0,
        class_weight={i:float(v) for i,v in enumerate(weights)},callbacks=[
            tf.keras.callbacks.ModelCheckpoint(str(best),monitor='val_loss',save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=6,restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=2,factor=.5,min_lr=1e-5),Progress()])
    train_seconds = time.time()-start
    del x,xv
    model = tf.keras.models.load_model(best,compile=False)
    write_json(OUT/'training_status.json',{'state':'evaluating','epoch':len(history),'max_epochs':args.epochs,'train_samples':len(y),'elapsed_seconds':train_seconds})
    xt,yt = load_split('test','test')
    pred = np.empty(len(yt),dtype='int32')
    t = time.perf_counter()
    for offset in range(0,len(yt),2048):
        pred[offset:offset+2048] = np.argmax(model(transform(xt[offset:offset+2048]),training=False).numpy(),axis=1)
        if offset % 204800 == 0: print(f'Evaluated {offset:,}/{len(yt):,}',flush=True)
    elapsed = time.perf_counter()-t
    precision,recall,f1,_ = precision_recall_fscore_support(yt,pred,labels=np.arange(len(labels)),average='macro',zero_division=0)
    cm = confusion_matrix(yt,pred,labels=np.arange(len(labels)))
    benign = mapping['BenignTraffic']
    binary_cm = confusion_matrix(yt != benign,pred != benign,labels=[False,True])
    bp,br,bf,_ = precision_recall_fscore_support(yt != benign,pred != benign,average='binary',zero_division=0)
    metrics = {'accuracy':float(accuracy_score(yt,pred)),'macro_precision':float(precision),'macro_recall':float(recall),'macro_f1':float(f1),'sample_count':len(yt),'inference_time_seconds':elapsed,'binary_precision':float(bp),'binary_recall':float(br),'binary_f1':float(bf),'binary_confusion_matrix':binary_cm.tolist(),'labels':labels,'confusion_matrix':cm.tolist(),'classwise':classification_report(yt,pred,labels=np.arange(len(labels)),target_names=labels,output_dict=True,zero_division=0)}
    write_json(OUT/'evaluation.json',metrics)
    metadata = {'model_name':'CNN-BiLSTM','representation':'flow_features','input_features':meta['feature_names'],'input_transform':'signed_log1p_standardized','label_mapping':mapping,'seed':42,'training_samples':len(y),'available_training_samples':total_train,'validation_samples':len(yv),'available_validation_samples':total_val,'test_samples':len(yt),'epochs_completed':len(history),'best_epoch':int(np.argmin([v['val_loss'] for v in history])+1),'training_seconds':train_seconds,'parameter_count':model.count_params(),'checkpoint_sha256':hashlib.sha256(best.read_bytes()).hexdigest(),'tensorflow_version':tf.__version__,'metrics_path':'output/delivery/evaluation.json','note':'Trained on CICIoT2023 tabular flow features, not raw packet payload bytes. Historical partitions supplied with project; source-level split independence has not been audited.'}
    write_json(MODEL/'metadata.json',metadata)
    # Small, genuine training reference for local SHAP explanations; never test data.
    xb,_ = load_split('train','train')
    ids = np.random.default_rng(42).choice(len(xb),size=min(32,len(xb)),replace=False)
    np.save(OUT/'shap_background.npy',xb[ids])
    write_json(OUT/'training_status.json',{'state':'complete',**metadata,'accuracy':metrics['accuracy'],'macro_f1':metrics['macro_f1']})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(1,2,figsize=(12,4))
    for ax,key,title in zip(axs,['loss','accuracy'],['Loss','Accuracy']):
        ax.plot([v['epoch'] for v in history],[v[key] for v in history],label='Training',color='#147d64')
        ax.plot([v['epoch'] for v in history],[v['val_'+key] for v in history],label='Validation',color='#7689ac')
        ax.set(title=title,xlabel='Epoch');ax.legend();ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4))
    ax.imshow(binary_cm,cmap='Greens')
    for (i,j),v in np.ndenumerate(binary_cm):ax.text(j,i,f'{v:,}',ha='center',va='center',color='white' if v>binary_cm.max()/2 else '#233c32')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Benign','Attack'],yticklabels=['Benign','Attack'],xlabel='Predicted',ylabel='Actual',title='Held-out intrusion detection')
    fig.tight_layout();fig.savefig(OUT/'intrusion_confusion_matrix.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in metrics.items() if k not in ['classwise','confusion_matrix','labels']},indent=2),flush=True)

if __name__=='__main__':
    try: main()
    except Exception as e:
        write_json(OUT/'training_status.json',{'state':'failed','error':str(e)})
        raise
