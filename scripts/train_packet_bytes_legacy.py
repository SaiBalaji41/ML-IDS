"""Train a separate raw-payload CNN-BiLSTM from a verified labeled PCAP manifest.

Manifest: path,label,split. All packets in a capture must share its verified label.
Separate source captures into train/validation/test BEFORE creating the manifest.
No flow CSV is transformed into pretend packet bytes.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('MPLCONFIGDIR','/tmp/ml-ids-matplotlib')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.preprocessing.packet_bytes import payload_to_array


def load_manifest(path,byte_length=256,max_per_capture=20000):
    from scapy.all import PcapReader, Raw
    path=Path(path).resolve();splits={s:[] for s in ['train','validation','test']}
    with path.open() as f:
        reader=csv.DictReader(f)
        if not {'path','label','split'} <= set(reader.fieldnames or []):raise ValueError('Manifest needs path,label,split columns.')
        rows=list(reader)
    hashes=set();provenance=[]
    for row in rows:
        if row['split'] not in splits:raise ValueError('split must be train, validation or test.')
        if not row['label'].strip():raise ValueError('Every capture requires a verified label.')
        capture=(path.parent/row['path']).resolve()
        digest=hashlib.file_digest(capture.open('rb'),'sha256').hexdigest()
        if digest in hashes:raise ValueError('Duplicate capture detected. Source captures must be disjoint.')
        hashes.add(digest);count=0
        with PcapReader(str(capture)) as packets:
            for packet in packets:
                if Raw not in packet or not bytes(packet[Raw].load):continue
                splits[row['split']].append((payload_to_array(bytes(packet[Raw].load),byte_length),row['label']))
                count+=1
                if count>=max_per_capture:break
        provenance.append({'path':str(capture),'label':row['label'],'split':row['split'],'sha256':digest,'payloads_used':count})
    if any(not values for values in splits.values()):raise ValueError('Each split needs captures with nonempty raw payloads.')
    labels=sorted({label for _,label in splits['train']})
    if len(labels)<2:raise ValueError('Training requires at least two labels including benign traffic.')
    for name,values in splits.items():
        present={label for _,label in values}
        if present!=set(labels):raise ValueError(f'{name} must contain all and only training labels; got {sorted(present)}.')
    return {s:(np.asarray([x for x,_ in values])[...,None],np.asarray([labels.index(label) for _,label in values])) for s,values in splits.items()},labels,provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--benign-label',default='BenignTraffic')
    p.add_argument('--byte-length',type=int,default=256);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--max-per-capture',type=int,default=20000)
    args=p.parse_args()
    if args.byte_length<8 or args.epochs<1 or args.max_per_capture<1:p.error('Byte length >=8; epochs and sample limit >=1 required.')
    data,labels,provenance=load_manifest(args.manifest,args.byte_length,args.max_per_capture)
    if args.benign_label not in labels:raise ValueError('benign-label must match an actual training label.')
    import tensorflow as tf
    from sklearn.metrics import classification_report,confusion_matrix
    tf.keras.utils.set_random_seed(42)
    model=tf.keras.Sequential([tf.keras.Input((args.byte_length,1)),tf.keras.layers.Conv1D(32,5,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling1D(2),tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(len(labels),activation='softmax')],name='Packet_Bytes_CNN_BiLSTM')
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    out=ROOT/'models/packet_bytes';out.mkdir(parents=True,exist_ok=True)
    history=model.fit(*data['train'],validation_data=data['validation'],epochs=args.epochs,batch_size=128,callbacks=[
        tf.keras.callbacks.ModelCheckpoint(str(out/'best_model.keras'),save_best_only=True,monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=True)])
    model=tf.keras.models.load_model(out/'best_model.keras',compile=False)
    x,y=data['test'];pred=np.argmax(model.predict(x,batch_size=512,verbose=0),axis=1)
    meta={'representation':'raw_payload_bytes','normalization':'uint8 / 255','padding':'right-zero','truncation':'first N bytes',
        'byte_length':args.byte_length,'labels':labels,'benign_label':args.benign_label,'seed':42,'epochs_completed':len(history.history['loss']),
        'samples':{s:len(v[1]) for s,v in data.items()},'capture_provenance':provenance,
        'test_report':classification_report(y,pred,target_names=labels,output_dict=True,zero_division=0),
        'confusion_matrix':confusion_matrix(y,pred,labels=np.arange(len(labels))).tolist(),
        'checkpoint_sha256':hashlib.sha256((out/'best_model.keras').read_bytes()).hexdigest()}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    (out/'training_history.json').write_text(json.dumps(history.history,indent=2))
    print(json.dumps(meta['test_report'],indent=2))

if __name__=='__main__':main()
