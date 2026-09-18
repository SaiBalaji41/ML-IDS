"""Replay, PCAP payload inference, and Kafka ingestion using the same IDS engines."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.realtime.engine import IDSEngine
from src.realtime.capture import CaptureService


def process_message(value,engine,capture):
    if not isinstance(value,dict):raise ValueError('Message must be an object.')
    representation=value.get('representation')
    if representation=='flow_features':return engine.classify({'features':value.get('features'),'model':value.get('model','cnn_bilstm')})
    if representation=='raw_payload_bytes':return capture.classify_hex(value.get('payload_hex'))
    raise ValueError('representation must be flow_features or raw_payload_bytes.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['replay','kafka','pcap'],default='replay');p.add_argument('--samples',type=int,default=20)
    p.add_argument('--model',default='cnn_bilstm',choices=['cnn_bilstm','xgboost']);p.add_argument('--broker',default='localhost:9092')
    p.add_argument('--topic',default='ids-flows');p.add_argument('--pcap',type=Path);p.add_argument('--output',type=Path,default=ROOT/'runtime/events.jsonl')
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    engine=IDSEngine(ROOT);capture=CaptureService(ROOT)
    def emit(event):
        value={'observed_at':time.time(),**event};line=json.dumps(value,allow_nan=False)
        print(line,flush=True)
        with args.output.open('a') as f:f.write(line+'\n')
    if args.mode=='replay':
        if args.samples<1:p.error('samples must be positive')
        remaining=args.samples
        while remaining:
            count=min(50,remaining)
            for event in engine.replay(count,args.model):emit(event)
            remaining-=count
    elif args.mode=='pcap':
        if not args.pcap:p.error('--pcap is required for PCAP inference')
        if not (ROOT/'models/packet_bytes/metadata.json').exists():p.error('Train a packet-byte model using labeled PCAPs before PCAP inference.')
        from scapy.all import PcapReader,Raw
        with PcapReader(str(args.pcap)) as packets:
            for i,packet in enumerate(packets):
                if Raw in packet:emit({'packet_index':i,**capture.classify_payload(bytes(packet[Raw].load))})
    else:
        from kafka import KafkaConsumer
        consumer=KafkaConsumer(args.topic,bootstrap_servers=args.broker,group_id='ml-ids-detector',enable_auto_commit=False,
                               auto_offset_reset='latest',max_poll_records=50,max_partition_fetch_bytes=1048576)
        print(f'Connected to {args.broker}, listening on {args.topic}',file=sys.stderr,flush=True)
        try:
            for msg in consumer:
                try:
                    value=json.loads(msg.value.decode('utf-8'));result=process_message(value,engine,capture)
                    emit({**result,'source':'kafka','topic':msg.topic,'partition':msg.partition,'offset':msg.offset})
                except (ValueError,TypeError) as e:
                    emit({'status':'rejected','error':str(e),'topic':msg.topic,'partition':msg.partition,'offset':msg.offset})
                # Commit only after inference or validation rejection is persisted.
                consumer.commit()
        except KeyboardInterrupt:pass
        finally:consumer.close()

if __name__=='__main__':main()
