"""Bounded Scapy capture. Raw payloads never enter the flow-feature model."""
from collections import deque
import json
from pathlib import Path
import threading
import time
import numpy as np
from src.preprocessing.packet_bytes import payload_to_array, parse_hex


class CaptureService:
    def __init__(self,root):
        self.root=Path(root);self.events=deque(maxlen=200);self.lock=threading.RLock()
        self.sniffer=None;self.error=None;self.interface=None;self.model=None;self.metadata=None
        self.packet_count=0;self.started_at=None

    def classify_hex(self,value):return self.classify_payload(parse_hex(value))

    def classify_payload(self,payload):
        meta=self.root/'models/packet_bytes/metadata.json'
        if not meta.exists():raise RuntimeError('Packet-byte detection needs a trained byte-model checkpoint. Available checkpoints expect 46 flow features. Train with scripts/train_packet_bytes.py and a labeled PCAP manifest.')
        if not payload:return {'status':'unclassified','reason':'No packet payload'}
        with self.lock:
            if self.model is None:
                import tensorflow as tf
                self.metadata=json.loads(meta.read_text())
                self.model=tf.keras.models.load_model(meta.parent/'best_model.keras',compile=False)
            x=payload_to_array(payload,self.metadata['byte_length'])[None,:,None]
            probs=self.model(x,training=False).numpy()[0]
            labels=self.metadata['labels'];i=int(np.argmax(probs))
        return {'status':'classified','predicted_label':labels[i],'confidence':round(float(probs[i])*100,2),
                'is_attack':labels[i]!=self.metadata['benign_label'],'source':'captured_payload','model':'packet_bytes'}

    def _on_packet(self,packet):
        from scapy.all import IP, IPv6, TCP, UDP, Raw
        ip=packet.getlayer(IP) or packet.getlayer(IPv6)
        if ip is None:return
        event={'timestamp':time.strftime('%H:%M:%S'),'src':str(ip.src),'dst':str(ip.dst),'length':len(packet),
               'protocol':'TCP' if TCP in packet else ('UDP' if UDP in packet else str(getattr(ip,'proto','IP'))),
               'status':'observed','source':'scapy_live','predicted_label':None}
        if (self.root/'models/packet_bytes/metadata.json').exists():
            if Raw in packet:
                try:event.update(self.classify_payload(bytes(packet[Raw].load)))
                except Exception as e:event['error']=str(e)
            else:event['status']='no_payload'
        with self.lock:self.packet_count+=1;self.events.appendleft(event)

    def start(self,interface=None):
        from scapy.all import AsyncSniffer, get_if_list
        if interface and interface not in get_if_list():raise ValueError('Unknown network interface.')
        if self.sniffer and self.sniffer.running:raise ValueError('Capture is already running.')
        self.error=None;self.interface=interface;self.started_at=time.time()
        self.sniffer=AsyncSniffer(iface=interface,prn=self._on_packet,store=False,count=1000,timeout=60)
        self.sniffer.start()
        # Scapy initializes the capture socket in its own thread.
        time.sleep(.2)
        if getattr(self.sniffer,'exception',None):
            self.error=str(self.sniffer.exception)
            raise RuntimeError('Capture could not start: '+self.error)
        return self.status()

    def stop(self):
        if self.sniffer and self.sniffer.running:
            try:self.sniffer.stop()
            except Exception as e:self.error=str(e)
        return self.status()

    def status(self):
        from scapy.all import get_if_list
        if self.sniffer and getattr(self.sniffer,'exception',None):self.error=str(self.sniffer.exception)
        with self.lock:
            return {'running':bool(self.sniffer and self.sniffer.running),'interface':self.interface,
                    'interfaces':get_if_list(),'packet_count':self.packet_count,'events':list(self.events),
                    'error':self.error,'limit':'60 seconds or 1,000 packets per capture',
                    'detection_ready':(self.root/'models/packet_bytes/metadata.json').exists(),
                    'note':'Without a trained packet-byte model, capture shows observed metadata only.'}
