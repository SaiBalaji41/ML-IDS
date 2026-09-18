"""Bounded passive capture with a bounded queue and independent inference worker."""
from collections import deque
from pathlib import Path
from queue import Queue, Empty, Full
import threading
import time
import uuid
from packet_ids.events import packet_event


class CaptureMonitor:
    def __init__(self):
        self.lock = threading.RLock()
        self.events = deque(maxlen=500)
        self.queue = Queue(maxsize=256)
        self.done = threading.Event()
        self.stop_requested = threading.Event()
        self.ready = threading.Event()
        self.sniffer = self.worker = self.watcher = None
        self.error = self.interface = self.pcap_path = None
        self.observed = self.processed = self.dropped = 0
        self.run_id = None
        self.engine = None
        self.writer = None

    @staticmethod
    def interfaces():
        from scapy.all import get_if_list
        return get_if_list()

    def start(self, interface, engine=None, seconds=30, packet_limit=1000, save_directory=None):
        from scapy.all import AsyncSniffer, PcapWriter
        if interface not in self.interfaces():
            raise ValueError('Select an available network interface.')
        if not 1 <= seconds <= 120 or not 1 <= packet_limit <= 10000:
            raise ValueError('Capture bounds are 1–120 seconds and 1–10000 packets.')
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                raise ValueError('A capture is already running.')
            self.events.clear()
            self.queue = Queue(maxsize=256)
            self.done.clear(); self.stop_requested.clear(); self.ready.clear()
            self.observed = self.processed = self.dropped = 0
            self.error = None
            self.interface, self.engine = interface, engine
            self.run_id = uuid.uuid4().hex[:12]
            self.pcap_path = None
            self.writer = None
            if save_directory is not None:
                directory = Path(save_directory)
                directory.mkdir(parents=True, exist_ok=True)
                self.pcap_path = str(directory / f'capture-{self.run_id}.pcap')
                self.writer = PcapWriter(self.pcap_path, sync=True)
            self.sniffer = AsyncSniffer(iface=interface, store=False, promisc=False,
                count=packet_limit, timeout=seconds, prn=self._enqueue, started_callback=self.ready.set)
            self.worker = threading.Thread(target=self._consume, daemon=True, name='packet-ids-inference')
            self.watcher = threading.Thread(target=self._watch, daemon=True, name='packet-ids-capture')
            self.worker.start()
            try:
                self.sniffer.start()
                self.watcher.start()
            except Exception:
                self.done.set()
                if self.writer:
                    self.writer.close()
                raise
        return self.snapshot()

    def _enqueue(self, packet):
        with self.lock:
            self.observed += 1
            identifier = f'{self.run_id}:{self.observed}'
            if self.writer:
                self.writer.write(packet)
        try:
            self.queue.put_nowait((identifier, packet))
        except Full:
            with self.lock:
                self.dropped += 1

    def _consume(self):
        while not self.done.is_set() or not self.queue.empty():
            try:
                identifier, packet = self.queue.get(timeout=.1)
            except Empty:
                continue
            try:
                event = packet_event(packet, self.engine, identifier)
                with self.lock:
                    self.events.append(event)
                    self.processed += 1
            except Exception as exc:
                with self.lock:
                    self.error = f'Packet processing failed: {exc}'
            finally:
                self.queue.task_done()

    def _watch(self):
        try:
            while self.sniffer.thread.is_alive():
                self.sniffer.thread.join(timeout=.05)
                if self.stop_requested.is_set() and self.ready.is_set() and self.sniffer.running:
                    self.sniffer.stop(join=False)
                    break
            self.sniffer.join()
        except Exception as exc:
            with self.lock:
                self.error = f'Capture failed: {exc}. The OS may require packet-capture privileges.'
        finally:
            with self.lock:
                if self.writer:
                    self.writer.close()
            self.done.set()

    def stop(self):
        self.stop_requested.set()
        return self.snapshot()

    def snapshot(self):
        with self.lock:
            return {'running': bool(self.worker and self.worker.is_alive()),
                    'interface': self.interface, 'observed': self.observed,
                    'processed': self.processed, 'dropped': self.dropped, 'queued': self.queue.qsize(),
                    'error': self.error, 'events': list(self.events), 'run_id': self.run_id,
                    'pcap_path': self.pcap_path, 'model_ready': self.engine is not None}
