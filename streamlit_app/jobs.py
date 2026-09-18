"""One local preparation/training process at a time; commands never use a shell."""
from pathlib import Path
import subprocess
import sys
import threading
import time


class JobManager:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.process = None
        self.log = None
        self.name = None
        self.started = None
        self.handle = None

    def start(self, name, arguments):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise ValueError('Another data preparation or training job is still running.')
            if self.handle:
                self.handle.close()
            runtime = self.root / 'runtime'
            runtime.mkdir(exist_ok=True)
            self.log = runtime / 'packet-training.log'
            self.handle = self.log.open('w')
            self.name, self.started = name, time.time()
            self.process = subprocess.Popen([sys.executable, str(self.root / 'scripts/packet_pipeline.py'), *arguments],
                                            cwd=self.root, stdout=self.handle, stderr=subprocess.STDOUT)

    def status(self):
        with self.lock:
            code = self.process.poll() if self.process else None
            if self.process is not None and code is not None and self.handle and not self.handle.closed:
                self.handle.close()
            # Read a bounded tail without loading a long training log into memory.
            tail = ''
            if self.log and self.log.exists():
                with self.log.open('rb') as stream:
                    stream.seek(max(0, self.log.stat().st_size - 12000))
                    tail = stream.read().decode('utf-8', errors='replace')
            return {'name': self.name, 'running': self.process is not None and code is None,
                    'exit_code': code, 'log_tail': tail}
