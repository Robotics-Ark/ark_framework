import struct
from pathlib import Path
from datetime import datetime
from threading import Thread
from queue import Queue
from ark.core.registerable import Registerable


class DataCollector(Registerable):
    __slots__ = ("_path", "_file_path", "_queue", "_thread")

    _SENTINEL = object()

    def __init__(self, node_name: str, queue_maxsize: int = 1000):
        self._path = Path.home() / ".ark" / "data" / node_name
        self._path.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._file_path = self._path / f"data_{stamp}.bin"

        self._queue: Queue[object] = Queue(maxsize=queue_maxsize)
        self.append = self._queue.put

        self.core_registration()

        self._thread = Thread(target=self._save_data, daemon=True)
        self._thread.start()

    def core_registration(self):
        print(".. todo: register data collector with ark core..")

    def _save_data(self):
        with open(self._file_path, "ab", buffering=1024 * 1024) as f:
            while True:
                b = self._queue.get()  # serialized message
                if b is self._SENTINEL:
                    self._queue.task_done()
                    break

                f.write(struct.pack("<I", len(b)))
                f.write(b)
                self._queue.task_done()

    def close(self):
        self._queue.put(self._SENTINEL)
        self._thread.join()
