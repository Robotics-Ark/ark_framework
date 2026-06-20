import signal
import threading


class Spinner:
    """A class that provides a spin method to wait for termination signals."""

    def __init__(self):
        self._stop_event = threading.Event()

    def spin(self):
        signal.signal(signal.SIGINT, lambda *_: self._stop_event.set())
        signal.signal(signal.SIGTERM, lambda *_: self._stop_event.set())
        self._stop_event.wait()
