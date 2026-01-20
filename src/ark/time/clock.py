import struct
import threading
import zenoh
from time import time_ns


class Clock:
    __slots__ = ("_sim", "_t", "_started", "_sim_time_cv", "_sub", "now")

    _FMT = "<q"
    _SIZE = 8

    def __init__(
        self, session: zenoh.Session, sim: bool, clock_channel_name: str | None = None
    ):
        self._sim = sim
        self._t = 0
        self._started = False
        self._sub = None

        if self._sim:
            if not clock_channel_name:
                raise ValueError("clock_channel_name must be provided when sim=True")
            self._sim_time_cv = threading.Condition()
            self._sub = session.declare_subscriber(
                clock_channel_name, self._on_clock_sample
            )
            self.now = self._sim_now
        else:
            self._sim_time_cv = None
            self.now = time_ns

    def _on_clock_sample(self, sample: zenoh.Sample):
        payload = bytes(sample.payload)
        if len(payload) != self._SIZE:
            return
        t = struct.unpack(self._FMT, payload)[0]

        with self._sim_time_cv:
            self._t = t
            self._started = True
            self._sim_time_cv.notify_all()

    def _sim_now(self) -> int:
        with self._sim_time_cv:
            while not self._started:
                self._sim_time_cv.wait()
            return self._t

    def wait_until(self, target: int) -> None:
        with self._sim_time_cv:
            while not self._started:
                self._sim_time_cv.wait()
            while self._t < target:
                self._sim_time_cv.wait()

    def notify(self) -> None:
        if not self._sim:
            return
        with self._sim_time_cv:
            self._sim_time_cv.notify_all()
