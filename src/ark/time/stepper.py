import threading
from typing import Callable
from ark.time.rate import Rate
from ark.time.clock import Clock


class Stepper(threading.Thread):

    def __init__(
        self,
        clock: Clock,
        hz: float,
        callback: Callable[[int], None],
    ):
        super().__init__(daemon=True)
        self._clock = clock
        self._rate = Rate(clock, hz)
        self.reset = self._rate.reset
        self._callback = callback
        self._closed = False
        self.start()

    def run(self) -> None:
        while not self._closed:
            self._rate.sleep()
            if self._closed:
                break
            self._callback(self._clock.now())

    def close(self) -> None:
        self._closed = True
        self._clock.notify()
