from .clock import Clock
from .sleeper import Sleeper


class Rate:

    __slots__ = ("_clock", "_sleep", "_time_step", "_next")

    def __init__(self, clock: Clock, hz: float):
        self._clock = clock
        self._sleep = Sleeper(clock)
        self._time_step = int(1e9 / hz)  # Convert hz to nanoseconds period
        self.reset()

    def reset(self) -> None:
        self._next = self._clock.now() + self._time_step

    def sleep(self):
        now = self._clock.now()
        remaining = self._next - now
        if remaining > 0:
            self._sleep(remaining)
            self._next += self._time_step
        else:
            # We are late, skip to next period
            self._next = now + self._time_step
