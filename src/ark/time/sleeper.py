from time import sleep
from ark.time.clock import Clock


def _wall_sleep(dur: int) -> None:
    if dur <= 0:
        return
    sleep(dur / 1e9)  # ns -> s


class Sleeper:
    __slots__ = ("_clock", "_sleep")

    def __init__(self, clock: Clock):
        self._clock = clock
        self._sleep = self._sim_sleep if clock._sim else _wall_sleep

    def _sim_sleep(self, dur: int) -> None:
        if dur <= 0:
            return
        now = self._clock.now()
        self._clock.wait_until(now + dur)

    def __call__(self, dur: int) -> None:
        self._sleep(dur)
