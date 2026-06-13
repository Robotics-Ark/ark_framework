import threading
from collections import deque
import zenoh
from gymnasium import Space
from ark.time import Time
from .subscriber import Subscriber
from .channel import Channel, ChannelNoise
from .stamped_sample import StampedSample


class NSampleListener(Subscriber):

    def __init__(
        self,
        n: int,
        channel: Channel,
        space: Space,
        session: zenoh.Session,
        check: bool,
        noise: ChannelNoise | None,
    ):
        self._lock = threading.Lock()
        self._window: deque[StampedSample] = deque(maxlen=n)
        super().__init__(channel, space, self._on_sample, session, check, noise)

    def _on_sample(self, stamped_sample: StampedSample) -> None:
        with self._lock:
            self._window.append(stamped_sample)

    def get_window(self) -> list[StampedSample]:
        with self._lock:
            return list(self._window)


class TSampleListener(Subscriber):

    def __init__(
        self,
        t: float,
        channel: Channel,
        space: Space,
        session: zenoh.Session,
        check: bool,
        noise: ChannelNoise | None,
    ):
        self._lock = threading.Lock()
        self._duration = Time.from_sec(t)
        self._window: deque[StampedSample] = deque()
        super().__init__(channel, space, self._on_sample, session, check, noise)

    def _on_sample(self, stamped_sample: StampedSample) -> None:
        cutoff = stamped_sample.time - self._duration
        with self._lock:
            self._window.append(stamped_sample)
            while self._window and self._window[0].time < cutoff:
                self._window.popleft()

    def get_window(self) -> list[StampedSample]:
        cutoff = self._clock.now() - self._duration
        with self._lock:
            while self._window and self._window[0].time < cutoff:
                self._window.popleft()
            return list(self._window)
