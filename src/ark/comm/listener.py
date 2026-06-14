import zenoh
import threading
from enum import IntEnum
from collections import deque
from gymnasium import Space
from ark.time import Time
from abc import abstractmethod
from .subscriber import Subscriber
from .channel import Channel, NOISE_TYPE
from .stamped_sample import StampedSample


class ReadyWhen(IntEnum):
    """Enum for specifying when a listener is considered ready."""

    # The listener is always ready, regardless of received samples.
    ALWAYS = 0

    # The listener is ready after receiving the first sample.
    FIRST_SAMPLE = 1

    # The listener is ready when the window is full (not applicable for time-based listeners).
    WINDOW_FULL = 2


is_ready = {
    ReadyWhen.ALWAYS: lambda window: True,
    ReadyWhen.FIRST_SAMPLE: lambda window: len(window) > 0,
    ReadyWhen.WINDOW_FULL: lambda window: len(window) == window.maxlen,
}


class Listener(Subscriber):

    def __init__(
        self,
        window_length: int | None,
        channel: Channel,
        space: Space,
        session: zenoh.Session,
        check: bool,
        noise: NOISE_TYPE,
        ready_when: ReadyWhen,
    ):
        self._window: deque[StampedSample] = deque(maxlen=window_length)
        self._lock = threading.Lock()
        self._ready_when = ready_when
        self._is_ready = is_ready[ready_when]
        super().__init__(channel, space, self._on_sample, session, check, noise)

    @abstractmethod
    def _on_sample(self, stamped_sample: StampedSample) -> None:
        """Callback function to handle incoming samples."""

    def is_ready(self) -> bool:
        with self._lock:
            return self._is_ready(self._window)

    @abstractmethod
    def get_window(self) -> tuple[StampedSample]:
        """Returns the current window of samples."""


class NSampleListener(Listener):

    def _on_sample(self, stamped_sample: StampedSample) -> None:
        with self._lock:
            self._window.append(stamped_sample)

    def get_window(self) -> tuple[StampedSample]:
        with self._lock:
            return tuple(self._window)


class TSampleListener(Listener):

    def __init__(
        self,
        t: float,
        channel: Channel,
        space: Space,
        session: zenoh.Session,
        check: bool,
        noise: NOISE_TYPE,
        ready_when: ReadyWhen,
    ):
        self._window_duration = Time.from_sec(t)
        if ready_when == ReadyWhen.WINDOW_FULL:
            raise ValueError("WINDOW_FULL is not applicable for time-based listeners")
        super().__init__(
            None,  # window_length is not used for time-based listeners
            channel,
            space,
            session,
            check,
            noise,
            ready_when,
        )

    def _trim_window(self, cutoff: Time) -> None:
        # NOTE: this must be called with the lock held
        while self._window and self._window[0].time < cutoff:
            self._window.popleft()

    def _on_sample(self, stamped_sample: StampedSample) -> None:
        cutoff = stamped_sample.time - self._window_duration
        with self._lock:
            self._window.append(stamped_sample)
            self._trim_window(cutoff)

    def get_window(self) -> tuple[StampedSample]:
        cutoff = self._clock.now() - self._window_duration
        with self._lock:
            self._trim_window(cutoff)
            return tuple(self._window)
