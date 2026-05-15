import zenoh
import threading
from enum import Enum
from gymnasium import Space
from typing import Callable
from ark.comm import Channel
from collections import deque
from ark.time import Clock, Time
from ark.comm.end_point import EndPoint
from ark.comm.stamped_sample import StampedSample


class Subscriber(EndPoint):
    """A Subscriber end point that can receive messages from a zenoh channel."""

    def __init__(
        self,
        channel: str | Channel,
        space: Space,
        session: zenoh.Session,
        clock: Clock,
        callback: Callable[[StampedSample], None],
    ):
        super().__init__(channel, session, clock)
        self._receive_space = space
        self._callback = callback
        self._z_objs["sub"] = self._session.declare_subscriber(
            self._channel, self._on_sample
        )
        self._z_objs["space_queryable"] = self.declare_space_queryable("sub", space)

    def _on_sample(self, z_sample: zenoh.Sample):
        sample = self.decode_sample(self._receive_space, z_sample.payload)
        t = self._clock.now()  # recieved time
        self._callback(StampedSample(t, sample))


class ReadyWhen(Enum):
    """Enum for specifying when a SampleWindowListener should be considered ready."""

    ALWAYS = "always"  # always ready
    FULL = "full"  # ready when the buffer is full
    ANY = "any"  # ready when there is at least one message in the buffer


class SampleWindowListener(Subscriber):
    """A listener that retains a fixed number of the most recent messages received."""

    is_ready_func = {
        ReadyWhen.ALWAYS: lambda s: True,
        ReadyWhen.FULL: lambda s: len(s._buffer) == s._buffer.maxlen,
        ReadyWhen.ANY: lambda s: len(s._buffer) > 0,
    }

    def __init__(
        self,
        channel: str | Channel,
        space: Space,
        session: zenoh.Session,
        clock: Clock,
        n_buffer: int,
        ready_when: ReadyWhen | str,
    ):
        self._buffer: deque[StampedSample] = deque(maxlen=int(n_buffer))
        self._ready_when = ReadyWhen(ready_when)
        self._is_ready = self.is_ready_func[self._ready_when]
        self._mutex = threading.Lock()
        super().__init__(channel, space, session, clock, self.append_buffer)

    def append_buffer(self, stamped_message: StampedSample) -> None:
        """Append the given message to the buffer, evicting old messages if necessary."""
        with self._mutex:
            self._buffer.append(stamped_message)

    def is_ready(self) -> bool:
        """Return whether this listener is ready according to its ready_when condition."""
        with self._mutex:
            return self._is_ready(self)

    def get(self) -> list[StampedSample]:
        """Return a list of the most recent messages in the buffer."""
        if not self.is_ready():
            raise RuntimeError("SampleWindowListener is not ready")
        with self._mutex:
            return list(self._buffer)


class TimeWindowListener(Subscriber):
    """A listener that retains messages received within a rolling time window."""

    def __init__(
        self,
        channel: str | Channel,
        space: Space,
        session: zenoh.Session,
        clock: Clock,
        window_sec: float,
    ):
        self._buffer: deque[StampedSample] = deque()
        self._window_nanosec = round(float(window_sec) * 1e9)
        self._mutex = threading.Lock()
        super().__init__(channel, space, session, clock, self._append)

    def _append(self, stamped_message: StampedSample) -> None:
        with self._mutex:
            self._buffer.append(stamped_message)
            self._prune(stamped_message.time)

    def _prune(self, now: Time) -> None:
        cutoff_nanosec = now.nanosec - self._window_nanosec
        buffer = self._buffer
        while buffer and buffer[0].time.nanosec < cutoff_nanosec:
            buffer.popleft()

    def get(self) -> list[StampedSample]:
        with self._mutex:
            self._prune(self._clock.now())
            return list(self._buffer)
