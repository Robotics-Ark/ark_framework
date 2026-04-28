import zenoh
from enum import Enum
from typing import Callable
from ark.comm import Channel
from collections import deque
from ark.time import Clock, Time
from ark.comm.end_point import EndPoint
from ark.comm.stamped_message import StampedMessage
from ark.comm.utils import message_from_sample


class Subscriber(EndPoint):
    """A Subscriber end point that can receive messages from a zenoh channel."""

    def __init__(
        self,
        node_name: str,
        session: zenoh.Session,
        channel: str | Channel,
        clock: Clock,
        callback: Callable[[StampedMessage], None],
    ):
        super().__init__(node_name, session, channel, clock)
        self._callback = callback
        self._sub = self._session.declare_subscriber(self._channel, self._on_sample)

    def _on_sample(self, sample: zenoh.Sample):
        trec = self._clock.now()
        self._callback(StampedMessage(trec, message_from_sample(sample)))

    def get_z_obj(self):
        return self._sub


class ReadyWhen(Enum):
    """Enum for specifying when a SampleWindowListener should be considered ready."""

    ALWAYS = "always"  # always ready
    FULL = "full"  # ready when the buffer is full
    ANY = "any"  # ready when there is at least one message in the buffer


class SampleWindowListener(Subscriber):

    is_ready_func = {
        ReadyWhen.ALWAYS: lambda s: True,
        ReadyWhen.FULL: lambda s: len(s._buffer) == s._buffer.maxlen,
        ReadyWhen.ANY: lambda s: len(s._buffer) > 0,
    }

    def __init__(
        self,
        node_name: str,
        session: zenoh.Session,
        channel: str | Channel,
        clock: Clock,
        n_buffer: int = 1,
        ready_when: ReadyWhen | str = ReadyWhen.ALWAYS,
    ):
        self._buffer: deque[StampedMessage] = deque(maxlen=int(n_buffer))
        self._ready_when = ReadyWhen(ready_when)
        self._is_ready = self.is_ready_func[self._ready_when]
        super().__init__(node_name, session, channel, clock, self._buffer.append)

    def is_ready(self) -> bool:
        """Return whether this listener is ready according to its ready_when condition."""
        return self._is_ready(self)

    def get(self) -> list[StampedMessage]:
        if not self.is_ready():
            raise RuntimeError("SampleWindowListener is not ready")
        return list(self._buffer)


class TimeWindowListener(Subscriber):
    """A listener that retains messages received within a rolling time window."""

    def __init__(
        self,
        node_name: str,
        session: zenoh.Session,
        channel: str | Channel,
        clock: Clock,
        window_sec: float,
    ):
        self._buffer: deque[StampedMessage] = deque()
        self._window_nanosec = round(float(window_sec) * 1e9)
        super().__init__(node_name, session, channel, clock, self._append)

    def _append(self, stamped_message: StampedMessage) -> None:
        self._buffer.append(stamped_message)
        self._prune(stamped_message.time)

    def _prune(self, now: Time) -> None:
        cutoff_nanosec = now.nanosec - self._window_nanosec
        buffer = self._buffer
        while buffer and buffer[0].time.nanosec < cutoff_nanosec:
            buffer.popleft()

    def get(self) -> list[StampedMessage]:
        self._prune(self._clock.now())
        return list(self._buffer)
