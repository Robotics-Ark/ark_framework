import zenoh
import threading
from enum import Enum
from gymnasium import Space
from typing import Callable
from collections import deque
from .end_point import EndPoint, Role
from .serialization import Decoder
from .stamped_sample import StampedSample


class Subscriber(EndPoint):
    """A Subscriber end point that can receive messages from a zenoh channel."""

    def __init__(
        self,
        decoder: Decoder,
        session: zenoh.Session,
        callback: Callable[[StampedSample], None],
    ):
        super().__init__(decoder.channel, session)
        self._decode = decoder
        self._callback = callback
        s = self._session.declare_subscriber(decoder.channel, self.on_sample)
        self.add_z_obj("sub", s, decoder.space, Role.SUBSCRIBER)

    def on_sample(self, z_sample: zenoh.Sample):
        sample = self._decode(z_sample)
        t = self._clock.now()  # received time
        self._callback(StampedSample(t, sample))

    @property
    def space(self) -> Space:
        return self._decode.space


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
        decoder: Decoder,
        session: zenoh.Session,
        window: int | None,
        ready_when: ReadyWhen | str,
    ):
        self._buffer: deque[StampedSample] = deque(maxlen=window)
        self._ready_when = ReadyWhen(ready_when)
        self._is_ready = self.is_ready_func[self._ready_when]
        self._mutex = threading.Lock()
        super().__init__(decoder, session, self.append_buffer)

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
            raise RuntimeError("listener is not ready")
        with self._mutex:
            return list(self._buffer)


class TimeWindowListener(SampleWindowListener):
    """A listener that retains messages received within a rolling time window."""

    def __init__(
        self,
        decoder: Decoder,
        session: zenoh.Session,
        window: float,
        ready_when_: (
            ReadyWhen | str
        ) = None,  # NOTE: this is not used, kept for compatibility with InboundChannelSpec
    ):
        self._buffer: deque[StampedSample] = deque()
        self._window_nanosec = round(float(window) * 1e9)
        super().__init__(
            decoder,
            session,
            None,  # no fixed buffer size
            ReadyWhen.ALWAYS,
        )

    def append_buffer(self, stamped_message):
        super().append_buffer(stamped_message)
        self.prune()

    def prune(self) -> None:
        """Remove messages from the buffer that are outside the time window."""
        now = self._clock.now()
        with self._mutex:
            cutoff_nanosec = now.nanosec - self._window_nanosec
            buffer = self._buffer
            while buffer and buffer[0].time.nanosec < cutoff_nanosec:
                buffer.popleft()

    def get(self) -> list[StampedSample]:
        """Return a list of messages received within the time window."""
        self.prune()
        return super().get()
