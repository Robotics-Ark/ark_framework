from __future__ import annotations
from dataclasses import dataclass

import zenoh
from enum import Enum
from collections import deque
from ark_msgs import Envelope
from ark.time import Clock, Stepper, Time
from abc import ABC, abstractmethod
from collections.abc import Callable
from google.protobuf.message import Message


class Channel(str):
    """A channel name that joins segments with `/` and can build child channels with `/`."""

    _separator = "/"

    def __new__(cls, *parts: str | "Channel"):

        if not parts:
            raise ValueError("ChannelName requires at least one part")

        normalized_parts: list[str] = []
        for part in parts:
            normalized_parts.extend(cls._normalize_part(part))

        if not normalized_parts:
            raise ValueError("ChannelName cannot be empty")

        return str.__new__(cls, cls._separator.join(normalized_parts))

    @classmethod
    def _normalize_part(cls, part: str | "Channel") -> list[str]:
        """Normalize a channel part by stripping separators and splitting into segments. Also validate that the part is a string or Channel."""
        if not isinstance(part, (str, cls)):
            raise TypeError(
                "Channel parts must be strings or Channel instances, "
                f"got {type(part).__name__}"
            )

        text = str(part).strip(cls._separator)
        if not text:
            raise ValueError("Channel parts cannot be empty")

        return [segment for segment in text.split(cls._separator) if segment]

    def __truediv__(self, other: str | "Channel") -> "Channel":
        """Join this channel with another part using the separator."""
        return type(self)(self, other)

    def joinpath(self, *others: str | "Channel") -> "Channel":
        """Join this channel with multiple other parts using the separator."""
        return type(self)(self, *others)

    @property
    def parts(self) -> tuple[str, ...]:
        """The individual segments of this channel as a tuple."""
        return tuple(str(self).split(self._separator))

    @property
    def parent(self) -> "Channel":
        """The parent channel of this channel."""
        if len(self.parts) < 2:
            raise ValueError("Root channel names do not have a parent")
        return type(self)(*self.parts[:-1])


class EndPoint(ABC):
    """Base class for all communication end points (Publisher, Subscriber, Querier, Queryable)."""

    def __init__(
        self,
        node_name: str,  # the name of the node that owns this end point
        session: zenoh.Session,  # the zenoh session that this end point uses to communicate
        channel: str | Channel,  # channel this end point uses to communicate
        clock: Clock,  # the clock to use for timestamps in trace meta information
    ):
        """Initialize the end point with the given node name, zenoh session, channel and clock."""
        self._node_name = node_name
        self._session = session
        self._channel = Channel(channel)
        self._clock = clock
        self._comm = None  # instance of Publisher, Subscriber, Querier or Queryable
        self.post_init()

    def post_init(self):
        """Post-initialization hook that can be overridden by subclasses to perform additional setup after the base initialization."""
        ...

    @abstractmethod
    def get_z_obj(
        self,
    ) -> zenoh.Publisher | zenoh.Subscriber | zenoh.Querier | zenoh.Queryable:
        """Get the underlying zenoh object that this end point uses to communicate. This is needed for proper cleanup of the zenoh resources."""

    def close(self):
        """Close this end point and release any zenoh resources it holds."""
        self.get_z_obj().undeclare()


class ObservationNoise(ABC):
    """Base class for noise processes that can be applied to messages received by Subscribers or Queryables."""

    def __init__(self, session: zenoh.Session):
        """Initialize the noise process with the given zenoh session and subscribe to reset messages."""
        self._session = session
        self._sub = self._session.declare_subscriber("ark/reset", self._on_reset_sample)

    def _on_reset_sample(self, sample: zenoh.Sample):
        """Handle a reset sample by resetting the noise process."""
        self.reset()

    @abstractmethod
    def reset(self):
        """Reset any internal state of the noise process, if necessary."""

    @abstractmethod
    def apply(self, msg: Message) -> Message:
        """Apply noise to the given message and return the noised message."""


class SourceEndPoint(EndPoint):
    """Base class for communication end points that can be the source of a trace (Publisher, Querier, Queryable)."""

    source_type_name: str | None = None

    def __init__(
        self,
        node_name: str,
        session: zenoh.Session,
        channel: str | Channel,
        clock: Clock,
        apply_noise: Callable[[Message], Message] | None = None,
    ):
        """Initialize the source end point with the given node name, zenoh session, channel, clock and optional noise function."""
        super().__init__(node_name, session, channel, clock)
        self._seq_index = 0
        self._apply_noise = apply_noise

    @property
    def type(self) -> int:
        """The source type enum value for this end point."""
        if self.source_type_name is None:
            raise NotImplementedError(
                "SourceEndPoint subclasses must define source_type_name"
            )

        enum_desc = Envelope.DESCRIPTOR.enum_types_by_name.get("SourceType")
        if enum_desc is None:
            raise AttributeError(
                "Envelope.SourceType is not available in the generated protobuf"
            )

        try:
            return enum_desc.values_by_name[self.source_type_name].number
        except KeyError as exc:
            raise AttributeError(
                f"Envelope.SourceType has no value named {self.source_type_name!r}"
            ) from exc

    def pack_envelope(self, msg: Message) -> Envelope:
        """Create an Envelope with the given message and appropriate trace meta information."""

        # Apply noise to the message if a noise function is provided
        if self._apply_noise:
            msg = self._apply_noise(msg)

        # Create trace meta
        t = Envelope.TraceMeta(
            src_node_name=self._node_name,
            sent_seq_index=self._seq_index,
            sent_ark_time_ns=self._clock.now().nanosec,
            sent_wall_time_ns=self._clock.wall_now().nanosec,
        )

        # Create Envelope
        e = Envelope(
            msg_type=msg.DESCRIPTOR.full_name,
            channel=self._channel,
            trace=t,
            payload=msg.SerializeToString(),
        )

        # Increment sequence index for next message
        self._seq_index += 1

        return e


def message_from_sample(sample: zenoh.Sample) -> Message:
    """Extract the message from the given sample."""
    env = Envelope()
    env.ParseFromString(bytes(sample.payload))
    return env.extract_message()


class Publisher(SourceEndPoint):
    """A Publisher end point that can publish messages to a zenoh channel."""

    source_type_name = "PUBLISH"

    def post_init(self):
        self._pub = self._session.declare_publisher(self._channel)

    def publish(self, msg: Message):
        env = self.pack_envelope(msg)
        self._pub.put(env.SerializeToString())

    def get_z_obj(self):
        return self._pub


class PeriodicPublisher(Publisher):

    def __init__(
        self,
        message_factory: Callable[[Time], Message],
        hz: float,
        node_name: str,
        session: zenoh.Session,
        channel: str | Channel,
        clock: Clock,
        apply_noise: Callable[[Message], Message] | None = None,
    ):
        """A Publisher that can publish messages at a fixed rate using a function that builds each message."""
        super().__init__(node_name, session, channel, clock, apply_noise=apply_noise)
        self._message_factory = message_factory
        self._stepper = Stepper(clock, hz, self._step)

    def _step(self, t: Time):
        self.publish(self._message_factory(t))


class Subscriber(EndPoint):
    """A Subscriber end point that can receive messages from a zenoh channel."""

    def __init__(
        self,
        node_name: str,
        session: zenoh.Session,
        channel: str | Channel,
        clock: Clock,
        callback: Callable[[Time, Message], None],
    ):
        super().__init__(node_name, session, channel, clock)
        self._callback = callback
        self._sub = self._session.declare_subscriber(self._channel, self._on_sample)

    def _on_sample(self, sample: zenoh.Sample):
        trec = self._clock.now()
        msg = message_from_sample(sample)
        self._callback(trec, msg)

    def get_z_obj(self):
        return self._sub


class ReadyWhen(Enum):
    """Enum for specifying when a SampleWindowListener should be considered ready."""

    ALWAYS = "always"  # always ready
    FULL = "full"  # ready when the buffer is full
    ANY = "any"  # ready when there is at least one message in the buffer


@dataclass(frozen=True, slots=True)
class BufferedMessage:
    """A message stored in a listener's buffer, along with its receive time."""

    time: Time
    message: Message


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
        self._buffer: deque[BufferedMessage] = deque(maxlen=int(n_buffer))
        self._ready_when = ReadyWhen(ready_when)
        self._is_ready = self.is_ready_func[self._ready_when]
        append = lambda t, m: self._buffer.append(BufferedMessage(t, m))
        super().__init__(node_name, session, channel, clock, append)

    def is_ready(self) -> bool:
        """Return whether this listener is ready according to its ready_when condition."""
        return self._is_ready(self)

    def get(self) -> list[BufferedMessage]:
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
        self._buffer: deque[BufferedMessage] = deque()
        self._window_nanosec = round(float(window_sec) * 1e9)
        super().__init__(node_name, session, channel, clock, self._append)

    def _append(self, t: Time, msg: Message) -> None:
        self._buffer.append(BufferedMessage(t, msg))
        self._prune(t)

    def _prune(self, now: Time) -> None:
        cutoff_nanosec = now.nanosec - self._window_nanosec
        buffer = self._buffer
        while buffer and buffer[0].time.nanosec < cutoff_nanosec:
            buffer.popleft()

    def get(self) -> list[BufferedMessage]:
        self._prune(self._clock.now())
        return list(self._buffer)


class Querier(SourceEndPoint):
    """A Querier end point that can send queries and receive replies from a zenoh channel."""

    source_type_name = "QUERY"

    def post_init(self):
        self._querier = self._session.declare_querier(self._channel)

    def query(
        self,
        req: Message | None = None,
        timeout: float = 10.0,
    ) -> tuple[Time, Message]:
        if req is None:
            replies = self._querier.get(timeout=timeout)
        else:
            req_env = self.pack_envelope(req)
            replies = self._querier.get(
                value=req_env.SerializeToString(), timeout=timeout
            )

        for reply in replies:
            if reply.ok is None:
                continue
            trec = self._clock.now()
            return trec, message_from_sample(reply.ok.sample)
        else:
            raise TimeoutError(
                f"No OK reply received for query on '{self._channel}' within {timeout}s"
            )

    def get_z_obj(self):
        return self._querier


class Queryable(SourceEndPoint):
    """A Queryable end point that can receive queries and send replies on a zenoh channel."""

    source_type_name = "REPLY"

    def __init__(
        self,
        node_name: str,
        session: zenoh.Session,
        channel: str | Channel,
        clock: Clock,
        callback: Callable[[Time, Message | None], Message | None],
        apply_noise: Callable[[Message], Message] | None = None,
    ):
        super().__init__(node_name, session, channel, clock, apply_noise=apply_noise)
        self._callback = callback
        self._queryable = self._session.declare_queryable(self._channel, self._on_query)

    def _on_query(self, query: zenoh.Query):
        try:
            req_msg = None
            if query.value is not None:
                req_msg = message_from_sample(query.value)

            trec = self._clock.now()
            resp_msg = self._callback(trec, req_msg)
            if resp_msg is None:
                return

            resp_env = self.pack_envelope(resp_msg)
            query.reply(resp_env.SerializeToString())
        except Exception:
            return

    def get_z_obj(self):
        return self._queryable
