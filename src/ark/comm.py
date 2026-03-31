from __future__ import annotations

import zenoh
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
        return type(self)(self, other)

    def joinpath(self, *others: str | "Channel") -> "Channel":
        return type(self)(self, *others)

    @property
    def parts(self) -> tuple[str, ...]:
        return tuple(str(self).split(self._separator))

    @property
    def parent(self) -> "Channel":
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
    ):
        self._node_name = node_name
        self._session = session
        self._channel = Channel(channel)
        self._comm = None  # instance of Publisher, Subscriber, Querier or Queryable
        self.post_init()

    def post_init(self):
        pass  # for subclasses to override if they need to do additional initialization after the base init

    @abstractmethod
    def close(self):
        """Close this end point and release any resources it holds."""


class ObservationNoise(ABC):

    def __init__(self, session: zenoh.Session):
        self._session = session
        self._sub = self._session.declare_subscriber("ark/reset", self._on_reset_sample)

    def _on_reset_sample(self, sample: zenoh.Sample):
        self.reset()

    @abstractmethod
    def reset(self):
        """Reset any internal state of the noise process, if necessary."""

    @abstractmethod
    def apply(self, msg: Message) -> Message:
        """Apply noise to the given message and return the noised message."""


class SourceEndPoint(EndPoint):
    """Base class for communication end points that can be the source of a trace (Publisher, Querier, Queryable)."""

    def __init__(
        self,
        node_name: str,
        session: zenoh.Session,
        channel: str | Channel,
        clock: Clock,
        apply_noise: Callable[[Message], Message] | None = None,
    ):
        super().__init__(node_name, session, channel)
        self._clock = clock
        self._seq_index = 0
        self._apply_noise = apply_noise

    source_type_name: str | None = None

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

    def close(self):
        self._pub.undeclare()


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
        msg = self._message_factory(t)
        self.publish(msg)


class Subscriber(EndPoint):
    """A Subscriber end point that can receive messages from a zenoh channel."""

    def __init__(
        self,
        node_name: str,
        session: zenoh.Session,
        channel: str | Channel,
        callback: Callable[[Message], None],
    ):
        super().__init__(node_name, session, channel)
        self._callback = callback
        self._sub = self._session.declare_subscriber(self._channel, self._on_sample)

    def _on_sample(self, sample: zenoh.Sample):
        self._callback(message_from_sample(sample))

    def close(self):
        self._sub.undeclare()


class Listener(Subscriber):

    def __init__(
        self,
        node_name: str,
        session: zenoh.Session,
        channel: str | Channel,
        n_buffer: int = 1,
        ready_when: str = "full",
    ):
        self._buffer = deque(maxlen=n_buffer)
        self._n_buffer = n_buffer
        if ready_when == "full":
            self.is_ready = self._is_ready_when_full
        elif ready_when == "any":
            self.is_ready = self._is_ready_when_any
        else:
            raise ValueError(f"Unknown ready_when value: {ready_when}")
        super().__init__(node_name, session, channel, self._buffer.append)

    def _is_ready_when_full(self) -> bool:
        return len(self._buffer) == self._n_buffer

    def _is_ready_when_any(self) -> bool:
        return len(self._buffer) > 0

    def get(self) -> list[Message]:
        if not self.is_ready():
            raise RuntimeError("Listener is not ready")
        return list(self._buffer)


class Querier(SourceEndPoint):
    """A Querier end point that can send queries and receive replies from a zenoh channel."""

    source_type_name = "QUERY"

    def post_init(self):
        self._querier = self._session.declare_querier(self._channel)

    def query(self, req: Message | None = None, timeout: float = 10.0) -> Message:
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
            return message_from_sample(reply.ok.sample)
        else:
            raise TimeoutError(
                f"No OK reply received for query on '{self._channel}' within {timeout}s"
            )

    def close(self):
        self._querier.undeclare()


class Queryable(SourceEndPoint):
    """A Queryable end point that can receive queries and send replies on a zenoh channel."""

    source_type_name = "REPLY"

    def __init__(
        self,
        node_name: str,
        session: zenoh.Session,
        clock: Clock,
        channel: str | Channel,
        callback: Callable[[Message | None], Message | None],
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

            resp_msg = self._callback(req_msg)
            if resp_msg is None:
                return

            resp_env = self.pack_envelope(resp_msg)
            query.reply(resp_env.SerializeToString())
        except Exception:
            return

    def close(self):
        self._queryable.undeclare()
