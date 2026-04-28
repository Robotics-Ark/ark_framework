import zenoh
from ark.comm import Channel
from ark.time import Clock
from ark_msgs import Envelope
from abc import ABC, abstractmethod
from google.protobuf.message import Message
from ark.comm.channel_noise import ChannelNoise, NoNoise


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


class SourceEndPoint(EndPoint):
    """Base class for communication end points that can be the source of a trace (Publisher, Querier, Queryable)."""

    source_type_name: str | None = None

    def __init__(
        self,
        node_name: str,
        session: zenoh.Session,
        channel: str | Channel,
        clock: Clock,
        noise: ChannelNoise | None = None,
    ):
        """Initialize the source end point with the given node name, zenoh session, channel, clock and optional noise function."""
        super().__init__(node_name, session, channel, clock)
        self._seq_index = 0
        self._noise = noise or NoNoise()

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
        msg = self._noise.apply(msg)

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
