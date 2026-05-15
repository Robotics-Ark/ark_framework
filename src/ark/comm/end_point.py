import zenoh
import threading
from dataclasses import dataclass
from typing import Any
from gymnasium import Space
from ark.comm import Channel
from ark.comm import msgs
from ark.time import Clock
from ark._msgs import Envelope
from ark.comm.channel_noise import ChannelNoise, NoNoise


@dataclass(frozen=True, slots=True)
class QuerySpace:
    """Request and reply spaces for a query channel."""

    request: Space
    reply: Space


class EndPoint:
    """Common channel, clock, lifecycle, and Zenoh helpers."""

    def __init__(
        self,
        channel: str | Channel,
        session: zenoh.Session,
        clock: Clock,
    ):
        """Initialize shared endpoint state."""
        self._channel = Channel(channel)
        self._session = session
        self._clock = clock
        self._z_objs: dict[str, Any] = {}

    def close(self) -> None:
        """Close this end point and release any zenoh resources it holds."""
        for z_obj in self._z_objs.values():
            z_obj.undeclare()

    def declare_space_queryable(
        self, role: str | Channel, space: Space
    ) -> zenoh.Queryable:
        """Declare a queryable that replies with this endpoint role's schema."""
        encoded_space = msgs.encode_space(space)
        space_channel = self._channel / role / "get_space"

        def on_space_query(z_query: zenoh.Query) -> None:
            with z_query:
                z_query.reply(z_query.key_expr, encoded_space)

        return self._session.declare_queryable(space_channel, on_space_query)

    @staticmethod
    def decode_sample(space: Space, payload: Any) -> Any:
        """Decode a sample from a serialized Ark envelope."""
        env = Envelope()
        env.ParseFromString(bytes(payload))
        return msgs.decode_sample(space, env.payload)


class EnvelopePacker:
    """Encodes outgoing samples into Ark envelopes with trace metadata."""

    def __init__(
        self,
        channel: str | Channel,
        space: Space,
        clock: Clock,
        node_name: str,
        src_type: Envelope.SourceType,
        noise: ChannelNoise | None,
        check_space: bool,
    ):
        self._channel = Channel(channel)
        self._space = space
        self._clock = clock
        self._node_name = node_name
        self._src_type = src_type
        self._noise = noise or NoNoise()
        self._check_space = check_space
        self._seq_index = 0
        self._pack_lock = threading.Lock()

    def pack(self, sample: Any) -> Envelope:
        with self._pack_lock:
            sample = self._noise.apply(sample)
            self._check_sample(sample)
            payload = msgs.encode_sample(self._space, sample)
            trace = self._next_trace()

        return Envelope(
            channel=str(self._channel),
            trace=trace,
            payload=payload,
        )

    def _check_sample(self, sample: Any) -> None:
        if self._check_space and not self._space.contains(sample):
            raise ValueError(
                f"Sample does not conform to the provided Gymnasium space {self._space}."
            )

    def _next_trace(self) -> Envelope.TraceMeta:
        trace = Envelope.TraceMeta(
            src_node_name=self._node_name,
            src_type=self._src_type,
            sent_seq_index=self._seq_index,
            sent_ark_time_ns=self._clock.now().nanosec,
            sent_wall_time_ns=self._clock.wall_now().nanosec,
        )
        self._seq_index += 1
        return trace


class SourceEndPoint(EndPoint):
    """Base class for endpoints that send samples."""

    def __init__(
        self,
        channel: str | Channel,
        session: zenoh.Session,
        clock: Clock,
        envelope_packer: EnvelopePacker,
    ):
        """Initialize endpoint state used for outbound envelopes."""
        super().__init__(channel, session, clock)
        self._envelope_packer = envelope_packer
