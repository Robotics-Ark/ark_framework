import zenoh
from typing import Any
from threading import Lock
from ark.time import Clock
from gymnasium import Space
from ark._msgs import Envelope
from ark.comm.channel import Channel
from ark.comm.sample import encode_sample, decode_sample
from ark.comm.stamped_sample import StampedSample
from ark.comm.channel_noise import ChannelNoise, NoNoise


class BaseCoder:

    def __init__(self, channel: str | Channel, space: Space, clock: Clock):
        self._channel = Channel(channel)
        self._space = space
        self._clock = clock

    @property
    def channel(self) -> Channel:
        """Read only property to access the Channel associated with this coder."""
        return self._channel

    @property
    def space(self) -> Space:
        """Read only property to access the Gymnasium space associated with this coder."""
        return self._space

    @property
    def clock(self) -> Clock:
        """Read only property to access the Clock associated with this coder."""
        return self._clock


class Encoder(BaseCoder):
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
        super().__init__(channel, space, clock)
        self._node_name = node_name
        self._src_type = src_type
        self._noise = noise or NoNoise()
        self._check_space = check_space
        self._seq_index = 0
        self._pack_lock = Lock()

    def __call__(self, sample: Any) -> bytes:
        with self._pack_lock:
            sample = self._noise.apply(sample)
            self._check_sample(sample)
            payload = encode_sample(self._space, sample)
            trace = self._next_trace()

        env = Envelope(
            channel=str(self._channel),
            trace=trace,
            payload=payload,
        )

        return env.SerializeToString()

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


class Decoder(BaseCoder):
    """Decodes incoming Ark envelopes."""

    def __call__(self, z_sample: zenoh.Sample | zenoh.Query) -> StampedSample:
        """Decode a sample from a serialized Ark envelope and return as a StampedSample."""
        env = Envelope()
        env.ParseFromString(bytes(z_sample.payload))
        sample = decode_sample(self._space, env.payload)
        trec = self._clock.now()
        return StampedSample(trec, sample)
