import zenoh
from gymnasium import Space
from typing import Callable
from ark.time import Clock
from .end_point import EndPoint
from .codec.registry import sample_codec
from .queryable_space import QueryableSpace
from .stamped_sample import StampedSample
from .channel import Channel, ChannelNoise, NoNoise


class Subscriber(EndPoint):

    role = "subscriber"

    def __init__(
        self,
        channel: Channel,
        space: Space,
        callback: Callable[[StampedSample], None],
        session: zenoh.Session,
        check: bool,
        noise: ChannelNoise | None,
    ):
        super().__init__(channel, session)
        self._space = space
        self._callback = callback
        self._check = check
        self._noise = noise or NoNoise()
        self._clock = Clock()
        self._z_sub = self._session.declare_subscriber(
            self._channel.name, self._on_callback
        )
        self._z_sub_qr = QueryableSpace(
            self._channel, self.role, self._space, self._session
        )
        self._codec = sample_codec.get(self._space)

    def _on_callback(self, sample: zenoh.Sample):
        t = self._clock.now()
        sample = self._codec.decode(sample.payload)
        sample = self._noise.apply(sample)
        if self._check and not self._space.contains(sample):
            raise ValueError(
                f"Sample {sample} does not conform to the space {self._space}"
            )
        stamped_sample = StampedSample(t, sample)
        self._callback(stamped_sample)

    def close(self):
        self._z_sub.undeclare()
        self._z_sub_qr.undeclare()
