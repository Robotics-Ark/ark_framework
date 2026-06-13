import zenoh
from typing import Any
from gymnasium import Space
from .end_point import EndPoint
from .queryable_space import QueryableSpace
from .channel import Channel, ChannelNoise, NoNoise
from .codec.registry import sample_codec


class Publisher(EndPoint):

    def __init__(
        self,
        channel: Channel,
        space: Space,
        session: zenoh.Session,
        check: bool,
        noise: ChannelNoise | None,
    ):
        super().__init__(channel, session)
        self._space = space
        self._check = check
        self._noise = noise or NoNoise()
        self._z_pub = self._session.declare_publisher(self._channel.full_name)
        self._z_pub_qr = QueryableSpace(
            self._channel, "publisher", self._space, self._session
        )
        self._codec = sample_codec.get(self._space)

    def publish(self, sample: Any):
        if self._check and not self._space.contains(sample):
            raise ValueError(
                f"Sample {sample} does not conform to channel space {self._space}"
            )
        noisy_sample = self._noise.apply(sample)
        self._z_pub.put(self._codec.encode(noisy_sample))

    def close(self) -> None:
        self._z_pub.undeclare()
        self._z_pub_qr.undeclare()
