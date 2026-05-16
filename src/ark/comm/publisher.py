import zenoh
from typing import Any, Callable
from ark.time import Time, Stepper
from .end_point import EndPoint, Role
from .serialization import Encoder


class Publisher(EndPoint):
    """A Publisher end point that can publish samples to a zenoh channel."""

    def __init__(
        self,
        encoder: Encoder,
        session: zenoh.Session,
    ):
        """Initialize the Publisher with the given node name, zenoh session, channel, clock and optional noise function."""
        super().__init__(encoder.channel, session)
        self._encode = encoder
        p = self._session.declare_publisher(self._channel)
        self.add_z_obj("pub", p, encoder.space, Role.PUBLISHER)

    def publish(self, sample: Any):
        """Publish a sample."""
        self._z_objs["pub"].put(self._encode(sample))


class PeriodicPublisher(Publisher):
    """A Publisher that can publish samples at a fixed rate using a function that builds each sample based on the current time."""

    def __init__(
        self,
        encoder: Encoder,
        session: zenoh.Session,
        hz: float,
        sample: Callable[[Time], Any],
    ):
        """A Publisher that can publish samples at a fixed rate using a function that builds each sample based on the current time."""
        super().__init__(encoder, session)
        self._sample = sample
        self._stepper = Stepper(encoder.clock, hz, self.step)

    def step(self, t: Time):
        self.publish(self._sample(t))

    def close(self):
        self._stepper.close()
        super().close()
