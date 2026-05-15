import zenoh
from gymnasium import Space
from ark._msgs import Envelope
from ark.comm import Channel
from typing import Any, Callable
from ark.time import Clock, Time, Stepper
from .end_point import SourceEndPoint, EnvelopePacker
from ark.comm.channel_noise import ChannelNoise


class Publisher(SourceEndPoint):
    """A Publisher end point that can publish samples to a zenoh channel."""

    def __init__(
        self,
        channel: str | Channel,
        space: Space,
        session: zenoh.Session,
        clock: Clock,
        node_name: str,
        noise: ChannelNoise | None,
        check_space: bool,
    ):
        """Initialize the Publisher with the given node name, zenoh session, channel, clock and optional noise function."""

        envelope_packer = EnvelopePacker(
            channel,
            space,
            clock,
            node_name,
            Envelope.SourceType.PUBLISH,
            noise,
            check_space,
        )
        super().__init__(channel, session, clock, envelope_packer)
        self._z_objs["pub"] = self._session.declare_publisher(self._channel)
        self._z_objs["space_queryable"] = self.declare_space_queryable("pub", space)

    def publish(self, sample: Any):
        """Publish a sample."""
        env = self._envelope_packer.pack(sample)
        self._z_objs["pub"].put(env.SerializeToString())


class PeriodicPublisher(Publisher):
    """A Publisher that can publish samples at a fixed rate using a function that builds each sample based on the current time."""

    def __init__(
        self,
        channel: str | Channel,
        space: Space,
        session: zenoh.Session,
        clock: Clock,
        node_name: str,
        noise: ChannelNoise,
        check_space: bool,
        hz: float,
        message_factory: Callable[[Time], Any],
    ):
        """A Publisher that can publish samples at a fixed rate using a function that builds each sample based on the current time."""
        super().__init__(channel, space, session, clock, node_name, noise, check_space)
        self._stepper = Stepper(clock, hz, lambda t: self.publish(message_factory(t)))

    def close(self):
        self._stepper.close()
        super().close()
