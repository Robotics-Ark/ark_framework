import zenoh
from ark_msgs import Envelope
from ark.comm import Channel
from typing import Callable
from ark.time import Clock, Time, Stepper
from .end_point import SourceEndPoint
from google.protobuf.message import Message
from ark.comm.channel_noise import ChannelNoise


class Publisher(SourceEndPoint):
    """A Publisher end point that can publish messages to a zenoh channel."""

    def __init__(
        self,
        env_name: str,
        node_name: str,
        session: zenoh.Session,
        channel: str | Channel,
        clock: Clock,
        noise: ChannelNoise | None = None,
    ):
        """Initialize the Publisher with the given environment name, node name, zenoh session, channel, clock and optional noise function."""
        super().__init__(
            Envelope.SourceType.PUBLISH,
            env_name,
            node_name,
            session,
            channel,
            clock,
            noise,
        )

    def post_init(self):
        """Declare the zenoh publisher for this end point after the base initialization."""
        self._pub = self._session.declare_publisher(self._channel)

    def publish(self, msg: Message):
        """Publish a message."""
        env = self.pack_envelope(msg)
        self._pub.put(env.SerializeToString())

    def get_z_obj(self):
        """Get the underlying zenoh Publisher object that this end point uses to communicate."""
        return self._pub


class PeriodicPublisher(Publisher):
    """A Publisher that can publish messages at a fixed rate using a function that builds each message based on the current time."""

    def __init__(
        self,
        message_factory: Callable[[Time], Message],
        hz: float,
        env_name: str,
        node_name: str,
        session: zenoh.Session,
        channel: str | Channel,
        clock: Clock,
        apply_noise: Callable[[Message], Message] | None = None,
    ):
        """A Publisher that can publish messages at a fixed rate using a function that builds each message based on the current time."""
        super().__init__(
            Envelope.SourceType.PUBLISH,
            env_name,
            node_name,
            session,
            channel,
            clock,
            apply_noise=apply_noise,
        )
        self._message_factory = message_factory
        self._stepper = Stepper(clock, hz, self._step)

    def _step(self, t: Time):
        """Build a message for the current time and publish."""
        self.publish(self._message_factory(t))
