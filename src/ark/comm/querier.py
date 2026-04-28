import zenoh
from ark.time import Clock
from ark.comm import Channel
from ark_msgs import Envelope
from ark.comm.channel_noise import ChannelNoise
from google.protobuf.message import Message
from ark.comm.end_point import SourceEndPoint
from ark.comm.stamped_message import StampedMessage
from ark.comm.utils import message_from_sample


class Querier(SourceEndPoint):
    """A Querier end point that can send queries and receive replies from a zenoh channel."""

    def __init__(
        self,
        env_name: str,
        node_name: str,
        session: zenoh.Session,
        channel: str | Channel,
        clock: Clock,
        noise: ChannelNoise | None = None,
    ):
        super().__init__(
            Envelope.SourceType.REPLY,
            env_name,
            node_name,
            session,
            channel,
            clock,
            noise,
        )

    def post_init(self):
        self._querier = self._session.declare_querier(self._channel)

    def query(
        self,
        req: Message | None = None,
        timeout: float = 10.0,
    ) -> StampedMessage:

        trec = self._clock.now()

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

            return StampedMessage(trec, message_from_sample(reply.ok.sample))
        else:
            raise TimeoutError(
                f"No OK reply received for query on '{self._channel}' within {timeout}s"
            )

    def get_z_obj(self):
        return self._querier
