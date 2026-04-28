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
        world_name: str,
        node_name: str,
        session: zenoh.Session,
        channel: str | Channel,
        clock: Clock,
        noise: ChannelNoise | None,
        timeout: float,
    ):
        src_type = Envelope.SourceType.QUERY
        super().__init__(src_type, world_name, node_name, session, channel, clock, noise)
        self._timeout = timeout

    def post_init(self):
        self._querier = self._session.declare_querier(
            self._channel, timeout=self._timeout
        )

    def query(self, req: Message | None = None) -> StampedMessage:
        _req = {}
        if req:
            _req["payload"] = self.pack_envelope(req).SerializeToString()

        for reply in self._querier.get(**_req):
            if reply.ok is None:
                continue
            trec = self._clock.now()
            return StampedMessage(trec, message_from_sample(reply.ok.sample))
        else:
            raise TimeoutError(
                f"No OK reply received for query on '{self._channel}' within {self._timeout}s"
            )

    def get_z_obj(self):
        return self._querier
