import zenoh
from typing import Callable
from ark.comm import Channel
from ark.time import Clock
from ark_msgs import Envelope
from ark.comm.channel_noise import ChannelNoise
from ark.comm.end_point import SourceEndPoint
from ark.comm.stamped_message import StampedMessage
from ark.comm.utils import message_from_sample
from google.protobuf.message import Message


class Queryable(SourceEndPoint):
    """A Queryable end point that can receive queries and send replies on a zenoh channel."""

    def __init__(
        self,
        world_name: str,
        node_name: str,
        session: zenoh.Session,
        channel: str | Channel,
        clock: Clock,
        callback: Callable[[StampedMessage], Message | None],
        noise: ChannelNoise | None = None,
    ):
        src_type = Envelope.SourceType.REPLY
        super().__init__(src_type, world_name, node_name, session, channel, clock, noise)
        self._callback = callback
        self._queryable = self._session.declare_queryable(self._channel, self._on_query)

    def _on_query(self, query: zenoh.Query):
        try:
            req_msg = None
            if query.value is not None:
                req_msg = message_from_sample(query.value)

            trec = self._clock.now()
            resp_msg = self._callback(StampedMessage(trec, req_msg))
            if resp_msg is None:
                return

            query.reply(self.pack_envelope(resp_msg).SerializeToString())
        except Exception:
            return

    def get_z_obj(self):
        return self._queryable
