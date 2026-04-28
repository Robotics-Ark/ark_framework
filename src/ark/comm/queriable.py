import zenoh
from typing import Callable
from ark.comm import Channel
from ark.time import Clock
from ark.comm.end_point import SourceEndPoint
from ark.comm.stamped_message import StampedMessage
from ark.comm.utils import message_from_sample
from google.protobuf.message import Message


class Queryable(SourceEndPoint):
    """A Queryable end point that can receive queries and send replies on a zenoh channel."""

    def __init__(
        self,
        node_name: str,
        session: zenoh.Session,
        channel: str | Channel,
        clock: Clock,
        callback: Callable[[StampedMessage], Message | None],
        noise: Callable[[Message], Message] | None = None,
    ):
        super().__init__(node_name, session, channel, clock, noise)
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

            resp_env = self.pack_envelope(resp_msg)
            query.reply(resp_env.SerializeToString())
        except Exception:
            return

    def get_z_obj(self):
        return self._queryable
