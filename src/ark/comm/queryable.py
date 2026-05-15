import zenoh
from typing import Any, Callable
from ark.comm import Channel
from ark.time import Clock
from ark._msgs import Envelope
from ark.comm.channel_noise import ChannelNoise
from ark.comm.stamped_sample import StampedSample
from ark.comm.end_point import QuerySpace, SourceEndPoint, EnvelopePacker


class Queryable(SourceEndPoint):
    """Receives query requests and replies with samples encoded in reply_space."""

    def __init__(
        self,
        channel: str | Channel,
        query_space: QuerySpace,
        session: zenoh.Session,
        clock: Clock,
        node_name: str,
        noise: ChannelNoise,
        check_space: bool,
        callback: Callable[[StampedSample], Any],
    ):
        self._query_space = query_space
        envelope_packer = EnvelopePacker(
            channel,
            self._query_space.reply,
            clock,
            node_name,
            Envelope.SourceType.REPLY,
            noise,
            check_space,
        )
        super().__init__(channel, session, clock, envelope_packer)
        self._callback = callback
        self._z_objs["queryable"] = self._session.declare_queryable(
            self._channel, self._on_query
        )
        self._z_objs["request_space_queryable"] = self.declare_space_queryable(
            "query/request", self._query_space.request
        )
        self._z_objs["reply_space_queryable"] = self.declare_space_queryable(
            "query/reply", self._query_space.reply
        )

    def _on_query(self, z_query: zenoh.Query) -> None:
        with z_query:
            try:
                if z_query.payload is None:
                    raise ValueError(f"Query on '{self._channel}' has no payload.")

                request_sample = self.decode_sample(
                    self._query_space.request, z_query.payload
                )
                received_time = self._clock.now()
                reply_sample = self._callback(
                    StampedSample(received_time, request_sample)
                )
                payload = self._envelope_packer.pack(reply_sample).SerializeToString()
                z_query.reply(z_query.key_expr, payload)
            except Exception as exc:
                message = (
                    f"Queryable '{self._channel}' failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                z_query.reply_err(message.encode("utf-8"))
                raise
