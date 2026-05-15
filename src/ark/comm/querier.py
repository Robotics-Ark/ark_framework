import zenoh
from typing import Any
from ark.time import Clock
from ark.comm import Channel
from ark._msgs import Envelope
from ark.comm.channel_noise import ChannelNoise
from ark.comm.end_point import QuerySpace, SourceEndPoint, EnvelopePacker
from ark.comm.stamped_sample import StampedSample


class Querier(SourceEndPoint):
    """Sends query requests and returns replies decoded with reply_space."""

    def __init__(
        self,
        channel: str | Channel,
        query_space: QuerySpace,
        session: zenoh.Session,
        clock: Clock,
        node_name: str,
        noise: ChannelNoise,
        check_space: bool,
        timeout: float,
    ):
        self._query_space = query_space
        envelope_packer = EnvelopePacker(
            channel,
            self._query_space.request,
            clock,
            node_name,
            Envelope.SourceType.QUERY,
            noise,
            check_space,
        )
        super().__init__(channel, session, clock, envelope_packer)
        self._timeout = timeout
        self._z_objs["querier"] = self._session.declare_querier(
            self._channel, timeout=self._timeout
        )

    def query(self, request: Any) -> StampedSample:
        query_kwargs = {
            "payload": self._envelope_packer.pack(request).SerializeToString(),
        }

        for z_reply in self._z_objs["querier"].get(**query_kwargs):
            return self._decode_reply(z_reply)
        else:
            raise TimeoutError(
                f"No OK reply received for query on '{self._channel}' within {self._timeout}s"
            )

    def _decode_reply(self, z_reply: zenoh.Reply) -> StampedSample:
        err = z_reply.err
        if err is not None:
            error_message = bytes(err.payload).decode("utf-8", errors="replace")
            raise RuntimeError(f"Query on '{self._channel}' failed: {error_message}")

        ok = z_reply.ok
        if ok is None:
            raise RuntimeError(f"Query on '{self._channel}' received an empty reply.")

        received_time = self._clock.now()
        return StampedSample(
            received_time,
            self.decode_sample(self._query_space.reply, ok.payload),
        )
