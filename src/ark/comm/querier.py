import zenoh
from typing import Any
from .query import Query
from ark.comm.end_point import Role
from .serialization import Encoder, Decoder
from ark.comm.stamped_sample import StampedSample


class Querier(Query):
    """Sends query requests and returns replies decoded with reply_space."""

    def __init__(
        self, encoder: Encoder, decoder: Decoder, session: zenoh.Session, timeout: float
    ):
        super().__init__(encoder, decoder, session)
        q = session.declare_querier(encoder.channel, timeout=timeout)
        self.add_z_obj("querier", q, self._query_space, Role.QUERIER)

    def __call__(self, request: Any) -> StampedSample:
        q: zenoh.Querier = self._z_objs["querier"]
        for z_reply in q.get(payload=self._encode(request)):
            if z_reply.ok:
                return self._decode(z_reply.ok)
            elif z_reply.err:
                err = bytes(z_reply.err.payload).decode("utf-8", errors="replace")
                raise RuntimeError(f"Query on '{self._channel}' failed: {err}")
        else:
            raise TimeoutError(
                f"No reply received for query on '{self._channel}' within {self._z_objs['querier'].timeout}s"
            )
