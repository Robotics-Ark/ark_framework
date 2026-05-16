import zenoh
from .query import Query
from .end_point import Role
from typing import Any, Callable
from .stamped_sample import StampedSample
from .serialization import Encoder, Decoder


class Queryable(Query):

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        session: zenoh.Session,
        callback: Callable[[StampedSample], Any],
    ):
        super().__init__(encoder, decoder, session)
        self._callback = callback
        q = session.declare_queryable(encoder.channel, self.on_query)
        self.add_z_obj("queryable", q, self._query_space, Role.QUERYABLE)

    def on_query(self, z_query: zenoh.Query) -> None:
        with z_query:
            try:
                if z_query.payload is None:
                    raise ValueError(f"Query on '{self._channel}' has no payload.")
                response = self._callback(self._decode(z_query))
                z_query.reply(z_query.key_expr, self._encode(response))
            except Exception as e:
                err = f"Error processing query on '{self._channel}': {e}"
                z_query.reply_err(err.encode("utf-8"))
