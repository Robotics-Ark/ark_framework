import zenoh
from gymnasium import Space
from .channel import Channel
from .channel import ChannelName
from .codec.registry import space_codec


def query_space(channel_name: ChannelName, role: str, session: zenoh.Session) -> Space:
    qr = session.declare_querier(channel_name / role / "get_space")
    try:
        for z_reply in qr.get():
            if z_reply.err is not None:
                err = bytes(z_reply.err.payload).decode("utf-8", errors="replace")
                raise RuntimeError(f"Space query failed: {err}")
            elif z_reply.ok is not None:
                codec = space_codec.get(bytes(z_reply.ok.payload))
                return codec.decode(bytes(z_reply.ok.payload))
        else:
            raise RuntimeError("Space query failed: No reply received.")
    finally:
        qr.undeclare()


class QueryableSpace:

    def __init__(
        self, channel: Channel, role: str, space: Space, session: zenoh.Session
    ):
        codec = space_codec.get(space)
        self._enc_space = codec.encode(space)
        self._qr = session.declare_queryable(
            channel.full_name / role / "get_space", self._on_query
        )

    def _on_query(self, query: zenoh.Query) -> None:
        with query:
            query.reply(query.key_expr, self._enc_space)

    def undeclare(self) -> None:
        self._qr.undeclare()
