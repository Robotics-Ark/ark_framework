import zenoh
from enum import Enum
from typing import Any
from gymnasium import Space
from .channel import Channel
from ark.comm.sample import encode_space, decode_space


class Role(Enum):
    """Roles for end points, used to differentiate between publisher and subscriber queryables on the same channel."""

    PUBLISHER = "publisher"
    SUBSCRIBER = "subscriber"
    QUERYABLE = "queryable"
    QUERIER = "querier"


class SpaceQueryable:
    """Exposes queryable that returns the encoded space of an end point's channel upon request."""

    def __init__(
        self, channel: Channel, role: Role, space: Space, session: zenoh.Session
    ):
        self._enc_space = encode_space(space)
        qr_channel = channel / role.value / "get_space"
        self._qr = session.declare_queryable(qr_channel, self.on_query)
        self.undeclare = self._qr.undeclare  # Expose for management by EndPoint

    def on_query(self, query: zenoh.Query) -> None:
        with query:
            query.reply(query.key_expr, self._enc_space)


def query_space(channel: Channel, role: Role, session: zenoh.Session) -> Space:
    """Helper function to query a channel for the space of a publisher or subscriber."""
    qr_channel = channel / role.value / "get_space"
    qr = session.declare_querier(qr_channel)
    try:
        for z_reply in qr.get():
            if z_reply.err is not None:
                err = bytes(z_reply.err.payload).decode("utf-8", errors="replace")
                raise RuntimeError(f"Space query failed: {err}")
            elif z_reply.ok is not None:
                return decode_space(z_reply.ok.payload)
        else:
            raise RuntimeError("Space query failed: No reply received.")
    finally:
        qr.undeclare()


class EndPoint:
    """Common channel, clock, lifecycle, and Zenoh helpers."""

    def __init__(self, channel: Channel, session: zenoh.Session):
        """Initialize shared endpoint state."""
        self._channel = Channel(channel)
        self._session = session
        self._z_objs: dict[str, Any] = {}

    def add_z_obj(self, name: str, z_obj: Any, space: Space, role: Role):
        """Add a zenoh object to this end point's lifecycle management."""
        if any(k.startswith(name) for k in self._z_objs.keys()):
            raise ValueError(
                f"Zenoh object name '{name}' conflicts with existing names in this end point."
            )
        self._z_objs[name] = z_obj
        self._z_objs[f"{name}/space_queryable"] = SpaceQueryable(
            self._channel, role, space, self._session
        )

    def close(self) -> None:
        """Close this end point and release any zenoh resources it holds."""
        for z_obj in self._z_objs.values():
            z_obj.undeclare()
