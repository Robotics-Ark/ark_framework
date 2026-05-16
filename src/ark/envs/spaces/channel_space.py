import zenoh
import numpy as np
from typing import Any
from ark.time import Clock
from dataclasses import dataclass
from ark._msgs import Envelope
from ark.comm.end_point import Role
from ark.comm.channel import Channel
from ark.comm.end_point import query_space
from gymnasium.spaces import Dict as GymDict
from ark.comm.publisher import Publisher
from ark.comm.serialization import Encoder, Decoder
from ark.comm.stamped_sample import StampedSample
from ark.comm.subscriber import SampleWindowListener, TimeWindowListener, ReadyWhen


class ChannelSpace(GymDict):
    """Base class for framework-internal channel aggregation spaces.

    These spaces are constructed from live Zenoh channel metadata and own
    communication resources. They are not user-facing application spaces and are
    not supported by ark.comm.sample space/sample serialization.
    """

    query_role: Role

    def __init__(
        self,
        channels: list[str | Channel],
        session: zenoh.Session,
        seed: dict | int | np.random.Generator | None,
    ):
        space = lambda ch: query_space(ch, self.query_role, session)
        super().__init__({str(ch): space(ch) for ch in channels}, seed=seed)
        self._z_objs = {}  # for storing Zenoh publishers, listeners, etc. in subclasses

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self.spaces.keys())})"

    def close(self):
        for z_obj in self._z_objs.values():
            z_obj.undeclare()


class OutboundChannels(ChannelSpace):
    """Framework-internal action space for publishing to subscriber channels.

    User applications should not expose or serialize this space directly; use it
    only as part of Ark's environment communication machinery.
    """

    query_role = Role.SUBSCRIBER

    def __init__(
        self,
        channels: list[str | Channel],
        session: zenoh.Session,
        clock: Clock,
        node_name: str,
        check_space: bool,
        seed: dict | int | np.random.Generator | None = None,
    ):
        super().__init__(channels, session, seed)

        init_enc = lambda ch: Encoder(
            ch,
            self[ch],
            clock,
            node_name,
            Envelope.SourceType.PUBLISH,
            None,  # noise
            check_space,
        )
        init_pub = lambda ch: Publisher(init_enc(ch), session)
        for ch in channels:
            self._z_objs[str(ch)] = init_pub(ch)

    def publish(self, action: dict[str, Any]):
        for ch, pub in self._z_objs.items():
            pub.publish(action[str(ch)])


@dataclass
class InboundChannelSpec:
    channel: str | Channel
    listener_cls: type[SampleWindowListener] | type[TimeWindowListener]
    window: int | float
    ready_when: ReadyWhen | None = None  # only used for SampleWindowListener

    @classmethod
    def from_dict(cls, d: dict):
        channel = Channel(d["channel"])
        if d["listener"] == "sample_window":
            listener_cls = SampleWindowListener
            window = d["window"]
            ready_when = ReadyWhen(d.get("ready_when", "ALWAYS"))
        elif d["listener"] == "time_window":
            listener_cls = TimeWindowListener
            window = d["window"]
            ready_when = None
        return cls(channel, listener_cls, window, ready_when)


class InboundChannels(ChannelSpace):
    """Framework-internal observation space for subscribed publisher channels.

    User applications should not expose or serialize this space directly; use it
    only as part of Ark's environment communication machinery.
    """

    query_role = Role.PUBLISHER

    def __init__(
        self,
        specs: list[InboundChannelSpec],
        session: zenoh.Session,
        clock: Clock,
        seed: dict | int | np.random.Generator | None = None,
    ):
        super().__init__([s.channel for s in specs], session, seed)

        def init_listener(s: InboundChannelSpec):
            dec = Decoder(
                s.channel,
                query_space(s.channel, Role.PUBLISHER, session),
                clock,
            )
            return s.listener_cls(dec, session, s.window, s.ready_when)

        for s in specs:
            self._z_objs[str(s.channel)] = init_listener(s)

    def is_ready(self) -> bool:
        return all(l.is_ready() for l in self._z_objs.values())

    def get(self) -> dict[str, list[StampedSample]]:
        return {ch: l.get() for ch, l in self._z_objs.items()}
