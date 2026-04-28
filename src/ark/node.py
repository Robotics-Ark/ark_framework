import ast
import sys
import zenoh
from pathlib import Path
from ark.comm import Channel
from ark.cli import BaseCliParser
from collections.abc import Callable
from ark.comm.channel_noise import ChannelNoise
from google.protobuf.message import Message
from ark.comm.querier import Querier
from ark.comm.queriable import Queryable
from ark.time import Clock, Rate, Stepper, Time
from ark.comm.stamped_message import StampedMessage
from ark.comm.default_z_session import default_session
from ark.comm.publisher import Publisher, PeriodicPublisher
from ark.comm.subscriber import (
    Subscriber,
    TimeWindowListener,
    SampleWindowListener,
    ReadyWhen,
)


class NodeCliParser(BaseCliParser):
    """Utility class to parse command line arguments for a Node, extracting parameters and channel remappings."""

    def __init__(self):
        self._params = {}
        self._remaps = {}

    def parse(self, args):
        for arg in args:
            if ":=" in arg:
                pname, pvalue = self._parse_param(arg)
                self._params[pname] = pvalue
            elif "--" in arg:
                from_channel, to_channel = self._parse_remap(arg)
                self._remaps[from_channel] = to_channel
            else:
                raise ValueError(f"Invalid argument format: {arg}")
        return self._params, self._remaps

    def _parse_param(self, arg: str) -> tuple[str, object]:
        param_name, param_value = arg.split(":=", 1)
        try:
            param_value = ast.literal_eval(param_value)
        except (ValueError, SyntaxError):
            pass  # If it cannot be parsed as a Python literal, use the string itself
        return param_name, param_value

    def _parse_remap(self, arg: str) -> tuple[str, str]:
        from_channel, to_channel = arg.split("--", 1)
        return str(from_channel), str(to_channel)


class Node:

    def __init__(self):

        # Parse command line arguments and retrieve parameters and remappings
        self._params, self._remaps = NodeCliParser().parse(sys.argv[1:])

        # Extract basic parameters
        self._world_name = self.get_param("__world_name", required=True)
        self._channel_ns = Channel.public(self._world_name)
        self._node_name = self.get_param("__node_name", required=True)
        self._sim = bool(self.get_param("__sim", required=True))
        z_cfg_path = self.get_param("__z_cfg_path")

        # Initialize the zenoh session
        if z_cfg_path:
            z_cfg = zenoh.Config.from_json5(Path(z_cfg_path).read_text())
            self._session = zenoh.open(z_cfg)
        else:
            self._session = default_session()

        # Setup the clock
        self.clock = Clock(self._sim, self._world_name, self._session)

        # Setup publisher, subscriber, querier and queryable dictionaries
        self._publishers = {}
        self._subscribers = {}
        self._queriers = {}
        self._queryables = {}

        # Setup dictionary to store rates and steppers
        self._rates = []
        self._steppers = []

    def get_param(
        self,
        param_name: str,
        default: str | bool | int | float | None = None,
        required: bool = False,
    ) -> str | bool | int | float | None:
        """Get a parameter value by name, returning a default if the parameter is not set."""
        value = self._params.get(param_name, default)
        if required and value is None:
            raise ValueError(f"Missing required parameter: {param_name}")
        return value

    def _resolve_channel(self, channel: str | Channel) -> Channel:
        channel = self._remaps.get(str(channel), channel)  # Apply remapping if exists
        return self._channel_ns / Channel.public(channel)  # Ensure channel namespace

    def create_publisher(
        self,
        channel: str | Channel,
        noise: Callable[[Message], Message] | None = None,
    ) -> Publisher:
        channel = self._resolve_channel(channel)
        pub = Publisher(
            self._world_name, self._node_name, self._session, channel, self.clock, noise
        )
        self._publishers[channel] = pub
        return pub

    def create_periodic_publisher(
        self,
        channel: str | Channel,
        hz: float,
        message_factory: Callable[[Time], Message],
        noise: Callable[[Message], Message] | None = None,
    ) -> PeriodicPublisher:
        channel = self._resolve_channel(channel)
        pub = PeriodicPublisher(
            self._world_name,
            self._node_name,
            self._session,
            channel,
            self.clock,
            hz,
            message_factory,
            noise,
        )
        self._publishers[channel] = pub
        return pub

    def create_subscriber(
        self, channel: str | Channel, callback: Callable[[StampedMessage], None]
    ) -> Subscriber:
        channel = self._resolve_channel(channel)
        sub = Subscriber(
            self._world_name,
            self._node_name,
            self._session,
            channel,
            self.clock,
            callback,
        )
        self._subscribers[channel] = sub
        return sub

    def create_sample_window_listener(
        self,
        channel: str | Channel,
        n_buffer: int = 1,
        ready_when: ReadyWhen | str = ReadyWhen.ALWAYS,
    ) -> SampleWindowListener:
        channel = self._resolve_channel(channel)
        lr = SampleWindowListener(
            self._world_name,
            self._node_name,
            self._session,
            channel,
            self.clock,
            n_buffer,
            ready_when,
        )
        self._subscribers[channel] = lr
        return lr

    def create_time_window_listener(
        self,
        channel: str | Channel,
        window_sec: float,
    ) -> TimeWindowListener:
        channel = self._resolve_channel(channel)
        lr = TimeWindowListener(
            self._world_name,
            self._node_name,
            self._session,
            channel,
            self.clock,
            window_sec,
        )
        self._subscribers[channel] = lr
        return lr

    def create_querier(
        self,
        channel: str | Channel,
        noise: ChannelNoise | None = None,
        timeout: float = 10.0,
    ) -> Querier:
        channel = self._resolve_channel(channel)
        querier = Querier(
            self._world_name,
            self._node_name,
            self._session,
            channel,
            self.clock,
            noise,
            timeout,
        )
        self._queriers[channel] = querier
        return querier

    def create_queryable(
        self,
        channel: str | Channel,
        callback: Callable[[StampedMessage], Message | None],
        noise: ChannelNoise | None = None,
    ) -> Queryable:
        channel = self._resolve_channel(channel)
        queryable = Queryable(
            self._world_name,
            self._node_name,
            self._session,
            channel,
            self.clock,
            callback,
            noise,
        )
        self._queryables[channel] = queryable
        return queryable

    def create_rate(self, hz: float) -> Rate:
        rate = Rate(self.clock, hz)
        self._rates.append(rate)
        return rate

    def create_stepper(self, hz: float, callback) -> Stepper:
        stepper = Stepper(self.clock, hz, callback)
        self._steppers.append(stepper)
        return stepper

    def spin(self):
        while True:
            pass

    def close(self):
        for pub in self._publishers.values():
            pub.close()
        for sub in self._subscribers.values():
            sub.close()
        for querier in self._queriers.values():
            querier.close()
        for queryable in self._queryables.values():
            queryable.close()
        for stepper in self._steppers:
            stepper.close()
        self._session.close()


def main(node_cls: type[Node]):
    node = node_cls()
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
