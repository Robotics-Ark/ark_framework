import ast
import sys
import json
import zenoh
from collections.abc import Callable
from google.protobuf.message import Message
from ark.time import Clock, Rate, Stepper, Time
from ark.comm import (
    Publisher,
    Subscriber,
    Querier,
    Queryable,
    Channel,
    Listener,
    PeriodicPublisher,
)


class Node:

    def __init__(self, z_cfg: dict):
        self._params = {}
        self._remaps = {}
        self._parse_cli_args()
        self._env_namespace = Channel(self.get_param("__env_namespace"))
        self._node_name = self.get_param("__node_name", type(self).__name__)
        self._session = self._init_zenoh_sesssion(z_cfg)

        # Setup the clock
        self.clock = Clock(self._sim, self._session)

        # Setup publisher, subscriber, querier and queryable dictionaries
        self._publishers = {}
        self._subscribers = {}
        self._queriers = {}
        self._queryables = {}

        # Setup dictionary to store rates and steppers
        self._rates = []
        self._steppers = []

    def _parse_cli_args(self):
        for arg in sys.argv[1:]:
            if ":=" in arg:
                param_name, param_value = arg.split(":=", 1)
                self._params[param_name] = self._parse_param_value(param_value)
            elif "--" in arg:
                from_channel, to_channel = arg.split("--", 1)
                self._remaps[from_channel] = to_channel
            else:
                raise ValueError(f"Invalid argument format: {arg}")

    def _parse_param_value(self, value: str) -> str | bool | int | float:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value  # If it cannot be parsed as a Python literal, return the string itself

    def get_param(
        self, param_name: str, default: str | bool | int | float | None = None
    ) -> str | bool | int | float | None:
        return self._params.get(param_name, default)

    def _resolve_channel(self, channel: str | Channel) -> Channel:
        channel_str = str(channel)
        return self._env_namespace / Channel(self._remaps.get(channel_str, channel_str))

    def _init_zenoh_sesssion(self, z_cfg: dict):
        _z_cfg = zenoh.Config.from_json5(json.dumps(z_cfg))
        return zenoh.open(_z_cfg)

    def create_publisher(
        self,
        channel: str | Channel,
        apply_noise: Callable[[Message], Message] | None = None,
    ) -> Publisher:
        channel = self._resolve_channel(channel)
        pub = Publisher(
            self._node_name, self._session, channel, self.clock, apply_noise=apply_noise
        )
        self._publishers[channel] = pub
        return pub

    def create_periodic_publisher(
        self,
        channel: str | Channel,
        hz: float,
        message_factory: Callable[[Time], Message],
        apply_noise: Callable[[Message], Message] | None = None,
    ) -> PeriodicPublisher:
        channel = self._resolve_channel(channel)
        pub = PeriodicPublisher(
            message_factory,
            hz,
            self._node_name,
            self._session,
            channel,
            self.clock,
            apply_noise,
        )
        self._publishers[channel] = pub
        return pub

    def create_subscriber(
        self, channel: str | Channel, callback: Callable[[Message], None]
    ) -> Subscriber:
        channel = self._resolve_channel(channel)
        sub = Subscriber(self._node_name, self._session, channel, callback)
        self._subscribers[channel] = sub
        return sub

    def create_listener(
        self, channel: str | Channel, n_buffer: int = 1, ready_when: str = "full"
    ) -> Listener:
        channel = self._resolve_channel(channel)
        lr = Listener(self._node_name, self._session, channel, n_buffer, ready_when)
        self._subscribers[channel] = lr
        return lr

    def create_querier(
        self,
        channel: str | Channel,
        apply_noise: Callable[[Message], Message] | None = None,
    ) -> Querier:
        channel = self._resolve_channel(channel)
        querier = Querier(
            self._node_name, self._session, channel, self.clock, apply_noise=apply_noise
        )
        self._queriers[channel] = querier
        return querier

    def create_queryable(
        self,
        channel: str | Channel,
        callback,
        apply_noise: Callable[[Message], Message] | None = None,
    ) -> Queryable:
        channel = self._resolve_channel(channel)
        queryable = Queryable(
            self._node_name,
            self._session,
            self.clock,
            channel,
            callback,
            apply_noise=apply_noise,
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
        stepper.start()
        return stepper

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
