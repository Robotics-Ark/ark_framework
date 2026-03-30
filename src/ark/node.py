import json
import zenoh
from collections.abc import Callable
from google.protobuf.message import Message
from ark.time import Clock, Rate, Stepper
from ark.comm import Publisher, Subscriber, Querier, Queryable, Channel, Listener


class Node:

    def __init__(self, node_name: str, z_cfg: dict):
        self._node_name = node_name
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

    def _init_zenoh_sesssion(self, z_cfg: dict):
        _z_cfg = zenoh.Config.from_json5(json.dumps(z_cfg))
        return zenoh.open(_z_cfg)

    def create_publisher(
        self, channel: Channel, apply_noise: Callable[[Message], Message] | None = None
    ) -> Publisher:
        pub = Publisher(
            self._node_name, self._session, channel, self.clock, apply_noise=apply_noise
        )
        self._publishers[channel] = pub
        return pub

    def create_subscriber(
        self, channel: Channel, callback: Callable[[Message], None]
    ) -> Subscriber:
        sub = Subscriber(self._node_name, self._session, channel, callback)
        self._subscribers[channel] = sub
        return sub

    def create_listener(
        self, channel: Channel, n_buffer: int = 1, ready_when: str = "full"
    ) -> Listener:
        lr = Listener(self._node_name, self._session, channel, n_buffer, ready_when)
        self._subscribers[channel] = lr
        return lr

    def create_querier(
        self, channel: Channel, apply_noise: Callable[[Message], Message] | None = None
    ) -> Querier:
        querier = Querier(
            self._node_name, self._session, channel, self.clock, apply_noise=apply_noise
        )
        self._queriers[channel] = querier
        return querier

    def create_queryable(
        self,
        channel: Channel,
        callback,
        apply_noise: Callable[[Message], Message] | None = None,
    ) -> Queryable:
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
