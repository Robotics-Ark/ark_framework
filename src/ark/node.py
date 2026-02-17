import json
import time
import torch
import zenoh
from ark.time.clock import Clock
from ark.time.rate import Rate
from ark.time.stepper import Stepper
from ark.comm.publisher import Publisher
from ark.comm.subscriber import Subscriber
from ark.comm.querier import Querier
from ark.comm.queriable import Queryable
from ark.data.data_collector import DataCollector
from ark.core.registerable import Registerable
from ark_msgs import Value


class Variable:

    def __init__(self, name, value, mode="input", out_fields=None):
        self.name = name
        self.mode = mode
        self.out_fields = out_fields or []
        self.tensor = torch.tensor(value, requires_grad=True)
        self.gradients = {f: 0.0 for f in self.out_fields}
        self.values = {f: 0.0 for f in self.out_fields}

    def update_gradients(self, grad_dict):
        self.gradients.update(grad_dict)


class BaseNode(Registerable):

    def __init__(
        self,
        env_name: str,
        node_name: str,
        z_cfg: dict,
        sim: bool = False,
        collect_data: bool = False,
    ):
        # self._z_cfg = zenoh.Config.from_json5(json.dumps(z_cfg))
        self._z_cfg = z_cfg
        self._session = zenoh.open(self._z_cfg)
        self._env_name = env_name
        self._node_name = node_name
        self._collect_data = collect_data
        self._data_collector = DataCollector(node_name) if collect_data else None
        self._clock = Clock(self._session, sim, "clock")
        self.core_registration()
        self._rates = []
        self._steppers = []
        self._pubs = {}
        self._subs = {}
        self._queriers = {}
        self._queriables = {}
        self._variables = {}

        self._session.declare_subscriber(f"{env_name}/reset", self._on_reset)

    def _on_reset(self, sample: zenoh.Sample):
        self.reset()

    def reset(self):
        pass  # can be overridden by subclasses if required

    def core_registration(self):
        print(".. todo: register node with ark core..")

    def create_publisher(self, channel) -> Publisher:
        pub = Publisher(
            self._node_name,
            self._session,
            self._clock,
            channel,
            self._data_collector,
        )
        pub.core_registration()
        self._pubs[channel] = pub
        return pub

    def create_subscriber(self, channel, callback) -> Subscriber:
        sub = Subscriber(
            self._node_name,
            self._session,
            self._clock,
            channel,
            self._data_collector,
            callback,
        )
        sub.core_registration()
        self._subs[channel] = sub
        return sub

    def create_querier(self, channel, target, timeout=10.0) -> Querier:
        querier = Querier(
            self._node_name,
            self._session,
            target,
            self._clock,
            channel,
            self._data_collector,
            # timeout,
        )
        querier.core_registration()
        self._queriers[channel] = querier
        # print session and channelinfo for debugging
        return querier

    def create_queryable(self, channel, handler) -> Queryable:
        queryable = Queryable(
            self._node_name,
            self._session,
            self._clock,
            channel,
            handler,
            self._data_collector,
        )
        queryable.core_registration()
        self._queriables[channel] = queryable
        return queryable

    def create_variable(self, name, value, mode="input", out_fields=None):
        var = Variable(name, value, mode, out_fields)
        self._variables[name] = var

        if mode == "input":
            if var.out_fields:
                for field in var.out_fields:
                    grad_channel = f"grad/{name}/{field}"

                    def _make_handler(v, fld):
                        def handler(_req):
                            return Value(
                                val=v.values.get(fld, 0.0),
                                grad=v.gradients.get(fld, 0.0),
                            )

                        return handler

                    self.create_queryable(grad_channel, _make_handler(var, field))

            def _make_sub_callback(v):
                def callback(msg):
                    v.tensor.data = torch.tensor(msg.val)

                return callback

            self.create_subscriber(f"param/{name}", _make_sub_callback(var))

        return var

    def create_rate(self, hz: float):
        rate = Rate(self._clock, hz)
        self._rates.append(rate)
        return rate

    def create_stepper(self, hz: float, callback) -> Stepper:
        stepper = Stepper(self._clock, hz, callback)
        self._steppers.append(stepper)
        return stepper

    def spin(self):
        while True:
            time.sleep(1.0)

    def close(self):
        closable_objs = (
            self._steppers
            + list(self._pubs.values())
            + list(self._subs.values())
            + list(self._queriers.values())
            + list(self._queriables.values())
        )
        for obj in closable_objs:
            obj.close()

        self._session.close()

        if self._data_collector:
            self._data_collector.close()
