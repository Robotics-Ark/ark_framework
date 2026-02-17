import json
import time
import threading
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

_BACKWARD_LOCK = threading.Lock()


class Variable:

    def __init__(self, name, value, mode="input", out_fields=None):
        self.name = name
        self.mode = mode
        self.out_fields = out_fields or []
        self.tensor = torch.tensor(value, requires_grad=True)
        self._outputs = {}

    def set_output(self, field, tensor):
        with _BACKWARD_LOCK:
            self._outputs[field] = tensor

    def set_outputs(self, mapping):
        with _BACKWARD_LOCK:
            self._outputs.update(mapping)

    def compute_grad(self, field):
        with _BACKWARD_LOCK:
            out_tensor = self._outputs.get(field)
            if out_tensor is None:
                return 0.0, 0.0
            val = float(out_tensor.detach())
            if self.tensor.grad is not None:
                self.tensor.grad.zero_()
            out_tensor.backward(retain_graph=True)
            grad = float(self.tensor.grad) if self.tensor.grad is not None else 0.0
            return val, grad


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
        """Create a differentiable variable with automatic gradient queryables.

        For "input" mode variables with out_fields, this sets up:
          - A queryable on "grad/{name}/{field}" for each field in out_fields.
            Gradients are computed lazily via backward() when queried, using
            output tensors registered by the user via Variable.set_outputs().
          - A subscriber on "param/{name}" that updates the variable's tensor
            value when a new parameter is published.

        Args:
            name: Variable identifier, used in channel names.
            value: Initial scalar value for the underlying tensor.
            mode: "input" creates queryables and a param subscriber.
            out_fields: Output field names (e.g. ["x", "y"]) that this
                variable contributes to. Each gets a gradient queryable.
        """
        var = Variable(name, value, mode, out_fields)
        self._variables[name] = var

        if mode == "input":
            # Create a gradient queryable for each output field.
            # On query, compute_grad() runs backward() on the registered
            # output tensor and returns the gradient w.r.t. this variable.
            if var.out_fields:
                for field in var.out_fields:
                    grad_channel = f"grad/{name}/{field}"

                    def _make_handler(v, fld):
                        def handler(_req):
                            val, grad = v.compute_grad(fld)
                            return Value(val=val, grad=grad)

                        return handler

                    self.create_queryable(grad_channel, _make_handler(var, field))

            # Subscribe to parameter updates so external nodes can set
            # this variable's value at runtime.
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
