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

    def __init__(self, name, value, mode, variables_registry, create_queryable_fn):
        self.name = name
        self.mode = mode
        self._variables_registry = variables_registry
        self._grads = {}  # input vars: {output_name: grad_value}

        if mode == "input":
            self._tensor = torch.tensor(value, requires_grad=True)
        else:
            self._tensor = None
            for inp_name, inp_var in variables_registry.items():
                if inp_var.mode == "input":
                    grad_channel = f"grad/{inp_name}/{name}"

                    def _make_handler(iv, ov_name, reg):
                        def handler(_req):
                            out_var = reg.get(ov_name)
                            val = float(out_var._tensor.detach()) if out_var and out_var._tensor is not None else 0.0
                            grad = iv._grads.get(ov_name, 0.0)
                            return Value(val=val, grad=grad)
                        return handler

                    create_queryable_fn(grad_channel, _make_handler(inp_var, name, variables_registry))

    @property
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, value):
        if self.mode == "output":
            self._tensor = value
            self._compute_and_store_grads()
        else:
            self._tensor.data = value.data if isinstance(value, torch.Tensor) else torch.tensor(value)

    def _compute_and_store_grads(self):
        if self._tensor is None or not self._tensor.requires_grad:
            return
        with _BACKWARD_LOCK:
            for var in self._variables_registry.values():
                if var.mode == "input" and var._tensor.grad is not None:
                    var._tensor.grad.zero_()
            self._tensor.backward(retain_graph=True)
            for var in self._variables_registry.values():
                if var.mode == "input":
                    grad = float(var._tensor.grad) if var._tensor.grad is not None else 0.0
                    var._grads[self.name] = grad


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

    def create_variable(self, name, value, mode="input"):
        """Create a differentiable variable.

        For "input" mode, a subscriber on "param/{name}" is created so that
        external nodes can update the tensor value at runtime.

        For "output" mode, queryables are created on "grad/{input_name}/{name}"
        for each existing input variable. Setting the tensor triggers an eager
        backward pass that caches gradients into each input variable.

        Args:
            name: Variable identifier, used in channel names.
            value: Initial scalar value for the underlying tensor.
            mode: "input" or "output".
        """
        var = Variable(name, value, mode, self._variables, self.create_queryable)
        self._variables[name] = var

        if mode == "input":
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
