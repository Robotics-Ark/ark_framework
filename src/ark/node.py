import json
import time
import threading
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
from ark.diff.variable import Variable
from ark_msgs import VariableInfo


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
        self._grad_lock = threading.Lock()
        self._registry_pub = self.create_publisher("ark/vars/register")

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

        For "output" mode, queryables are created on "grad/{input_name}/{name}"
        for each existing input variable. Setting the tensor triggers an eager
        backward pass that caches gradients into each input variable.

        Args:
            name: Variable identifier, used in channel names.
            value: Initial scalar value for the underlying tensor.
            mode: "input" or "output".
        """
        var = Variable(name, value, mode, self._variables, self._grad_lock, self.create_queryable)
        self._variables[name] = var

        if mode == "output":
            grad_channels = [
                f"grad/{inp_name}/{name}"
                for inp_name, v in self._variables.items()
                if v.mode == "input"
            ]
            self._registry_pub.publish(VariableInfo(
                output_name=name,
                node_name=self._node_name,
                grad_channels=grad_channels,
            ))

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
