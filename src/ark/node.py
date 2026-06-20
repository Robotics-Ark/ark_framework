import ast
import sys
import zenoh
from pathlib import Path
from typing import Callable
from gymnasium import Space
from .base import Spinner
from .reset import ResetObject
from .comm.end_point import EndPoint
from .comm.publisher import Publisher
from .comm.subscriber import Subscriber
from .comm.querier import Querier
from .comm.queryable import Queryable
from .comm.queryable_space import query_space
from .parameters import ParameterServer, PARAM_TYPE
from .time import Rate, Stepper, Clock, Time
from .comm.stamped_sample import StampedSample
from .comm.zenoh_session import default_session
from .comm.channel import Channel, ChannelName
from .noise import NOISE_TYPE
from .comm.listener import NSampleListener, TSampleListener, ReadyWhen


class Node(ResetObject, Spinner):

    def __init__(
        self,
        env_name: str,
        node_name: str,
        parameters: dict[str, PARAM_TYPE],
        channel_remaps: dict[str, str],
        session: zenoh.Session,
    ):
        self._env_name = env_name
        self._node_name = node_name
        self._channel_remaps = channel_remaps
        self._session = session
        self._sim = bool(parameters.get("sim", False))
        self._param_server = ParameterServer(
            f"{self._env_name}/{self._node_name}/parameters",
            parameters,
            self._session,
        )
        self._end_points = {}
        self._steppers = []
        self._clock = Clock(self._env_name, self._session)
        ResetObject.__init__(self, env_name, session)
        Spinner.__init__(self)

    def reset(self, _seed: int | None = None):
        """Reset the node's state."""
        for stepper in self._steppers:
            stepper.reset()

    def _noise(self, noise: NOISE_TYPE) -> NOISE_TYPE:
        """Strip noise when running on real hardware.

        Real sensors and actuators already have physical noise; adding
        simulated noise on top would corrupt the measurements.
        """
        return noise if self._sim else None

    def _add_end_point(self, channel_name: ChannelName | str, end_point: EndPoint):
        if channel_name in self._end_points:
            raise ValueError(f"Channel '{channel_name}' already exists")
        self._end_points[channel_name] = end_point

    def get_parameter(
        self, param_name: str, default: PARAM_TYPE | None = None
    ) -> PARAM_TYPE:
        try:
            return self._param_server.get(param_name)
        except KeyError:
            if default is None:
                raise
            self._param_server.set(param_name, default)
            return default

    def _resolve_channel_name(self, channel_name: ChannelName | str) -> ChannelName:
        if isinstance(channel_name, str):
            channel_name = ChannelName(channel_name)
        remapped_name = self._channel_remaps.get(str(channel_name), str(channel_name))
        return ChannelName(remapped_name)

    def _resolve_channel(self, channel_name: ChannelName | str) -> Channel:
        resolved_name = self._resolve_channel_name(channel_name)
        return Channel(resolved_name, self._env_name)

    def query_space(self, channel_name: ChannelName | str, role: str) -> Space:
        channel = self._resolve_channel(channel_name)
        return query_space(channel.full_name, role, self._session)

    def create_publisher(
        self,
        channel_name: ChannelName | str,
        space: Space,
        check: bool = False,
        noise: NOISE_TYPE = None,
    ) -> Publisher:
        pub = Publisher(
            self._resolve_channel(channel_name),
            space,
            self._session,
            check,
            self._noise(noise),
        )
        self._add_end_point(channel_name, pub)
        return pub

    def create_subscriber(
        self,
        channel_name: ChannelName | str,
        space: Space,
        callback: Callable[[StampedSample], None],
        check: bool = False,
        noise: NOISE_TYPE = None,
    ) -> Subscriber:
        sub = Subscriber(
            self._resolve_channel(channel_name),
            space,
            callback,
            self._session,
            check,
            self._noise(noise),
        )
        self._add_end_point(channel_name, sub)
        return sub

    def create_n_sample_listener(
        self,
        channel_name: ChannelName | str,
        space: Space,
        n: int,
        check: bool = False,
        noise: NOISE_TYPE = None,
        ready_when: ReadyWhen = ReadyWhen.ALWAYS,
    ) -> NSampleListener:
        listener = NSampleListener(
            n,
            self._resolve_channel(channel_name),
            space,
            self._session,
            check,
            self._noise(noise),
            ready_when,
        )
        self._add_end_point(channel_name, listener)
        return listener

    def create_t_sample_listener(
        self,
        channel_name: ChannelName | str,
        space: Space,
        t: float,
        check: bool = False,
        noise: NOISE_TYPE = None,
        ready_when: ReadyWhen = ReadyWhen.ALWAYS,
    ) -> TSampleListener:
        listener = TSampleListener(
            t,
            self._resolve_channel(channel_name),
            space,
            self._session,
            check,
            self._noise(noise),
            ready_when,
        )
        self._add_end_point(channel_name, listener)
        return listener

    def create_queryable(
        self,
        channel_name: ChannelName | str,
        req_space: Space,
        res_space: Space,
        callback: Callable,
        check_req: bool = False,
        check_res: bool = False,
        req_noise: NOISE_TYPE = None,
        res_noise: NOISE_TYPE = None,
    ) -> Queryable:
        queryable = Queryable(
            self._resolve_channel(channel_name),
            req_space,
            res_space,
            callback,
            self._session,
            check_req,
            check_res,
            self._noise(req_noise),
            self._noise(res_noise),
        )
        self._add_end_point(channel_name, queryable)
        return queryable

    def create_querier(
        self,
        channel_name: ChannelName | str,
        req_space: Space,
        res_space: Space,
        check_req: bool = False,
        check_res: bool = False,
        req_noise: NOISE_TYPE = None,
        res_noise: NOISE_TYPE = None,
    ) -> Querier:
        querier = Querier(
            self._resolve_channel(channel_name),
            req_space,
            res_space,
            self._session,
            check_req,
            check_res,
            self._noise(req_noise),
            self._noise(res_noise),
        )
        self._add_end_point(channel_name, querier)
        return querier

    def create_rate(self, hz: float) -> Rate:
        return Rate(self._clock, hz)

    def create_stepper(self, hz: float, callback: Callable[[Time], None]) -> Stepper:
        stepper = Stepper(self._clock, hz, callback)
        self._steppers.append(stepper)
        return stepper

    def now(self) -> Time:
        return self._clock.now()

    def close(self):
        self.stop_spinning()
        for ep in self._end_points.values():
            ep.close()
        for s in self._steppers:
            s.close()
        self._end_points.clear()
        self._param_server.close()
        self._session.close()


class NodeArgumentParser:

    def __init__(self, args: list[str]):
        self._args = args[1:]  # skip the script name

    def parse(
        self,
    ) -> tuple[str, str, dict[str, PARAM_TYPE], dict[str, str], zenoh.Session]:
        parameters: dict[str, PARAM_TYPE] = {}
        channel_remaps: dict[str, str] = {}

        for arg in self._args:
            if ":=" in arg:
                name, value = self._parse_param(arg)
                parameters[name] = value
            elif "--" in arg:
                from_ch, to_ch = self._parse_remap(arg)
                channel_remaps[from_ch] = to_ch
            else:
                raise ValueError(
                    f"Invalid argument {arg!r}. Use 'name:=value' for parameters or 'from--to' for remaps."
                )

        if "env_name" not in parameters:
            raise ValueError(
                "Required parameter 'env_name' not provided. Use env_name:=<value>"
            )
        if "node_name" not in parameters:
            raise ValueError(
                "Required parameter 'node_name' not provided. Use node_name:=<value>"
            )

        env_name = str(parameters.pop("env_name"))
        node_name = str(parameters.pop("node_name"))

        z_config_path = parameters.pop("z_config_path", None)
        if z_config_path is not None:
            z_cfg = zenoh.Config.from_json5(Path(str(z_config_path)).read_text())
            session = zenoh.open(z_cfg)
        else:
            session = default_session()

        return env_name, node_name, parameters, channel_remaps, session

    def _parse_param(self, arg: str) -> tuple[str, PARAM_TYPE]:
        name, value_str = arg.split(":=", 1)
        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            value = value_str
        return name, value

    def _parse_remap(self, arg: str) -> tuple[str, str]:
        from_ch, to_ch = arg.split("--", 1)
        return from_ch, to_ch


def main(node_cls: type[Node]):
    parser = NodeArgumentParser(sys.argv)
    env_name, node_name, parameters, channel_remaps, session = parser.parse()
    node = node_cls(env_name, node_name, parameters, channel_remaps, session)
    try:
        node.spin()
    finally:
        node.close()
