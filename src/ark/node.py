import ast
import sys
import signal
import threading
import zenoh
from pathlib import Path
from typing import Callable
from gymnasium import Space
from .comm.end_point import EndPoint
from .comm.publisher import Publisher
from .comm.subscriber import Subscriber
from .comm.listener import NSampleListener, TSampleListener
from .comm.querier import Querier
from .comm.queryable import Queryable
from .reset import ResetObject
from .parameters import ParameterServer, PARAM_TYPE
from .time import Rate, Stepper, Clock, Time
from .comm.stamped_sample import StampedSample
from .comm.default_z_session import default_session
from .comm.channel import Channel, ChannelName, ChannelNoise


class Node(ResetObject):

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
        self._param_server = ParameterServer(
            f"{self._env_name}/{self._node_name}/parameters",
            parameters,
            self._session,
        )
        self._end_points = {}
        self._rates = []
        self._steppers = []
        self._clock = Clock(self._env_name, self._session)
        self._stop_event = threading.Event()

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

    def create_publisher(
        self,
        channel_name: ChannelName | str,
        space: Space,
        check: bool = False,
        noise: ChannelNoise | None = None,
    ) -> Publisher:
        pub = Publisher(
            self._resolve_channel(channel_name), space, self._session, check, noise
        )
        self._add_end_point(channel_name, pub)
        return pub

    def create_subscriber(
        self,
        channel_name: ChannelName | str,
        space: Space,
        callback: Callable[[StampedSample], None],
        check: bool = False,
        noise: ChannelNoise | None = None,
    ) -> Subscriber:
        sub = Subscriber(
            self._resolve_channel(channel_name),
            space,
            callback,
            self._session,
            check,
            noise,
        )
        self._add_end_point(channel_name, sub)
        return sub

    def create_n_sample_listener(
        self,
        channel_name: ChannelName | str,
        space: Space,
        n: int,
        check: bool = False,
        noise: ChannelNoise | None = None,
    ) -> NSampleListener:
        listener = NSampleListener(
            n,
            self._resolve_channel(channel_name),
            space,
            self._session,
            check,
            noise,
        )
        self._add_end_point(channel_name, listener)
        return listener

    def create_t_sample_listener(
        self,
        channel_name: ChannelName | str,
        space: Space,
        t: float,
        check: bool = False,
        noise: ChannelNoise | None = None,
    ) -> TSampleListener:
        listener = TSampleListener(
            t,
            self._resolve_channel(channel_name),
            space,
            self._session,
            check,
            noise,
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
        req_noise: ChannelNoise | None = None,
        res_noise: ChannelNoise | None = None,
    ) -> Queryable:
        queryable = Queryable(
            self._resolve_channel(channel_name),
            req_space,
            res_space,
            callback,
            self._session,
            check_req,
            check_res,
            req_noise,
            res_noise,
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
        req_noise: ChannelNoise | None = None,
        res_noise: ChannelNoise | None = None,
    ) -> Querier:
        querier = Querier(
            self._resolve_channel(channel_name),
            req_space,
            res_space,
            self._session,
            check_req,
            check_res,
            req_noise,
            res_noise,
        )
        self._add_end_point(channel_name, querier)
        return querier

    def create_rate(self, hz: float) -> Rate:
        rate = Rate(self._clock, hz)
        self._rates.append(rate)
        return rate

    def create_stepper(self, hz: float, callback: Callable[[Time], None]) -> Stepper:
        stepper = Stepper(self._clock, hz, callback)
        self._steppers.append(stepper)
        return stepper

    def spin(self):
        signal.signal(signal.SIGINT, lambda *_: self._stop_event.set())
        signal.signal(signal.SIGTERM, lambda *_: self._stop_event.set())
        self._stop_event.wait()

    def close(self):
        self._stop_event.set()
        super().close()
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
