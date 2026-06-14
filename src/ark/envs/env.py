import zenoh
from typing import Any
from gymnasium import Env
from enum import IntEnum
from ark.node import Node
from dataclasses import dataclass
from ark.parameters import PARAM_TYPE
from ark.reset import ResetCoordinator
from gymnasium.spaces import Dict, Sequence
from ark.comm.channel import ChannelName, NOISE_TYPE
from ark.comm.listener import ReadyWhen, NSampleListener, TSampleListener


class ListenerType(IntEnum):
    NSAMPLE = 1
    TSAMPLE = 2


@dataclass
class InboundChannelSpec(frozen=True, slots=True):
    channel_name: ChannelName | str
    window_length: int | float = 1
    listener_type: ListenerType = ListenerType.NSAMPLE
    check: bool = False
    noise: NOISE_TYPE = None
    ready_when: ReadyWhen = ReadyWhen.ALWAYS

    def init_listener(self, node: Node) -> NSampleListener | TSampleListener:
        space = node.query_space(self.channel_name, "publisher")
        if self.listener_type is ListenerType.NSAMPLE:
            if not isinstance(self.window_length, int):
                raise ValueError(
                    f"Invalid window_length: {self.window_length}. Must be an int for NSampleListener."
                )
            return node.create_n_sample_listener(
                self.channel_name,
                space,
                self.window_length,
                self.check,
                self.noise,
                self.ready_when,
            )
        elif self.listener_type is ListenerType.TSAMPLE:
            if not isinstance(self.window_length, float):
                raise ValueError(
                    f"Invalid window_length: {self.window_length}. Must be a float for TSampleListener."
                )
            return node.create_t_sample_listener(
                self.channel_name,
                space,
                self.window_length,
                self.check,
                self.noise,
                self.ready_when,
            )
        else:
            raise ValueError(f"Invalid listener_type: {self.listener_type}")


@dataclass
class ActionChannelSpec(frozen=True, slots=True):
    channel_name: ChannelName | str
    check: bool = False
    noise: NOISE_TYPE = None

    def init_publisher(self, node: Node):
        space = node.query_space(self.channel_name, "subscriber")
        return node.create_publisher(self.channel_name, space, self.check, self.noise)


INBOUND_CHANNEL_TYPE = str | InboundChannelSpec | list[str] | list[InboundChannelSpec]
ACTION_CHANNEL_TYPE = str | ActionChannelSpec | list[str] | list[ActionChannelSpec]


def ensure_inbound_channels(channels: INBOUND_CHANNEL_TYPE) -> list[InboundChannelSpec]:

    if not channels:
        return []

    if isinstance(channels, str):
        channels = [channels]

    if isinstance(channels, InboundChannelSpec):
        channels = [channels]

    # Check all elements are of the same type
    all_str = all(isinstance(ch, str) for ch in channels)
    all_spec = all(isinstance(ch, InboundChannelSpec) for ch in channels)
    if not (all_str or all_spec):
        raise ValueError(
            "All elements of channels must be either str or InboundChannelSpec."
        )

    if all_str:
        return [InboundChannelSpec(ch) for ch in channels]
    else:
        return channels


def ensure_action_channels(channels: ACTION_CHANNEL_TYPE) -> list[ActionChannelSpec]:

    if not channels:
        raise ValueError("action channels cannot be empty.")

    if isinstance(channels, str):
        channels = [channels]

    all_str = all(isinstance(ch, str) for ch in channels)
    all_spec = all(isinstance(ch, ActionChannelSpec) for ch in channels)
    if not (all_str or all_spec):
        raise ValueError(
            "All elements of action_channels must be either str or ActionChannelSpec."
        )

    if all_str:
        return [ActionChannelSpec(ch) for ch in channels]
    else:
        return channels


class ArkEnv(Env):

    def __init__(
        self,
        env_name: str,
        parameters: dict[str, PARAM_TYPE],
        step_duration: float | None,
        session: zenoh.Session,
        observation_channels: INBOUND_CHANNEL_TYPE,
        action_channels: ACTION_CHANNEL_TYPE,
        state_channels: INBOUND_CHANNEL_TYPE = [],
    ):
        super().__init__()
        try:
            self._sim = parameters["sim"]
        except KeyError:
            raise ValueError("Missing 'sim' parameter in environment parameters")
        self._env_name = env_name
        self._step_hz = (
            1.0 / float(step_duration) if step_duration is not None else None
        )
        self._session = session
        self._node = Node(
            env_name,
            "env",
            parameters,
            {},  # no channel remaps for env node
            self._session,
        )
        self._rate = None
        self._reset_coordinator = ResetCoordinator(self._env_name, self._session)

        # Setup inbound channels, spaces, and listeners
        (
            self._observation_channels,
            self.observation_space,
            self._observation_listeners,
        ) = self._init_inbound_channels(observation_channels)
        (
            self._state_channels,
            self.state_space,
            self._state_listeners,
        ) = self._init_inbound_channels(state_channels)

        # Setup action channels, space, publishers
        self._action_channels: list[ActionChannelSpec] = ensure_action_channels(
            action_channels
        )
        self.action_space = Dict(
            {
                ch_name.channel_name: self._node.query_space(
                    ch_name.channel_name, "publisher"
                )
                for ch_name in self._action_channels
            }
        )
        self._action_publishers = {
            ch.channel_name: ch.init_publisher(self._node)
            for ch in self._action_channels
        }

    def _init_inbound_channels(
        self, channels: INBOUND_CHANNEL_TYPE
    ) -> tuple[
        list[InboundChannelSpec], Dict, dict[str, NSampleListener | TSampleListener]
    ]:
        channels = ensure_inbound_channels(channels)
        query_space = lambda ch: self._node.query_space(ch.channel_name, "publisher")
        # NOTE: listeners are used for inbound channels, so each
        # subspace of Dict is Sequence(query_space(ch))
        space = Dict({ch.channel_name: Sequence(query_space(ch)) for ch in channels})
        listeners = {ch.channel_name: ch.init_listener(self._node) for ch in channels}
        return channels, space, listeners

    def _get_observation(self, channel_name: str) -> tuple:
        return self._observation_listeners[channel_name].get_window()

    def get_observation(self) -> dict[str, tuple]:
        return {
            ch.channel_name: self._get_observation(ch.channel_name)
            for ch in self._observation_channels
        }

    def _get_state(self, channel_name: str) -> tuple:
        return self._state_listeners[channel_name].get_window()

    def get_state(self) -> dict[str, tuple]:
        return {
            ch.channel_name: self._get_state(ch.channel_name)
            for ch in self._state_channels
        }

    def get_reward(self) -> float:
        return 0.0

    def get_terminated(self) -> bool:
        return False

    def get_truncated(self) -> bool:
        return False

    def get_info(self) -> dict:
        return {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)
        self._reset_coordinator.reset(seed)
        if self._step_hz:
            self._rate = self._node.create_rate(self._step_hz)
        if isinstance(seed, int):
            self.observation_space.seed(seed)
            self.action_space.seed(seed)
            self.state_space.seed(seed)
        return self.get_observation(), self.get_info()

    def _apply_action(self, action: dict[str, Any]):
        for ch_name in self.action_space.keys():
            try:
                act = action[ch_name]
            except KeyError:
                raise KeyError(
                    f"Action for channel '{ch_name}' is missing in the given action dict."
                )
            self._action_publishers[ch_name].publish(act)

    def step(self, action: dict[str, Any]):
        self._apply_action(action)
        if self._rate:
            self._rate.sleep()
        return (
            self.get_observation(),
            self.get_reward(),
            self.get_terminated(),
            self.get_truncated(),
            self.get_info(),
        )

    def close(self):
        self._reset_coordinator.close()
        self._node.close()
        self._session.close()
