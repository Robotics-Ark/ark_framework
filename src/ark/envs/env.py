import zenoh
import numpy as np
from typing import Any
from gymnasium import Env
from dataclasses import dataclass
from abc import ABC, abstractmethod
from ark.comm.channel import Channel
from ark.reset import ResetableContainer
from ark.envs.spaces.channel_space import (
    InboundChannelSpec,
    InboundChannels,
    OutboundChannels,
)
from ark.time import Rate, Clock

ObservationChannelSpec = InboundChannelSpec
StateChannelSpec = InboundChannelSpec


@dataclass
class Transition:
    state: Any
    action: Any
    next_state: Any


class ArkEnv(Env, ABC):

    node_name = "env"

    def __init__(
        self,
        world_name: str,
        session: zenoh.Session,
        clock: Clock,
        observation_channels_specs: list[ObservationChannelSpec],
        action_channels: list[str | Channel],
        check_action_space: bool = False,
        state_channels_specs: list[StateChannelSpec] | None = None,
        seed: dict | int | np.random.Generator | None = None,
    ):
        super().__init__()
        self._session = session
        self.action_space = OutboundChannels(
            action_channels,
            session,
            clock,
            self.node_name,
            check_action_space,
            seed=seed,
        )
        self.observation_space = InboundChannels(
            observation_channels_specs,
            session,
            clock,
            seed=seed,
        )
        if state_channels_specs is not None:
            self.state_space = InboundChannels(
                state_channels_specs,
                session,
                clock,
                seed=seed,
            )
        else:
            self.state_space = None

        self._reset_container = ResetableContainer(world_name, self._session, clock)
        self._transition: Transition | None = None

    def get_state(self):
        if self.state_space:
            return self.state_space.get()
        else:
            raise RuntimeError("State space not defined for this environment.")

    def get_obs(self):
        return self.observation_space.get()

    def get_reward(self) -> float:
        return 0.0

    def get_terminated(self) -> bool:
        return False

    def get_truncated(self) -> bool:
        return False

    def get_info(self) -> dict:
        return {}

    def reset(
        self,
        *,
        seed: dict | int | np.random.Generator | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed, options=options)
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
            if self.state_space is not None:
                self.state_space.seed(seed)
        self._reset_container.reset(seed)
        obs = self.get_obs()
        info = self.get_info()
        return obs, info

    @abstractmethod
    def _handle_step(self): ...

    def step(self, action):
        state = self._get_state()
        self.action_space.publish(action)
        self._handle_step()
        next_state = self._get_state()

        # store the transition for use in reward, terminated, truncated, and info calculations
        self._transition = Transition(state, action, next_state)

        obs = self.get_obs()
        reward = self.get_reward(state, action, next_state)
        terminated = self.get_terminated(state, action, next_state)
        truncated = self.get_truncated()
        info = self.get_info()
        return obs, reward, terminated, truncated, info


class ControllerArkEnv(ArkEnv):

    def __init__(
        self,
        world_name: str,
        session: zenoh.Session,
        clock: Clock,
        observation_channels_specs: list[ObservationChannelSpec],
        action_channels: list[str | Channel],
        hz: float,
        check_action_space: bool = False,
        state_channels_specs: list[StateChannelSpec] | None = None,
        seed: dict | int | np.random.Generator | None = None,
    ):
        super().__init__(
            world_name,
            session,
            clock,
            observation_channels_specs,
            action_channels,
            check_action_space,
            state_channels_specs,
            seed,
        )
        self._hz = hz
        self._rate = None

    def reset(
        self,
        seed: dict | int | np.random.Generator | None = None,
        options: dict[str, Any] | None = None,
    ):
        self._rate = Rate(self._clock, self._hz)
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def _handle_step(self):
        self._rate.sleep()
