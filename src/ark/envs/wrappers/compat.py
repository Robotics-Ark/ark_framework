from gymnasium import Wrapper
from gymnasium.spaces import Dict

from ark.envs.env import ListenerType
from ark.envs.wrappers.action import FlattenActionWrapper
from ark.envs.wrappers.observation import (
    SqueezeWindowObservationWrapper,
    StackWindowObservationWrapper,
)


class SB3CompatWrapper(Wrapper):
    """Adapts ArkEnv for StableBaselines3 compatibility.

    Applies, as appropriate:
    - SqueezeWindowObservationWrapper (window_length=1 NSampleListener channels)
    - StackWindowObservationWrapper   (window_length>1 NSampleListener channels)
    - FlattenActionWrapper            (single-key Dict action space)

    TSampleListener channels are left unchanged since their window size is not fixed.
    """

    def __init__(self, env):
        obs_channels = env.unwrapped._observation_channels
        all_nsample = all(
            ch.listener_type is ListenerType.NSAMPLE for ch in obs_channels
        )
        if all_nsample and obs_channels:
            if all(ch.window_length == 1 for ch in obs_channels):
                env = SqueezeWindowObservationWrapper(env)
            else:
                env = StackWindowObservationWrapper(env)

        if isinstance(env.action_space, Dict) and len(env.action_space.spaces) == 1:
            env = FlattenActionWrapper(env)

        super().__init__(env)
