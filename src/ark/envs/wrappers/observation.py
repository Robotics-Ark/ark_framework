import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Dict

from ark.envs.env import ListenerType


class SqueezeWindowObservationWrapper(ObservationWrapper):
    """Removes the window tuple for channels with NSampleListener and window_length=1.

    Converts each Sequence(space) observation channel to its bare inner space,
    unwrapping the single-element tuple returned by get_window().
    """

    def __init__(self, env):
        super().__init__(env)
        channels = env.unwrapped._observation_channels
        if not all(
            ch.listener_type is ListenerType.NSAMPLE and ch.window_length == 1
            for ch in channels
        ):
            raise ValueError(
                "SqueezeWindowObservationWrapper requires all channels to be "
                "NSampleListener with window_length=1."
            )
        self.observation_space = Dict(
            {
                ch.channel_name: env.observation_space[ch.channel_name].feature_space
                for ch in channels
            }
        )

    def observation(self, observation):
        return {k: v[0] for k, v in observation.items()}


class StackWindowObservationWrapper(ObservationWrapper):
    """Converts fixed-length Sequence(Box) observation channels to stacked Box spaces.

    Requires all channels to use NSampleListener (fixed window size) and have
    Box feature spaces. Each channel's window tuple is stacked into a single
    array of shape (window_length, *inner_shape).
    """

    def __init__(self, env):
        super().__init__(env)
        channels = env.unwrapped._observation_channels
        if not all(ch.listener_type is ListenerType.NSAMPLE for ch in channels):
            raise ValueError(
                "StackWindowObservationWrapper only supports NSampleListener channels."
            )
        new_spaces = {}
        for ch in channels:
            inner = env.observation_space[ch.channel_name].feature_space
            if not isinstance(inner, Box):
                raise TypeError(
                    f"StackWindowObservationWrapper requires Box feature space; "
                    f"channel '{ch.channel_name}' has {type(inner).__name__}."
                )
            n = ch.window_length
            new_shape = (n,) + inner.shape
            new_spaces[ch.channel_name] = Box(
                low=np.broadcast_to(inner.low, new_shape).copy(),
                high=np.broadcast_to(inner.high, new_shape).copy(),
                dtype=inner.dtype,
            )
        self.observation_space = Dict(new_spaces)

    def observation(self, observation):
        return {k: np.stack(v) for k, v in observation.items()}
