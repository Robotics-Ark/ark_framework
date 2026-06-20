from gymnasium import ActionWrapper
from gymnasium.spaces import Dict


class FlattenActionWrapper(ActionWrapper):
    """Unwraps a single-key Dict action space to the bare inner space."""

    def __init__(self, env):
        super().__init__(env)
        space = env.action_space
        if not isinstance(space, Dict) or len(space.spaces) != 1:
            raise ValueError(
                "FlattenActionWrapper requires a Dict action space with exactly one key."
            )
        self._action_key = next(iter(space.spaces))
        self.action_space = space.spaces[self._action_key]

    def action(self, action):
        return {self._action_key: action}
