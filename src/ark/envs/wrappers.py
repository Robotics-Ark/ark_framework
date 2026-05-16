from gymnasium import Wrapper


class TerminalPauseWrapper(Wrapper):

    def _pause(self):
        input("Press Enter to continue...")


class TerminalPreResetPauseWrapper(TerminalPauseWrapper):
    """A wrapper that adds a pause before resetting the environment, allowing the user to prepare for the next episode."""

    def reset(self, **kwargs):
        self._pause()
        obs, info = self.env.reset(**kwargs)
        return obs, info


class TerminalPostResetPauseWrapper(TerminalPauseWrapper):
    """A wrapper that adds a pause after resetting the environment, allowing the user to prepare for the next episode."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._pause()
        return obs, info
