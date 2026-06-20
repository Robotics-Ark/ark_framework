import zenoh
import numpy as np
from typing import Any
from abc import abstractmethod
from ark.reset import ResetObject
from ark.time import Clock, Sleep, Time


class ChannelNoise(ResetObject):

    def __init__(self, env_name: str, session: zenoh.Session):
        super().__init__(env_name, session)
        self._rng = None

    def reset(self, seed: int | None = None):
        if isinstance(seed, int) or seed is None:
            self._rng = np.random.default_rng(seed)

    @abstractmethod
    def apply(self, sample: Any) -> Any:
        """Apply noise to a sample drawn from this channel's Gymnasium space."""


class NoNoise:

    def apply(self, sample: Any) -> Any:
        """Return the sample unchanged."""
        return sample


class BoxGaussianNoise(ChannelNoise):

    def __init__(
        self,
        env_name: str,
        session: zenoh.Session,
        loc: float | np.ndarray[float] = 0.0,
        scale: float | np.ndarray[float] = 1.0,
    ):
        super().__init__(env_name, session)
        self.loc = loc
        self.scale = scale

    def apply(self, sample: np.ndarray):
        noise = self._rng.normal(loc=self.loc, scale=self.scale, size=sample.shape)
        return sample + noise


class LatencyNoise(ChannelNoise):
    """Simulates network latency by sleeping before delivering a sample.

    Uses ark's Sleep (not time.sleep) so it respects simulated time during RL.
    std > 0 enables randomised latency for domain randomisation.
    """

    def __init__(
        self,
        env_name: str,
        session: zenoh.Session,
        mean: float,
        std: float = 0.0,
    ):
        super().__init__(env_name, session)
        self._mean = mean
        self._std = std
        self._sleep = Sleep(Clock(env_name, session))

    def apply(self, sample: Any) -> Any:
        if self._std > 0.0:
            dur = max(0.0, float(self._rng.normal(self._mean, self._std)))
        else:
            dur = self._mean
        self._sleep(Time.from_sec(dur))
        return sample


NOISE_TYPE = list[ChannelNoise | NoNoise] | ChannelNoise | NoNoise | None


def normalise_noise(noise: NOISE_TYPE) -> list[ChannelNoise | NoNoise]:
    if noise is None:
        return []
    if isinstance(noise, (ChannelNoise, NoNoise)):
        return [noise]
    return list(noise)
