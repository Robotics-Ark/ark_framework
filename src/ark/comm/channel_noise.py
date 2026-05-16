import zenoh
import numpy as np
from typing import Any
from abc import abstractmethod
from ark.reset import ResetableObject


class ChannelNoise(ResetableObject):

    def __init__(
        self, world_name: str, session: zenoh.Session, seed: int | None = None
    ):
        super().__init__(world_name, session)
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: dict | int | None = None):
        if isinstance(seed, int) or seed is None:
            self._rng = np.random.default_rng(self._seed if seed is None else seed)

    @abstractmethod
    def apply(self, sample: Any) -> Any:
        """Apply noise to a sample drawn from this channel's Gymnasium space."""


class BoxGaussianNoise(ChannelNoise):

    def __init__(
        self,
        loc,
        scale,
        world_name: str,
        session: zenoh.Session,
        seed: int | None = None,
    ):
        super().__init__(world_name, session, seed)
        self.loc = loc
        self.scale = scale

    def apply(self, sample: np.ndarray):
        noise = self._rng.normal(loc=self.loc, scale=self.scale, size=sample.shape)
        return sample + noise


class NoNoise:

    def apply(self, sample: Any) -> Any:
        """Return the sample unchanged."""
        return sample
