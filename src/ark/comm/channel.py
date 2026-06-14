from __future__ import annotations

import zenoh
import numpy as np
from typing import Any
from dataclasses import dataclass
from ark.reset import ResetObject
from ark.time import Clock, Sleep, Time
from abc import abstractmethod


class ChannelName(str):
    _separator = "/"

    def __new__(cls, *parts: str | "ChannelName") -> "ChannelName":
        if not parts:
            raise ValueError("ChannelName requires at least one part")

        normalized_parts: list[str] = []
        for part in parts:
            normalized_parts.extend(cls._normalize_part(part))

        if not normalized_parts:
            raise ValueError("ChannelName cannot be empty")

        return str.__new__(cls, cls._separator.join(normalized_parts))

    @classmethod
    def _normalize_part(cls, part: str | "ChannelName") -> list[str]:
        if not isinstance(part, str):
            raise TypeError(
                "ChannelName parts must be strings or ChannelName instances, "
                f"got {type(part).__name__}"
            )

        text = part.strip(cls._separator)
        if not text:
            raise ValueError("ChannelName parts cannot be empty")

        return [segment for segment in text.split(cls._separator) if segment]

    def __truediv__(self, other: str | "ChannelName") -> "ChannelName":
        return type(self)(self, other)

    def joinpath(self, *others: str | "ChannelName") -> "ChannelName":
        return type(self)(self, *others)

    @property
    def parts(self) -> tuple[str, ...]:
        return tuple(self.split(self._separator))


@dataclass(frozen=True, slots=True)
class Channel:
    name: ChannelName
    env_name: str

    @property
    def full_name(self) -> ChannelName:
        return ChannelName(self.env_name) / self.name


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


class NoNoise:

    def apply(self, sample: Any) -> Any:
        """Return the sample unchanged."""
        return sample


NOISE_TYPE = list[ChannelNoise | NoNoise] | ChannelNoise | NoNoise | None


def normalise_noise(noise: NOISE_TYPE) -> list[ChannelNoise | NoNoise]:
    if noise is None:
        return []
    if isinstance(noise, (ChannelNoise, NoNoise)):
        return [noise]
    return list(noise)


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
