from abc import ABC, abstractmethod
from typing import Any
from gymnasium import Space


class SensorDriver(ABC):
    """Abstract base class for a sensor on a robot or in the environment.

    Users subclass this in their driver package. The framework publishes
    get_state() samples on a Zenoh channel at the configured frequency.

    Example::

        class MyFTSensorDriver(SensorDriver):

            @property
            def state_space(self) -> Box:
                return Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)

            def get_state(self) -> np.ndarray:
                # read Fx, Fy, Fz, Mx, My, Mz from hardware
                ...
    """

    @property
    @abstractmethod
    def state_space(self) -> Space:
        """Gymnasium space describing the sample returned by get_state()."""

    @abstractmethod
    def get_state(self) -> Any:
        """Return the current sensor reading as a sample matching state_space."""

    def reset(self, seed: int | None = None):
        """Called by DriverNode on environment reset. Override if needed."""
