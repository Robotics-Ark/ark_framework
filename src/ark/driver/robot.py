from typing import Any
import numpy as np
from .joint_group import JointGroupDriver
from .sensor import SensorDriver


class RobotDriver:
    """Aggregates joint group and sensor drivers for a single robot."""

    def __init__(
        self,
        name: str,
        joint_group_drivers: dict[str, JointGroupDriver],
        sensor_drivers: dict[str, SensorDriver],
    ):
        self.name = name
        self._joint_group_drivers = joint_group_drivers
        self._sensor_drivers = sensor_drivers

    @property
    def joint_group_names(self) -> list[str]:
        return list(self._joint_group_drivers.keys())

    @property
    def sensor_names(self) -> list[str]:
        return list(self._sensor_drivers.keys())

    def joint_group_driver(self, name: str) -> JointGroupDriver:
        return self._joint_group_drivers[name]

    def sensor_driver(self, name: str) -> SensorDriver:
        return self._sensor_drivers[name]

    def get_joint_group_state(self, group_name: str) -> dict[str, np.ndarray]:
        return self._joint_group_drivers[group_name].get_state()

    def set_joint_group_target(self, group_name: str, target: np.ndarray):
        self._joint_group_drivers[group_name].set_target(target)

    def get_sensor_state(self, sensor_name: str) -> Any:
        return self._sensor_drivers[sensor_name].get_state()

    def reset(self, seed: int | None = None):
        for driver in self._joint_group_drivers.values():
            driver.reset(seed)
        for driver in self._sensor_drivers.values():
            driver.reset(seed)
