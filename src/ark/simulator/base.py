from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ark.frames import FrameForest
from ark.envs.spaces.geometry_space import RigidTransform
from ark.simulator.driver import SimulatedRobotDriver, SimulatedSensorDriver


@dataclass
class SimulatedWorld:
    """Everything a SimulatorNode needs to wire Zenoh channels after a reset.

    robot_drivers:
        Mapping of robot name → SimulatedRobotDriver. The SimulatorNode
        publishes ``{robot_name}/{group_name}/state`` for each joint group
        and subscribes to ``{robot_name}/{group_name}/command``.

    sensor_drivers:
        Standalone sensors not attached to a robot (e.g. a fixed camera).
        The SimulatorNode publishes ``{sensor_name}/state`` for each.

    object_pose_getters:
        Free-flyer objects whose world-frame pose is queryable from the
        simulator. Keyed by object name; the callable returns a sample
        matching a RigidTransform space (dict with "translation" and
        "rotation" keys). The SimulatorNode publishes
        ``{object_name}/pose`` for each at the configured frequency.
    """

    robot_drivers: dict[str, SimulatedRobotDriver] = field(default_factory=dict)
    sensor_drivers: dict[str, SimulatedSensorDriver] = field(default_factory=dict)
    object_pose_getters: dict[str, Callable[[], dict[str, np.ndarray]]] = field(
        default_factory=dict
    )


class Simulator(ABC):
    """Abstract physics simulator backend.

    Concrete subclasses implement _init_simulator, reset_simulator,
    step_simulator, and close for a specific physics engine (PyBullet,
    MuJoCo, …). The Simulator does not own SimulatedTime — that belongs
    to SimulatorNode so that time management is centralised there.

    Domain randomisation config (``dr_cfg``) is backend-specific; each
    subclass interprets its own schema.  ``domain_randomize`` is called by
    SimulatorNode after every ``reset_simulator`` call, before the first
    physics step of each episode.
    """

    def __init__(
        self,
        time_step_sec: float,
        sim_cfg: dict[str, Any],
        asset_cfg: dict[str, Any],
        gravity: dict[str, float],
        dr_cfg: dict[str, Any] | None = None,
    ):
        self._time_step_sec = time_step_sec
        self._sim_cfg = sim_cfg
        self._asset_cfg = asset_cfg
        self._gravity = gravity
        self._dr_cfg: dict[str, Any] = dr_cfg or {}
        self._frame_forest = FrameForest()
        self._init_simulator()

    @property
    def time_step_sec(self) -> float:
        return self._time_step_sec

    def register_frame_forest(self, frame_forest: FrameForest):
        self._frame_forest = frame_forest

    @abstractmethod
    def _init_simulator(self):
        """One-time initialisation (e.g. connect to physics engine)."""

    @abstractmethod
    def reset_simulator(self) -> SimulatedWorld:
        """Reset physics to initial state and return the live driver handles."""

    @abstractmethod
    def step_simulator(self):
        """Advance the physics by one time step."""

    @abstractmethod
    def close(self):
        """Release all physics engine resources."""

    def domain_randomize(self, rng: np.random.Generator) -> None:
        """Apply domain randomisation for this episode.

        Called by SimulatorNode after reset_simulator() and before the first
        step. The default implementation is a no-op; override to randomise
        physics parameters (mass, friction, gravity, damping, …) according to
        ``self._dr_cfg``.

        Subclasses must restore nominal values before applying new random
        perturbations so that successive episodes are independent.
        """
