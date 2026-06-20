import numpy as np
from abc import ABC, abstractmethod
from gymnasium.spaces import Box
from .modes import ControllerMode
from ark.envs.spaces.sensor_space import JointState


class JointGroupDriver(ABC):
    """Abstract base class for a group of joints on a robot.

    Users subclass this in their driver package and implement the abstract
    methods to interface with their hardware. The framework wraps the driver
    in a DriverNode that handles all Zenoh communication.

    Minimal implementation example::

        class MyArmDriver(JointGroupDriver):

            @property
            def joint_names(self) -> list[str]:
                return ["joint1", "joint2", "joint3"]

            @property
            def control_mode(self) -> ControllerMode:
                return ControllerMode.JOINT_POSITION

            @property
            def state_space(self) -> JointState:
                return JointState(
                    joint_names=self.joint_names,
                    position_limits=Limits(
                        lower=np.full(3, -np.pi),
                        upper=np.full(3, np.pi),
                    ),
                )

            def is_ready(self) -> bool:
                return True

            def get_state(self) -> dict[str, np.ndarray]:
                # read from hardware and return dict matching state_space
                ...

            def set_target(self, target: np.ndarray):
                # send target to hardware
                ...
    """

    @property
    @abstractmethod
    def joint_names(self) -> list[str]:
        """Ordered list of joint names in this group. Single source of truth."""

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    @property
    @abstractmethod
    def control_mode(self) -> ControllerMode:
        """Control mode this driver operates in."""

    @property
    @abstractmethod
    def state_space(self) -> JointState:
        """Gymnasium space describing the state returned by get_state()."""

    @property
    def command_space(self) -> Box:
        """Command space accepted by set_target(). Derived from state_space by default."""
        space = self.state_space
        if self.control_mode == ControllerMode.JOINT_POSITION:
            pos = space["position"]
            return Box(pos.low, pos.high, shape=(self.dof,), dtype=pos.dtype)
        if self.control_mode == ControllerMode.JOINT_TORQUE:
            if "effort" not in space.spaces:
                raise ValueError(
                    "state_space must include 'effort' limits to derive a torque command_space."
                )
            eff = space["effort"]
            return Box(eff.low, eff.high, shape=(self.dof,), dtype=eff.dtype)
        raise ValueError(
            f"No default command_space for control mode: {self.control_mode}"
        )

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True when the driver has valid state and can accept commands."""

    @abstractmethod
    def get_state(self) -> dict[str, np.ndarray]:
        """Return the current joint state as a dict matching state_space."""

    @abstractmethod
    def set_target(self, target: np.ndarray):
        """Send a command to the joint group. target matches command_space."""

    def reset(self, seed: int | None = None):
        """Called by DriverNode on environment reset. Override if the driver needs reset logic."""
