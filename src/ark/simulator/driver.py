from ark.driver import ControllerMode, JointGroupDriver, SensorDriver, RobotDriver


class SimulatedJointGroupDriver(JointGroupDriver):
    """Intermediate abstract base for simulated joint group drivers.

    Stores joint_names and control_mode so concrete simulator backends
    (e.g. PybulletJointGroupDriver) only need to implement the
    physics-specific parts: state_space, is_ready, get_state, set_target.
    """

    def __init__(self, joint_names: list[str], control_mode: ControllerMode):
        self._joint_names = joint_names
        self._control_mode = control_mode

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def control_mode(self) -> ControllerMode:
        return self._control_mode


class SimulatedSensorDriver(SensorDriver):
    """Intermediate abstract base for simulated sensor drivers."""


class SimulatedRobotDriver(RobotDriver):
    """RobotDriver backed by a simulator rather than real hardware."""
