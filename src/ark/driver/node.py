import zenoh
from ark.node import Node
from ark.parameters import PARAM_TYPE
from ark.comm.stamped_sample import StampedSample
from .robot import RobotDriver


class DriverNode(Node):
    """Bridges a RobotDriver to the Zenoh network.

    For each joint group the node:
      - Subscribes to ``{group_name}/command`` and forwards to driver.set_target()
      - Publishes ``{group_name}/state`` at the configured frequency

    For each sensor the node:
      - Publishes ``{sensor_name}/state`` at the configured frequency

    Channel names follow the defaults above and can be remapped via
    channel_remaps like any other Node.

    This node always runs in real (sim=False) mode. It never applies noise —
    noise in real deployments comes from reality itself.
    """

    def __init__(
        self,
        env_name: str,
        node_name: str,
        robot_driver: RobotDriver,
        joint_group_frequencies: dict[str, float],
        sensor_frequencies: dict[str, float],
        parameters: dict[str, PARAM_TYPE],
        channel_remaps: dict[str, str],
        session: zenoh.Session,
    ):
        super().__init__(env_name, node_name, parameters, channel_remaps, session)
        self._robot_driver = robot_driver
        self._setup_joint_groups(joint_group_frequencies)
        self._setup_sensors(sensor_frequencies)

    def _setup_joint_groups(self, frequencies: dict[str, float]):
        for group_name in self._robot_driver.joint_group_names:
            driver = self._robot_driver.joint_group_driver(group_name)

            state_pub = self.create_publisher(
                f"{group_name}/state",
                driver.state_space,
            )

            def _on_command(stamped: StampedSample, drv=driver):
                drv.set_target(stamped.sample)

            self.create_subscriber(
                f"{group_name}/command",
                driver.command_space,
                _on_command,
            )

            hz = frequencies.get(group_name, 100.0)

            def _publish_state(t, drv=driver, pub=state_pub):
                if drv.is_ready():
                    pub.publish(drv.get_state())

            self.create_stepper(hz, _publish_state)

    def _setup_sensors(self, frequencies: dict[str, float]):
        for sensor_name in self._robot_driver.sensor_names:
            driver = self._robot_driver.sensor_driver(sensor_name)

            state_pub = self.create_publisher(
                f"{sensor_name}/state",
                driver.state_space,
            )

            hz = frequencies.get(sensor_name, 30.0)

            def _publish_state(t, drv=driver, pub=state_pub):
                pub.publish(drv.get_state())

            self.create_stepper(hz, _publish_state)

    def reset(self, seed: int | None = None):
        super().reset(seed)
        self._robot_driver.reset(seed)
