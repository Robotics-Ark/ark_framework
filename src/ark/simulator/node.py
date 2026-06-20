from __future__ import annotations

import time
import threading
import numpy as np
import zenoh

from ark.node import Node
from ark.parameters import PARAM_TYPE
from ark.time import SimulatedTime
from ark.envs.spaces.geometry_space import RigidTransform
from ark.comm.stamped_sample import StampedSample
from ark.simulator.base import Simulator, SimulatedWorld
from ark.simulator.driver import SimulatedRobotDriver


_COMPUTING_CHANNEL = "_ark/{env_name}/computing"


class SimulatorNode(Node):
    """Drives a physics simulator and bridges it to the Zenoh network.

    Responsibilities:
    - Runs the physics step loop at ``sim_time_freq`` Hz.
    - Publishes joint state, sensor state, and object poses after each step.
    - Ticks SimulatedTime after publishing so observers always see consistent state.
    - Switches to real-time pacing (1× wall clock) while ArkEnv.compute() is active,
      so the sim advances at the same rate as reality during policy computation.

    Channel conventions (all relative to env_name, remappable):
      {robot_name}/{group_name}/state    — joint state publisher
      {robot_name}/{group_name}/command  — joint command subscriber
      {sensor_name}/state                — sensor state publisher
      {object_name}/pose                 — object pose publisher
    """

    def __init__(
        self,
        env_name: str,
        node_name: str,
        simulator: Simulator,
        sim_time: SimulatedTime,
        parameters: dict[str, PARAM_TYPE],
        channel_remaps: dict[str, str],
        session: zenoh.Session,
    ):
        super().__init__(env_name, node_name, parameters, channel_remaps, session)
        self._simulator = simulator
        self._sim_time = sim_time
        self._step_period = simulator.time_step_sec

        # State that changes after the first reset
        self._world_ready = False
        self._state_getters: list[tuple] = []  # (Publisher, callable) pairs

        # Real-time pacing flag — set to True by ArkEnv.compute()
        self._computing = False
        self._computing_sub = session.declare_subscriber(
            _COMPUTING_CHANNEL.format(env_name=env_name),
            self._on_compute_flag,
        )

        # Physics step loop runs in its own thread.
        # Uses _stop_event (inherited from Node.spin / Node.close) to halt.
        self._step_thread = threading.Thread(
            target=self._step_loop, daemon=True, name=f"{node_name}.step_loop"
        )
        self._step_thread.start()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None):
        super().reset(seed)
        world = self._simulator.reset_simulator()
        self._simulator.domain_randomize(np.random.default_rng(seed))
        self._sim_time.reset()

        if not self._world_ready:
            self._wire_world(world)
            self._world_ready = True

    def _wire_world(self, world: SimulatedWorld):
        """Create publishers and subscribers for all items in SimulatedWorld.

        Called once after the first reset. Subsequent resets reuse these
        endpoints — the world structure is expected to be stable across
        episodes.
        """
        self._state_getters.clear()

        for robot_name, robot_driver in world.robot_drivers.items():
            self._wire_robot(robot_name, robot_driver)

        for sensor_name, sensor_driver in world.sensor_drivers.items():
            pub = self.create_publisher(
                f"{sensor_name}/state",
                sensor_driver.state_space,
            )
            self._state_getters.append((pub, sensor_driver.get_state))

        pose_space = RigidTransform()
        for obj_name, pose_getter in world.object_pose_getters.items():
            pub = self.create_publisher(f"{obj_name}/pose", pose_space)
            self._state_getters.append((pub, pose_getter))

    def _wire_robot(self, robot_name: str, robot_driver: SimulatedRobotDriver):
        for group_name in robot_driver.joint_group_names:
            driver = robot_driver.joint_group_driver(group_name)

            state_pub = self.create_publisher(
                f"{robot_name}/{group_name}/state",
                driver.state_space,
            )
            self._state_getters.append((state_pub, driver.get_state))

            def _on_command(stamped: StampedSample, drv=driver):
                drv.set_target(stamped.sample)

            self.create_subscriber(
                f"{robot_name}/{group_name}/command",
                driver.command_space,
                _on_command,
            )

        for sensor_name in robot_driver.sensor_names:
            driver = robot_driver.sensor_driver(sensor_name)
            pub = self.create_publisher(
                f"{robot_name}/{sensor_name}/state",
                driver.state_space,
            )
            self._state_getters.append((pub, driver.get_state))

    # ------------------------------------------------------------------
    # Physics step loop
    # ------------------------------------------------------------------

    def _step_loop(self):
        """Continuously step the simulator until close() is called.

        Pacing:
        - Free-running (computing=False): step as fast as possible.
          The sim races ahead; ArkEnv.rate.sleep() blocks on sim time ticks.
        - Real-time (computing=True): sleep one step period between steps
          so sim time advances at 1× wall-clock speed, matching reality.
        """
        while not self._stop_event.is_set():
            if self._computing:
                time.sleep(self._step_period)

            self._simulator.step_simulator()

            if self._world_ready:
                for pub, getter in self._state_getters:
                    pub.publish(getter())

            self._sim_time.tick()

    # ------------------------------------------------------------------
    # Compute mode flag
    # ------------------------------------------------------------------

    def _on_compute_flag(self, sample: zenoh.Sample):
        payload = bytes(sample.payload)
        self._computing = bool(payload[0]) if payload else False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        super().close()
        self._computing_sub.undeclare()
        self._step_thread.join(timeout=2.0)
        self._sim_time.close()
        self._simulator.close()
