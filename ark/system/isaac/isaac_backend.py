from __future__ import annotations

import ast
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Optional

from ark.system.isaac.isaac_object import IsaacSimObject
from ark.system.simulation.simulator_backend import SimulatorBackend
from ark.tools.log import log
from ark.utils import lazy

try:
    from isaacsim import SimulationApp
except ImportError as exc:
    raise ImportError(
        "Isaac Sim Python packages are required for the IsaacSim backend. "
        "Install and source Isaac Sim before selecting backend_type=isaacsim."
    ) from exc


def import_class_from_directory(path: Path) -> tuple[type, Optional[type]]:
    """Load the component class and its optional Isaac Sim driver declaration.

    Components that want to provide a custom Isaac Sim driver should expose a
    ``Drivers`` enum with an ``ISAAC_DRIVER`` entry (mirroring the existing
    ``PYBULLET_DRIVER`` / ``MUJOCO_DRIVER`` pattern).  If none is provided the
    backend falls back to :class:`IsaacSimRobotDriver`.
    """

    class_name = path.name
    file_path = path / f"{class_name}.py"
    file_path = file_path.resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=file_path)

    module_dir = os.path.dirname(file_path)
    sys.path.insert(0, module_dir)

    class_names = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    ]

    driver_cls = None
    if "Drivers" in class_names:
        spec = importlib.util.spec_from_file_location(class_names[0], file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[class_names[0]] = module
        spec.loader.exec_module(module)

        class_ = getattr(module, class_names[0])
        sys.path.pop(0)
        driver_cls = class_.ISAAC_DRIVER
        class_names.remove("Drivers")

    spec = importlib.util.spec_from_file_location(class_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[class_name] = module
    spec.loader.exec_module(module)

    class_ = getattr(module, class_names[0])
    sys.path.pop(0)

    if driver_cls is not None and hasattr(driver_cls, "value"):
        driver_cls = driver_cls.value

    return class_, driver_cls


class IsaacSimBackend(SimulatorBackend):
    """Backend wrapper for running ARK simulations inside Isaac Sim."""

    def __init__(self, global_config: dict[str, Any]) -> None:
        """Initialize the backend and parse simulator configuration.

        Determines connection mode (GUI or headless), sets defaults, and
        registers the backend's custom event loop.

        Args:
            global_config (dict[str, Any]): Global ARK simulation configuration
                containing the simulator and component specifications.
        """
        self._app = None
        self.world = None
        self._headless = True
        self.timestep = 0.0

        super().__init__(global_config)

        sim_cfg = self.global_config["simulator"]["config"]
        connection_mode = sim_cfg.get("connection_mode", "headless").lower()
        self._headless = connection_mode != "gui"
        self.custom_event_loop = self.run

    def initialize(self) -> None:
        """Initialize the Isaac Sim application, stage, and scene components."""

        # Create a simulator
        self._app = SimulationApp({"headless": self._headless})

        sim_cfg = self.global_config["simulator"]["config"]
        physics_dt = 1 / sim_cfg.get("sim_frequency", 120.0)
        render_dt = 1 / sim_cfg.get("render_frequency", 60)

        # Creates scene
        self.world = lazy.isaacsim.core.api.World(
            stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt
        )
        self.world.scene.add_default_ground_plane()

        # Set gravity
        gravity = sim_cfg.get("gravity", [0.0, 0.0, -9.81])
        self.set_gravity(gravity)

        self.timestep = physics_dt

        # Add components to the simulator
        if self.global_config.get("objects", None):
            for object_name, object_config in self.global_config["objects"].items():
                self.add_sim_component(object_name, object_config)

        if self.global_config.get("robots", None):
            for robot_name, robot_config in self.global_config["robots"].items():
                self.add_robot(robot_name, robot_config)

        if self.global_config.get("sensors", None):
            for sensor_name, sensor_config in self.global_config["sensors"].items():
                self.add_sensor(sensor_name, sensor_config)

        # Allow simulator to settle
        self._app.update()
        for _ in range(250):
            self.world.step(render=True)

    def set_gravity(self, gravity: tuple[float, float, float]) -> None:
        """Set gravity for the simulation.

        Args:
            gravity (tuple[float, float, float]): Gravity vector in XYZ format.

        """
        if self.world is None:
            return

        # Isaac's physics_context.set_gravity expects a scalar
        gravity_scalar = float(gravity[2])
        self.world._physics_context.set_gravity(gravity_scalar)

    def reset_simulator(self) -> None:
        """Reset the Isaac Sim world and all components."""

        if self.world is None:
            return

        self.world.reset()
        for robot in self.robot_ref:
            robot._driver.sim_reset()

        for obj in self.object_ref:
            self.object_ref[obj].reset_component()

        # TODO check sensor reset

        for _ in range(10):
            self.world.step(render=True)

    def add_robot(
        self,
        name: str,
        robot_config: dict[str, Any],
    ) -> None:
        """Dynamically load a robot class and driver into the simulation.

        Args:
            name (str): Name of the robot component.
            robot_config (dict[str, Any]): Robot configuration, including:
                - class_dir (str): Directory containing the robot + driver classes.
                - Additional robot-specific fields.
        """
        class_path = Path(robot_config["class_dir"])
        if class_path.is_file():
            class_path = class_path.parent

        RobotClass, DriverClass = import_class_from_directory(class_path)

        driver = DriverClass(
            component_name=name,
            component_config=robot_config,
            sim_app=self._app,
            world=self.world,
        )
        robot = RobotClass(
            name=name,
            driver=driver,
            global_config=self.global_config,
        )
        self.robot_ref[name] = robot

    def add_sensor(
        self,
        name: str,
        sensor_config: dict[str, Any],
    ) -> None:
        """Load and register a sensor class and its driver.

        Args:
            name (str): Sensor name.
            sensor_config (dict[str, Any]): Sensor configuration parameters.
        """
        class_path = Path(sensor_config["class_dir"])
        if class_path.is_file():
            class_path = class_path.parent

        SensorClass, DriverClass = import_class_from_directory(class_path)

        driver = DriverClass(
            component_name=name,
            component_config=sensor_config,
            world=self.world,
        )
        sensor = SensorClass(
            name=name,
            driver=driver,
            global_config=self.global_config,
        )
        self.sensor_ref[name] = sensor

    def add_sim_component(
        self,
        name: str,
        obj_config: dict[str, Any],
    ) -> None:
        """Add a static object to the simulation via `IsaacSimObject`.

        Args:
            name (str): Name of the object.
            obj_config (dict[str, Any]): Object configuration.
        """
        obj = IsaacSimObject(
            name=name, world=self.world, global_config=self.global_config
        )
        self.object_ref[name] = obj
        log.ok(f"Loaded '{name}' into Isaac Sim stage.")

    @staticmethod
    def remove(name: str) -> None:
        log.warn("Dynamic removal is not supported in the IsaacSim backend yet.")

    def run(self, sim_node) -> None:
        """Main simulation loop for Isaac Sim integration.

        Handles LCM message processing for the simulator node, robots,
        and sensors, then advances physics and rendering each cycle.

        Args:
            sim_node: The main ARK simulation node controlling global stepping.

        Loop Behavior:
            - Runs while the Isaac Sim app window is open.
            - Processes LCM messages with zero timeout.
            - Calls `sim_node.step()` for ARK updates.
            - Calls backend `_step()` for physics/render stepping.
        """
        lcms = (
            [sim_node._lcm]
            + [r._lcm for r in self.robot_ref.values()]
            + [s._lcm for s in self.sensor_ref.values()]
        )
        while self._app.is_running():
            for lc in lcms:
                lc.handle_timeout(0)
            sim_node.step()
            self._step()

    def _step(self) -> None:
        """Execute a single backend simulation step."""
        self._step_sim_components()
        self.world.step(render=True)
        self._simulation_time += self.timestep

    def step(self) -> None:
        """Unused ARK interface override."""
        pass

    def shutdown_backend(self) -> None:
        """Shutdown Isaac Sim backend and all components."""
        for robot in self.robot_ref:
            self.robot_ref[robot].kill_node()
        for obj in self.object_ref:
            self.object_ref[obj].kill_node()
        for sensor in self.sensor_ref:
            self.sensor_ref[sensor].kill_node()
        if self._app is not None:
            self._app.close()
