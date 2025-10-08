"""Newton backend integration for the ARK simulator."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import newton
import numpy as np
import warp as wp

from ark.tools.log import log
from ark.system.simulation.simulator_backend import SimulatorBackend
from ark.system.newton.newton_camera_driver import NewtonCameraDriver
from ark.system.newton.newton_multibody import NewtonMultiBody
from ark.system.newton.newton_robot_driver import NewtonRobotDriver


def import_class_from_directory(path: Path) -> tuple[type[Any], Optional[type[Any]]]:
    """Import a class (and optional driver enum) from ``path``."""
    class_name = path.name
    file_path = (path / f"{class_name}.py").resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=str(file_path))

    module_dir = str(file_path.parent)
    sys.path.insert(0, module_dir)
    try:
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        drivers_attr: Optional[type[Any]] = None

        spec = importlib.util.spec_from_file_location(class_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[class_name] = module
        spec.loader.exec_module(module)

        if "Drivers" in class_names:
            drivers_cls = getattr(module, "Drivers", None)
            drivers_attr = getattr(drivers_cls, "NEWTON_DRIVER", None) if drivers_cls else None
            class_names.remove("Drivers")

        target_name = class_names[0] if class_names else class_name
        target_cls = getattr(module, target_name)
    finally:
        sys.path.pop(0)
        sys.modules.pop(class_name, None)

    return target_cls, drivers_attr


class NewtonBackend(SimulatorBackend):
    """Simulation backend using the Newton physics engine."""

    def initialize(self) -> None:
        self.ready = False
        sim_cfg = self.global_config.get("simulator", {}).get("config", {})

        self.sim_frequency = float(sim_cfg.get("sim_frequency", 240.0))
        self.sim_substeps = max(int(sim_cfg.get("substeps", 1)), 1)
        base_dt = 1.0 / self.sim_frequency if self.sim_frequency > 0 else 0.005
        self.set_time_step(base_dt)

        self.builder = newton.ModelBuilder()
        gravity = tuple(sim_cfg.get("gravity", [0.0, 0.0, -9.81]))
        self.set_gravity(gravity)

        device_name = sim_cfg.get("device")
        if device_name:
            try:
                wp.set_device(device_name)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"Newton backend: unable to select device '{device_name}': {exc}")

        if self.global_config.get("objects"):
            for obj_name, obj_cfg in self.global_config["objects"].items():
                obj_type = obj_cfg.get("type", "primitive")
                self.add_sim_component(obj_name, obj_type, obj_cfg)

        if self.global_config.get("robots"):
            for robot_name, robot_cfg in self.global_config["robots"].items():
                self.add_robot(robot_name, robot_cfg)

        if self.global_config.get("sensors"):
            for sensor_name, sensor_cfg in self.global_config["sensors"].items():
                from ark.system.driver.sensor_driver import SensorType
                sensor_type = SensorType(sensor_cfg.get("type", "camera").upper())
                self.add_sensor(sensor_name, sensor_type, sensor_cfg)

        self.model = self.builder.finalize()
        self.solver = self._create_solver(sim_cfg.get("solver", "xpbd"))

        self.state_current = self.model.state()
        self.state_next = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_current)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_current)

        self._state_accessor: Callable[[], newton.State] = lambda: self.state_current
        self._bind_runtime_handles()

        self.ready = True

    def _bind_runtime_handles(self) -> None:
        state_accessor = lambda: self.state_current
        for robot in self.robot_ref.values():
            driver = getattr(robot, "_driver", None)
            if isinstance(driver, NewtonRobotDriver):
                driver.bind_runtime(self.model, self.control, state_accessor, self._substep_dt)
        for obj in self.object_ref.values():
            if isinstance(obj, NewtonMultiBody):
                obj.bind_runtime(self.model, state_accessor)
        for sensor in self.sensor_ref.values():
            driver = getattr(sensor, "_driver", None)
            if isinstance(driver, NewtonCameraDriver):
                driver.bind_runtime(self.model, state_accessor)

    def _create_solver(self, solver_name: str) -> newton.solvers.SolverBase:
        name = (solver_name or "xpbd").lower()
        if name in {"xpbd", "solverxpbd"}:
            return newton.solvers.SolverXPBD(self.model)
        if name in {"semiimplicit", "semi_implicit", "solversemiimplicit"}:
            return newton.solvers.SolverSemiImplicit(self.model)
        if name in {"featherstone", "solverfeatherstone"}:
            return newton.solvers.SolverFeatherstone(self.model)
        if name in {"mujoco", "solvermujoco"}:
            return newton.solvers.SolverMuJoCo(self.model)
        log.warning(f"Unknown Newton solver '{solver_name}', falling back to XPBD.")
        return newton.solvers.SolverXPBD(self.model)

    def set_gravity(self, gravity: tuple[float, float, float]) -> None:
        gx, gy, gz = gravity
        axis_map = {0: newton.Axis.X, 1: newton.Axis.Y, 2: newton.Axis.Z}
        components = np.array([gx, gy, gz], dtype=float)
        idx = int(np.argmax(np.abs(components)))
        magnitude = float(components[idx]) if np.any(components) else -9.81
        self.builder.up_axis = axis_map[idx]
        self.builder.gravity = magnitude

    def set_time_step(self, time_step: float) -> None:
        self._time_step = time_step
        self._substep_dt = time_step / self.sim_substeps

    def reset_simulator(self) -> None:
        for robot in list(self.robot_ref.values()):
            robot.kill_node()
        for obj in list(self.object_ref.values()):
            obj.kill_node()
        for sensor in list(self.sensor_ref.values()):
            sensor.kill_node()

        self.robot_ref.clear()
        self.object_ref.clear()
        self.sensor_ref.clear()

        self._simulation_time = 0.0
        self.initialize()
        log.ok("Newton simulator reset complete.")

    def add_robot(self, name: str, robot_config: dict[str, Any]) -> None:
        """Add a robot to the simulation.

        Args:
            name: Name of the robot.
            robot_config: Configuration dictionary for the robot.
        """
        # Create articulation container for the robot
        self.builder.add_articulation()

        class_path = Path(robot_config["class_dir"])
        if class_path.is_file():
            class_path = class_path.parent
        RobotClass, driver_enum = import_class_from_directory(class_path)
        driver_cls = getattr(driver_enum, "value", driver_enum) or NewtonRobotDriver

        driver = driver_cls(name, robot_config, builder=self.builder)
        robot = RobotClass(name=name, global_config=self.global_config, driver=driver)
        self.robot_ref[name] = robot

    def add_sim_component(self, name: str, type: str, obj_config: dict[str, Any]) -> None:
        """Add a generic simulation object.

        Args:
            name: Name of the object.
            type: Type identifier (e.g. "cube", "sphere").
            obj_config: Configuration dictionary for the object.
        """
        component = NewtonMultiBody(name=name, builder=self.builder, global_config=self.global_config)
        self.object_ref[name] = component

    def add_sensor(self, name: str, sensor_type: Any, sensor_config: dict[str, Any]) -> None:
        """Add a sensor to the simulation.

        Args:
            name: Name of the sensor.
            sensor_type: Type of the sensor (SensorType enum).
            sensor_config: Configuration dictionary for the sensor.
        """
        class_path = Path(sensor_config["class_dir"])
        if class_path.is_file():
            class_path = class_path.parent
        SensorClass, driver_enum = import_class_from_directory(class_path)
        driver_cls = getattr(driver_enum, "value", driver_enum) or NewtonCameraDriver

        driver = driver_cls(name, sensor_config)
        sensor = SensorClass(name=name, driver=driver, global_config=self.global_config)
        self.sensor_ref[name] = sensor

    def remove(self, name: str) -> None:
        if name in self.robot_ref:
            self.robot_ref[name].kill_node()
            del self.robot_ref[name]
            return
        if name in self.sensor_ref:
            self.sensor_ref[name].kill_node()
            del self.sensor_ref[name]
            return
        if name in self.object_ref:
            self.object_ref[name].kill_node()
            del self.object_ref[name]
            return
        log.warning(f"Newton backend: component '{name}' does not exist.")

    def _all_available(self) -> bool:
        for robot in self.robot_ref.values():
            if robot._is_suspended:  # noqa: SLF001
                return False
        for obj in self.object_ref.values():
            if obj._is_suspended:  # noqa: SLF001
                return False
        return True

    def step(self) -> None:
        if not self._all_available():
            return

        self._step_sim_components()
        for _ in range(self.sim_substeps):
            self.state_current.clear_forces()
            self.contacts = self.model.collide(self.state_current)
            self.solver.step(
                self.state_current,
                self.state_next,
                self.control,
                self.contacts,
                self._substep_dt,
            )
            self.state_current, self.state_next = self.state_next, self.state_current

        self._simulation_time += self._time_step

    def shutdown_backend(self) -> None:
        for robot in list(self.robot_ref.values()):
            robot.kill_node()
        for obj in list(self.object_ref.values()):
            obj.kill_node()
        for sensor in list(self.sensor_ref.values()):
            sensor.kill_node()
        self.robot_ref.clear()
        self.object_ref.clear()
        self.sensor_ref.clear()
        self.ready = False
