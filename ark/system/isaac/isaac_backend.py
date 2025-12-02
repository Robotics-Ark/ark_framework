from __future__ import annotations

import ast
import asyncio
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from ark.system.isaac.isaac_camera_driver import IsaacCameraDriver
from ark.system.isaac.isaac_object import IsaacSimObject
from ark.system.isaac.isaac_robot_driver import IsaacSimRobotDriver
from ark.system.simulation.simulator_backend import SimulatorBackend
from ark.tools.log import log


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

        driver_cls = getattr(class_, "ISAAC_DRIVER", None)
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
        """Initialize the Isaac Sim app and stage."""
        try:
            from isaacsim import SimulationApp
        except ImportError as exc:
            raise ImportError(
                "Isaac Sim Python packages are required for the IsaacSim backend. "
                "Install and source Isaac Sim before selecting backend_type=isaacsim."
            ) from exc

        self._app = SimulationApp({"headless": False})
        from isaacsim.core.api import World

        sim_cfg = self.global_config["simulator"]["config"]
        physics_dt = 1 / sim_cfg.get("sim_frequency", 120.0)
        # render_dt = 1 / sim_cfg.get("render_frequency", physics_dt)
        render_dt = 1 / sim_cfg.get("render_frequency", 60)

        self.world = World(physics_dt=physics_dt, rendering_dt=render_dt)
        self.world.scene.add_default_ground_plane()

        # gravity = sim_cfg.get("gravity", [0.0, 0.0, -9.81])
        # self.set_gravity(gravity)
        self.timestep = physics_dt

        if self.global_config.get("objects", None):
            for object_name, object_config in self.global_config["objects"].items():
                self.add_sim_component(object_name, object_config)

        if self.global_config.get("robots", None):
            for robot_name, robot_config in self.global_config["robots"].items():
                self.add_robot(robot_name, robot_config)

        if self.global_config.get("sensors", None):
            for sensor_name, sensor_config in self.global_config["sensors"].items():
                self.add_sensor(sensor_name, sensor_config)

        self.world.reset()
        self.world.step(render=True)

    def set_gravity(self, gravity: tuple[float, float, float]) -> None:
        """Set gravity using the physics context (World lacks set_gravity in some versions)."""
        if self.world is None:
            return

        # Isaac's physics_context.set_gravity expects a scalar; convert vector -> z component
        if isinstance(gravity, (list, tuple)):
            try:
                gravity_scalar = float(gravity[2])
            except Exception:
                gravity_scalar = float(gravity[0]) if len(gravity) > 0 else -9.81
        else:
            gravity_scalar = float(gravity)

        # Isaac versions differ on where the physics context hangs off the World
        phys_ctx = getattr(self.world, "physics_context", None) or getattr(
            self.world, "_physics_context", None
        )
        if phys_ctx and hasattr(phys_ctx, "set_gravity"):
            phys_ctx.set_gravity(gravity_scalar)
            return

        # Fallback to global singleton if present
        try:
            from omni.isaac.core.physics_context.physics_context import PhysicsContext
        except Exception:
            phys_ctx = None
        else:
            phys_ctx = PhysicsContext.instance()

        if phys_ctx and hasattr(phys_ctx, "set_gravity"):
            phys_ctx.set_gravity(gravity_scalar)
        else:
            log.warn("Could not set gravity: physics context not available.")

    def reset_simulator(self) -> None:
        if self.world is None:
            return

        self.world.reset()
        for robot_name, robot in self.robot_ref.items():
            robot._driver.sim_reset()

    def add_robot(
        self,
        name: str,
        robot_config: dict[str, Any],
    ) -> None:
        class_path = Path(robot_config["class_dir"])
        if class_path.is_file():
            class_path = class_path.parent

        RobotClass, DriverClass = import_class_from_directory(class_path)
        DriverClass = DriverClass or IsaacSimRobotDriver

        driver = DriverClass(
            component_name=name,
            component_config=robot_config,
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
        class_path = Path(sensor_config["class_dir"])
        if class_path.is_file():
            class_path = class_path.parent

        SensorClass, DriverClass = import_class_from_directory(class_path)
        DriverClass = DriverClass or IsaacCameraDriver

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
        """Load static USD/URDF/primitive assets into the stage via IsaacSimObject."""
        obj = IsaacSimObject(
            name=name, world=self.world, global_config=self.global_config
        )
        self.object_ref[name] = obj
        log.ok(f"Loaded '{name}' into Isaac Sim stage.")

    def remove(self, name: str) -> None:
        log.warn("Dynamic removal is not supported in the IsaacSim backend yet.")

    def run(self, lcm_handle):
        while self._app.is_running():
            lcm_handle.handle_timeout(0)
            self._step()

    def _step(self):
        print("stepping....")
        self._step_sim_components()
        print("Stepping world...")
        self.world.step(render=True)
        print("Finished Stepping world...")
        self._simulation_time += self.timestep

    def step(self) -> None:
        print("CALLED ZOMBIE STEP.....")
        pass

    def shutdown_backend(self) -> None:
        # TODO make it clean
        if self.world is not None:
            try:
                # Newer Isaac releases expose clear(), older ones provide close()
                close_fn = getattr(self.world, "clear", None) or getattr(
                    self.world, "close", None
                )
                if close_fn:
                    close_fn()
            except Exception:
                pass
        if self._app is not None:
            try:
                self._app.close()
            except Exception:
                pass
