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
from ark.system.newton.newton_builder import NewtonBuilder
from ark.system.newton.newton_camera_driver import NewtonCameraDriver
from ark.system.newton.newton_lidar_driver import NewtonLiDARDriver
from ark.system.newton.newton_multibody import NewtonMultiBody
from ark.system.newton.newton_robot_driver import NewtonRobotDriver
from ark.system.newton.newton_viewer import NewtonViewerManager
from arktypes import *

import textwrap

def import_class_from_directory(path: Path) -> tuple[type[Any], Optional[type[Any]]]:
    """Import a class (and optional driver enum) from ``path``.
        The helper searches for ``<ClassName>.py`` inside ``path`` and imports the
    class with the same name.  If a ``Drivers`` class is present in the module
    its ``NEWTON_DRIVER`` attribute is returned alongside the main class.

    @param path Path to the directory containing the module.
    @return Tuple ``(cls, driver_cls)`` where ``driver_cls`` is ``None`` when no
            driver is defined.
    @rtype Tuple[type, Optional[type]]

    """

    ## Extract the class name from the last part of the directory path (last directory name)
    class_name = path.name
    file_path = (path / f"{class_name}.py").resolve() ##just add the resolve here instead of newline
    ## Defensive check for the filepath, raise error if not found
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=str(file_path))

    module_dir = str(file_path.parent)
    sys.path.insert(0, module_dir)
        ## Import the module dynamically and extract class names defensively
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

    def _determine_up_axis(self, gravity: tuple[float, float, float]) -> newton.Axis:
        """Determine up axis from gravity vector."""
        gx, gy, gz = gravity
        components = np.array([gx, gy, gz], dtype=float)
        if not np.any(components):
            return newton.Axis.Z  # Default to Z-up if no gravity
        idx = int(np.argmax(np.abs(components)))
        axis_map = {0: newton.Axis.X, 1: newton.Axis.Y, 2: newton.Axis.Z}
        return axis_map[idx]

    def _extract_gravity_magnitude(self, gravity: tuple[float, float, float]) -> float:
        """Extract gravity magnitude from gravity vector."""
        gx, gy, gz = gravity
        components = np.array([gx, gy, gz], dtype=float)
        if not np.any(components):
            return -9.81  # Default gravity
        idx = int(np.argmax(np.abs(components)))
        return float(components[idx])

    def _apply_joint_defaults(self, sim_cfg: dict[str, Any]) -> None:
        """Apply Newton-specific joint defaults from configuration.

        This must be called BEFORE loading any robots so defaults apply to all URDFs.
        """
        newton_cfg = sim_cfg.get("newton_physics", {})
        joint_cfg = newton_cfg.get("joint_defaults", {})

        if not joint_cfg:
            log.info("Newton backend: No joint_defaults in config, using Newton defaults")
            return

        # Map string mode to Newton enum
        mode_str = joint_cfg.get("mode", "TARGET_POSITION").upper()
        mode_value = None
        if mode_str == "TARGET_POSITION":
            mode_value = newton.JointMode.TARGET_POSITION
        elif mode_str == "TARGET_VELOCITY":
            mode_value = newton.JointMode.TARGET_VELOCITY
        elif mode_str == "FORCE":
            mode_value = newton.JointMode.FORCE

        # Build kwargs dict for set_default_joint_config
        joint_defaults = {}
        if mode_value is not None:
            joint_defaults["mode"] = mode_value
        if "target_ke" in joint_cfg:
            joint_defaults["target_ke"] = float(joint_cfg["target_ke"])
        if "target_kd" in joint_cfg:
            joint_defaults["target_kd"] = float(joint_cfg["target_kd"])
        if "limit_ke" in joint_cfg:
            joint_defaults["limit_ke"] = float(joint_cfg["limit_ke"])
        if "limit_kd" in joint_cfg:
            joint_defaults["limit_kd"] = float(joint_cfg["limit_kd"])
        if "armature" in joint_cfg:
            joint_defaults["armature"] = float(joint_cfg["armature"])

        # Apply via NewtonBuilder
        self.scene_builder.set_default_joint_config(**joint_defaults)

        log.info(
            f"Newton backend: Applied joint defaults - "
            f"ke={self.scene_builder.builder.default_joint_cfg.target_ke}, "
            f"kd={self.scene_builder.builder.default_joint_cfg.target_kd}, "
            f"armature={self.scene_builder.builder.default_joint_cfg.armature}"
        )

    def _add_ground_support(self, sim_cfg: dict[str, Any], ground_cfg: dict[str, Any]) -> None:
        """Add ground plane support compatible with all solvers.

        MuJoCo solver cannot handle builder.add_ground_plane(), so we use explicit
        box geometry for MuJoCo while keeping native ground plane for other solvers.
        """
        solver_name = (sim_cfg.get("solver", "xpbd") or "xpbd").lower()

        if solver_name in {"mujoco", "solvermujoco"}:
            # MuJoCo: Use explicit box geometry as ground
            friction = ground_cfg.get("friction", 0.8)
            restitution = ground_cfg.get("restitution", 0.0)
            thickness = ground_cfg.get("thickness", 0.02)

            ground_cfg_newton = newton.ModelBuilder.ShapeConfig()
            ground_cfg_newton.density = 0.0  # Static body
            ground_cfg_newton.ke = 1.0e5     # Contact stiffness
            ground_cfg_newton.kd = 1.0e3     # Contact damping
            ground_cfg_newton.mu = friction
            ground_cfg_newton.restitution = restitution

            self.scene_builder.builder.add_shape_box(
                body=-1,  # World body
                hx=100.0,
                hy=thickness,
                hz=100.0,
                xform=wp.transform(
                    wp.vec3(0.0, -thickness, 0.0),
                    wp.quat_identity()
                ),
                cfg=ground_cfg_newton
            )
            log.ok("Newton backend: Added MuJoCo-compatible ground (explicit box geometry)")
        else:
            # XPBD/Featherstone/etc: Use native ground plane
            self.scene_builder.builder.add_ground_plane()
            log.ok("Newton backend: Added native ground plane")

    def initialize(self) -> None:
        self.ready = False
        sim_cfg = self.global_config.get("simulator", {}).get("config", {})

        self.sim_frequency = float(sim_cfg.get("sim_frequency", 240.0))
        self.sim_substeps = max(int(sim_cfg.get("substeps", 1)), 1)
        base_dt = 1.0 / self.sim_frequency if self.sim_frequency > 0 else 0.005
        self.set_time_step(base_dt)

        # Create NewtonBuilder instead of raw ModelBuilder
        gravity = tuple(sim_cfg.get("gravity", [0.0, 0.0, -9.81]))
        up_axis = self._determine_up_axis(gravity)
        gravity_magnitude = self._extract_gravity_magnitude(gravity)

        self.scene_builder = NewtonBuilder(
            model_name="ark_world",
            up_axis=up_axis,
            gravity=gravity_magnitude
        )

        # Create solver-specific adapter early (before scene building)
        solver_name = sim_cfg.get("solver", "xpbd") or "xpbd"
        self.adapter = self._create_scene_adapter(solver_name)
        log.info(f"Newton backend: Using {self.adapter.solver_name} solver adapter")

        device_name = sim_cfg.get("device")
        if device_name:
            try:
                wp.set_device(device_name)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"Newton backend: unable to select device '{device_name}': {exc}")

        self._apply_joint_defaults(sim_cfg)

        # Add ground plane if requested (adapter handles solver-specific implementation)
        ground_cfg = self.global_config.get("ground_plane", {})
        if ground_cfg.get("enabled", False):
            from ark.system.newton.geometry_descriptors import GeometryDescriptor
            descriptor = GeometryDescriptor.from_ground_plane_config(ground_cfg)
            self.adapter.adapt_ground_plane(descriptor)

        # Add robots FIRST - URDF loader must come before primitives
        # Otherwise Newton's add_urdf overwrites primitive body indices
        if self.global_config.get("robots"):
            for robot_name, robot_cfg in self.global_config["robots"].items():
                self.add_robot(robot_name, robot_cfg)

        # Add objects AFTER robots to preserve body indices
        if self.global_config.get("objects"):
            for obj_name, obj_cfg in self.global_config["objects"].items():
                obj_type = obj_cfg.get("type", "primitive")
                self.add_sim_component(obj_name, obj_type, obj_cfg)

        if self.global_config.get("sensors"):
            for sensor_name, sensor_cfg in self.global_config["sensors"].items():
                from ark.system.driver.sensor_driver import SensorType
                sensor_type = SensorType(sensor_cfg.get("type", "camera").upper())
                self.add_sensor(sensor_name, sensor_type, sensor_cfg)

        # Finalize model via NewtonBuilder and get metadata
        self.model, self.scene_metadata = self.scene_builder.finalize(
            device=device_name or "cuda:0"
        )

        if self.model is None:
            log.error("Newton backend: Model finalization failed, returned None")
            raise RuntimeError("Failed to finalize Newton model")

        log.ok(
            f"Newton backend: Model finalized successfully - "
            f"{self.model.joint_count} joints, {self.model.body_count} bodies"
        )

        # Create solver through adapter (handles solver-specific configuration)
        self.solver = self.adapter.create_solver(self.model, sim_cfg)

        self.state_current = self.model.state()
        self.state_next = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_current)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_current)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_next)

        self._state_accessor: Callable[[], newton.State] = lambda: self.state_current
        self._bind_runtime_handles()

        # Sync state_next from state_current after drivers apply initial configurations
        # This is critical because drivers only modify state_current via state_accessor,
        # and Newton swaps buffers during stepping. Both states must be synchronized.
        if self.state_current.joint_q is not None and self.state_next.joint_q is not None:
            self.state_next.joint_q.assign(self.state_current.joint_q)
            self.state_next.joint_qd.assign(self.state_current.joint_qd)
            newton.eval_fk(self.model, self.state_next.joint_q, self.state_next.joint_qd, self.state_next)
            log.info("Newton backend: Synchronized state_next from state_current after initial config")

        # CRITICAL FIX: Initialize control.joint_target from current state
        # Without this, control.joint_target starts at zeros and PD controller
        # drives all joints toward zero instead of maintaining current positions!
        # This follows Newton's own examples (see example_basic_urdf.py:72)
        if self.control.joint_target is not None and self.state_current.joint_q is not None:
            self.control.joint_target.assign(self.state_current.joint_q)
            target_sample = self.control.joint_target.numpy()[:min(7, len(self.control.joint_target))]
            log.ok(f"Newton backend: Initialized control.joint_target from state: {target_sample}")
        else:
            log.error("Newton backend: FAILED to initialize control.joint_target - array is None!")

        # Initialize viewer manager
        self.viewer_manager = NewtonViewerManager(sim_cfg, self.model)

        # Log successful initialization
        log.ok(
            f"Newton backend: Initialized with "
            f"{len(self.robot_ref)} robot(s), "
            f"{len(self.sensor_ref)} sensor(s), "
            f"{len(self.object_ref)} object(s)"
        )

        self.ready = True

    def _bind_runtime_handles(self) -> None:
        state_accessor = lambda: self.state_current
        bound_robots = 0
        bound_objects = 0
        bound_sensors = 0

        for robot in self.robot_ref.values():
            driver = getattr(robot, "_driver", None)
            if isinstance(driver, NewtonRobotDriver):
                driver.bind_runtime(self.model, self.control, state_accessor, self._substep_dt)
                bound_robots += 1

        for obj in self.object_ref.values():
            if isinstance(obj, NewtonMultiBody):
                obj.bind_runtime(self.model, state_accessor)
                bound_objects += 1

        for sensor in self.sensor_ref.values():
            driver = getattr(sensor, "_driver", None)
            if isinstance(driver, NewtonCameraDriver):
                # Pass viewer_manager for RGB capture capability
                driver.bind_runtime(self.model, state_accessor, viewer_manager=self.viewer_manager)
                bound_sensors += 1
            elif isinstance(driver, NewtonLiDARDriver):
                driver.bind_runtime(self.model, state_accessor)
                bound_sensors += 1

        log.info(
            f"Newton backend: Bound runtime handles - "
            f"{bound_robots} robots, {bound_objects} objects, {bound_sensors} sensors"
        )

    def _create_scene_adapter(self, solver_name: str):
        """Factory method to create solver-specific scene adapter.

        Creates the appropriate adapter based on solver name. Adapters handle
        solver-specific scene building requirements (e.g., MuJoCo ground plane
        workaround) in a transparent, maintainable way.

        Args:
            solver_name: Name of the solver ("xpbd", "mujoco", "featherstone", etc.)

        Returns:
            Solver-specific adapter instance

        Example:
            >>> adapter = self._create_scene_adapter("mujoco")
            >>> adapter.adapt_ground_plane(descriptor)  # Uses box workaround
        """
        from ark.system.newton.scene_adapters import (
            XPBDAdapter,
            MuJoCoAdapter,
        )

        # Map solver names to adapter classes
        adapter_map = {
            "xpbd": XPBDAdapter,
            "solverxpbd": XPBDAdapter,
            "mujoco": MuJoCoAdapter,
            "solvermujoco": MuJoCoAdapter,
            # TODO: Add Featherstone and SemiImplicit adapters in future
        }

        # Get adapter class (default to XPBD if unknown)
        solver_key = solver_name.lower()
        adapter_cls = adapter_map.get(solver_key)

        if not adapter_cls:
            log.warning(
                f"Unknown solver '{solver_name}', falling back to XPBD adapter"
            )
            adapter_cls = XPBDAdapter

        return adapter_cls(self.scene_builder)

    def _create_solver(self, sim_cfg: dict[str, Any]) -> newton.solvers.SolverBase:
        solver_name = sim_cfg.get("solver", "xpbd")
        iterations = int(sim_cfg.get("solver_iterations", 1))

        name = (solver_name or "xpbd").lower()
        if name in {"xpbd", "solverxpbd"}:
            # XPBD requires more iterations than MuJoCo for position control convergence
            # Newton examples use 20 effective iterations (2 base × 10 substeps)
            # Increase to minimum of 8 iterations for reliable TARGET_POSITION mode
            xpbd_iterations = max(iterations, 8)
            solver = newton.solvers.SolverXPBD(self.model, iterations=xpbd_iterations)
            if xpbd_iterations > iterations:
                log.info(f"Newton backend: Using XPBD solver with {xpbd_iterations} iterations (increased from config value {iterations} for position control)")
            else:
                log.info(f"Newton backend: Using XPBD solver with {xpbd_iterations} iterations")
            return solver
        if name in {"semiimplicit", "semi_implicit", "solversemiimplicit"}:
            return newton.solvers.SolverSemiImplicit(self.model)
        if name in {"featherstone", "solverfeatherstone"}:
            return newton.solvers.SolverFeatherstone(self.model)
        if name in {"mujoco", "solvermujoco"}:
            return newton.solvers.SolverMuJoCo(self.model)
        log.warning(f"Unknown Newton solver '{solver_name}', falling back to XPBD.")
        return newton.solvers.SolverXPBD(self.model, iterations=iterations)

    def set_gravity(self, gravity: tuple[float, float, float]) -> None:
        """Set gravity (only works before builder is created)."""
        gx, gy, gz = gravity
        axis_names = {0: "X", 1: "Y", 2: "Z"}
        components = np.array([gx, gy, gz], dtype=float)
        idx = int(np.argmax(np.abs(components)))
        magnitude = float(components[idx]) if np.any(components) else -9.81

        # This method is called before builder is created, so we just log
        # The actual gravity is set during NewtonBuilder initialization
        log.info(
            f"Newton backend: Gravity set to {magnitude:.2f} m/s² along {axis_names[idx]}-axis "
            f"(input: [{gx}, {gy}, {gz}])"
        )

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
        # NOTE: Don't call add_articulation() here - Newton's add_urdf() creates
        # articulations automatically. Calling it here would create duplicates.

        class_path = Path(robot_config["class_dir"])
        if class_path.is_file():
            class_path = class_path.parent
        RobotClass, driver_enum = import_class_from_directory(class_path)
        driver_cls = getattr(driver_enum, "value", driver_enum) or NewtonRobotDriver

        # Pass scene_builder.builder (raw Newton ModelBuilder) to driver
        driver = driver_cls(name, robot_config, builder=self.scene_builder.builder)
        robot = RobotClass(name=name, global_config=self.global_config, driver=driver)
        self.robot_ref[name] = robot

    def add_sim_component(self, name: str, type: str, obj_config: dict[str, Any]) -> None:
        """Add a generic simulation object.

        Args:
            name: Name of the object.
            type: Type identifier (e.g. "cube", "sphere").
            obj_config: Configuration dictionary for the object.
        """
        # Pass the wrapped builder to NewtonMultiBody
        component = NewtonMultiBody(
            name=name,
            builder=self.scene_builder.builder,
            global_config=self.global_config
        )
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

        # Determine driver class based on sensor type in config
        config_type = sensor_config.get("type", "camera").lower()
        if driver_enum is not None:
            # Use custom driver if specified in module
            driver_cls = getattr(driver_enum, "value", driver_enum)
        elif config_type == "lidar":
            driver_cls = NewtonLiDARDriver
        else:
            # Default to camera driver
            driver_cls = NewtonCameraDriver

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

        # Note: Do NOT call eval_fk() here - Newton's viewer.log_state() internally
        # handles FK computation when updating shape transforms for rendering.
        self.viewer_manager.render(self.state_current, self.contacts, self._simulation_time)

    def shutdown_backend(self) -> None:
        if hasattr(self, 'viewer_manager'):
            self.viewer_manager.shutdown()
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
