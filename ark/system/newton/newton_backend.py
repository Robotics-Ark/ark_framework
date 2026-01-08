"""Newton backend integration for the ARK simulator."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
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
from ark.client.frequencies.rate import Rate
from arktypes import *


def import_class_from_directory(path: Path) -> tuple[type, Optional[type]]:
    """!Load a class from ``path``.

    The helper searches for ``<ClassName>.py`` inside ``path`` and imports the
    class with the same name. If a ``Drivers`` class is present in the module
    its ``NEWTON_DRIVER`` attribute is returned alongside the main class.

    @param path Path to the directory containing the module.
    @return Tuple ``(cls, driver_cls)`` where ``driver_cls`` is ``None`` when no
            driver is defined.
    @rtype Tuple[type, Optional[type]]
    """

    def _resolve_driver_entry(entry: Any | None, module_dir: str) -> Optional[type]:
        if entry is None:
            return None
        if isinstance(entry, type):
            return entry
        if hasattr(entry, "value"):
            entry = entry.value
        if isinstance(entry, type):
            return entry
        if isinstance(entry, str):
            module_path, class_name = entry.rsplit(".", 1)
            try:
                module = importlib.import_module(module_path)
            except ModuleNotFoundError:
                module_file = Path(module_dir) / f"{module_path.split('.')[-1]}.py"
                if not module_file.exists():
                    raise
                spec = importlib.util.spec_from_file_location(module_path, module_file)
                if spec is None or spec.loader is None:
                    raise ImportError(
                        f"Could not load module from '{module_file}' for driver '{entry}'"
                    )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            return getattr(module, class_name)
        if hasattr(entry, "load") and callable(entry.load):
            return entry.load()
        return None

    class_name = path.name
    file_path = (path / f"{class_name}.py").resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=str(file_path))

    module_dir = os.path.dirname(file_path)
    sys.path.insert(0, module_dir)

    try:
        class_names = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ]

        spec = importlib.util.spec_from_file_location(class_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec from {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[class_name] = module
        spec.loader.exec_module(module)

        driver_cls = None
        drivers_cls = getattr(module, "Drivers", None)
        if isinstance(drivers_cls, type):
            driver_entry = getattr(drivers_cls, "NEWTON_DRIVER", None)
            driver_cls = _resolve_driver_entry(driver_entry, module_dir)

        if class_name in class_names:
            target_class = getattr(module, class_name)
        else:
            non_driver_classes = [name for name in class_names if name != "Drivers"]
            if len(non_driver_classes) != 1:
                raise ValueError(
                    f"Expected a single class definition in {file_path}, found {len(non_driver_classes)}."
                )
            target_class = getattr(module, non_driver_classes[0])

        return target_class, driver_cls
    finally:
        sys.path.pop(0)
        sys.modules.pop(class_name, None)


# def import_class_from_directory(path: Path) -> tuple[type[Any], Optional[type[Any]]]:
#     """Import a class (and optional driver enum) from ``path``.
#         The helper searches for ``<ClassName>.py`` inside ``path`` and imports the
#     class with the same name.  If a ``Drivers`` class is present in the module
#     its ``NEWTON_DRIVER`` attribute is returned alongside the main class.
#
#     @param path Path to the directory containing the module.
#     @return Tuple ``(cls, driver_cls)`` where ``driver_cls`` is ``None`` when no
#             driver is defined.
#     @rtype Tuple[type, Optional[type]]
#
#     """
#
#     ## Extract the class name from the last part of the directory path (last directory name)
#     class_name = path.name
#     file_path = (
#         path / f"{class_name}.py"
#     ).resolve()  ##just add the resolve here instead of newline
#     ## Defensive check for the filepath, raise error if not found
#     if not file_path.exists():
#         raise FileNotFoundError(f"The file {file_path} does not exist.")
#
#     with open(file_path, "r", encoding="utf-8") as handle:
#         tree = ast.parse(handle.read(), filename=str(file_path))
#
#     module_dir = str(file_path.parent)
#     sys.path.insert(0, module_dir)
#     ## Import the module dynamically and extract class names defensively
#     try:
#         class_names = [
#             node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
#         ]
#         drivers_attr: Optional[type[Any]] = None
#
#         spec = importlib.util.spec_from_file_location(class_name, file_path)
#         if spec is None or spec.loader is None:
#             raise ImportError(f"Could not create module spec for {file_path}")
#         module = importlib.util.module_from_spec(spec)
#         sys.modules[class_name] = module
#         spec.loader.exec_module(module)
#
#         if "Drivers" in class_names:
#             # Load the module dynamically
#             spec = importlib.util.spec_from_file_location(class_names[0], file_path)
#             module = importlib.util.module_from_spec(spec)
#             sys.modules[class_names[0]] = module
#             spec.loader.exec_module(module)
#
#             class_ = getattr(module, class_names[0])
#             sys.path.pop(0)
#
#             breakpoint()
#             drivers = class_.NEWTON_DRIVER.load()
#             class_names.remove("Drivers")
#
#         target_name = class_names[0] if class_names else class_name
#         target_cls = getattr(module, target_name)
#     finally:
#         sys.path.pop(0)
#         sys.modules.pop(class_name, None)
#
#     return target_cls, drivers


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
            log.info(
                "Newton backend: No joint_defaults in config, using Newton defaults"
            )
            return

        # Build kwargs dict for set_default_joint_config
        joint_defaults = {}
        mode_str = joint_cfg.get("mode", "").upper()
        if mode_str:
            log.info(
                "Newton backend: joint_defaults.mode is deprecated in newton-physics; "
                "using target_pos/target_vel + gains instead."
            )
        if "target_pos" in joint_cfg:
            joint_defaults["target_pos"] = float(joint_cfg["target_pos"])
        if "target_vel" in joint_cfg:
            joint_defaults["target_vel"] = float(joint_cfg["target_vel"])
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

    def initialize(self) -> None:
        self.ready = False
        sim_cfg = self.global_config.get("simulator", {}).get("config", {})
        # Ensure namespace and channel metadata exist for downstream components
        if "namespace" not in self.global_config:
            namespace = self.global_config.get("simulator", {}).get("namespace", "ark")
            self.global_config["namespace"] = namespace
        self.global_config.setdefault("observation_channels", None)
        self.global_config.setdefault("action_channels", None)

        self.sim_frequency = float(sim_cfg.get("sim_frequency", 240.0))
        self.sim_substeps = max(int(sim_cfg.get("substeps", 1)), 1)
        base_dt = 1.0 / self.sim_frequency if self.sim_frequency > 0 else 0.005
        self.set_time_step(base_dt)

        # Create NewtonBuilder instead of raw ModelBuilder
        gravity = tuple(sim_cfg.get("gravity", [0.0, 0.0, -9.81]))
        up_axis = self._determine_up_axis(gravity)
        gravity_magnitude = self._extract_gravity_magnitude(gravity)

        self.scene_builder = NewtonBuilder(
            model_name="ark_world", up_axis=up_axis, gravity=gravity_magnitude
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
                log.warning(
                    f"Newton backend: unable to select device '{device_name}': {exc}"
                )

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

                sensor_type = SensorType(sensor_cfg.get("type", "camera").lower())
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

        # Apply safety multiplier to rigid_contact_max to handle complex mesh collisions
        # Newton calculates a conservative contact limit, but complex mesh-mesh interactions
        # (like Franka Panda's 10 STL collision geometries) can exceed this estimate.
        # The multiplier provides headroom for contact-rich scenarios without mesh simplification.
        newton_cfg = sim_cfg.get("newton_physics", {})
        contact_multiplier = float(newton_cfg.get("rigid_contact_multiplier", 4.0))
        original_max = self.model.rigid_contact_max
        self.model.rigid_contact_max = int(original_max * contact_multiplier)
        log.info(
            f"Newton backend: Increased rigid_contact_max from {original_max} to "
            f"{self.model.rigid_contact_max} (multiplier={contact_multiplier})"
        )

        # Create solver through adapter (handles solver-specific configuration)
        self.solver = self.adapter.create_solver(self.model, sim_cfg)

        self.state_current = self.model.state()
        self.state_next = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_current)

        # Use model arrays for initial FK (these contain initial config from builder)
        # This matches Newton's own examples (see test_franka_standalone.py)
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_current
        )
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_next
        )

        self._state_accessor: Callable[[], newton.State] = lambda: self.state_current

        # Initialize viewer manager
        self.viewer_manager = NewtonViewerManager(sim_cfg, self.model)
        if self.viewer_manager.gui_enabled:
            # When GUI is active, step physics from the main thread to keep GL interop happy.
            self.custom_event_loop = self._viewer_event_loop

        self._bind_runtime_handles()

        # NOTE: No state sync needed here - both state_current and state_next were already
        # initialized with eval_fk using model.joint_q above. Since drivers no longer call
        # _apply_initial_configuration() during bind_runtime, there's no state modification
        # to synchronize. The initial config was applied to builder before finalize.

        # CRITICAL FIX: Initialize control.joint_target_pos from current state
        # Without this, control.joint_target_pos starts at zeros and PD controller
        # drives all joints toward zero instead of maintaining current positions!
        # This follows Newton's own examples (see example_basic_urdf.py:72)
        if self.control.joint_target_pos is not None and self.state_current.joint_q is not None:
            self.control.joint_target_pos.assign(self.state_current.joint_q)
            target_sample = self.control.joint_target_pos.numpy()[:min(7, len(self.control.joint_target_pos))]
            log.ok(f"Newton backend: Initialized control.joint_target_pos from state: {target_sample}")
        else:
            log.error("Newton backend: FAILED to initialize control.joint_target_pos - array is None!")

        if self.control.joint_target_vel is not None:
            self.control.joint_target_vel.zero_()

        # # Initialize viewer manager
        # self.viewer_manager = NewtonViewerManager(sim_cfg, self.model)
        # if self.viewer_manager.gui_enabled:
        #     # When GUI is active, step physics from the main thread to keep GL interop happy.
        #     self.custom_event_loop = self._viewer_event_loop

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
                driver.bind_runtime(
                    self.model, self.control, state_accessor, self._substep_dt
                )
                bound_robots += 1

        for obj in self.object_ref.values():
            if isinstance(obj, NewtonMultiBody):
                obj.bind_runtime(self.model, state_accessor)
                bound_objects += 1

        for sensor in self.sensor_ref.values():
            driver = getattr(sensor, "_driver", None)
            if isinstance(driver, NewtonCameraDriver):
                # Pass viewer_manager for RGB capture capability
                driver.bind_runtime(
                    self.model, state_accessor, viewer_manager=self.viewer_manager
                )
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
            FeatherstoneAdapter,
        )

        # Map solver names to adapter classes
        adapter_map = {
            "xpbd": XPBDAdapter,
            "solverxpbd": XPBDAdapter,
            "mujoco": MuJoCoAdapter,
            "solvermujoco": MuJoCoAdapter,
            "featherstone": FeatherstoneAdapter,
            "solverfeatherstone": FeatherstoneAdapter,
        }

        # Get adapter class (default to XPBD if unknown)
        solver_key = solver_name.lower()
        adapter_cls = adapter_map.get(solver_key)

        if not adapter_cls:
            log.warning(f"Unknown solver '{solver_name}', falling back to XPBD adapter")
            adapter_cls = XPBDAdapter

        return adapter_cls(self.scene_builder)

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

    def add_sim_component(
        self, name: str, _type: str, _obj_config: dict[str, Any]
    ) -> None:
        """Add a generic simulation object.

        Args:
            name: Name of the object.
            _type: Type identifier (unused, NewtonMultiBody reads from global_config).
            _obj_config: Configuration dictionary (unused, read from global_config).
        """
        component = NewtonMultiBody(
            name=name,
            builder=self.scene_builder.builder,
            global_config=self.global_config,
        )
        self.object_ref[name] = component

    def add_sensor(
        self, name: str, _sensor_type: Any, sensor_config: dict[str, Any]
    ) -> None:
        """Add a sensor to the simulation.

        Args:
            name: Name of the sensor.
            _sensor_type: SensorType enum (unused, type determined from sensor_config).
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

    def _viewer_event_loop(self, sim_node) -> None:
        """Run Newton simulation when GUI viewer requires main-thread ownership."""
        sim_cfg = self.global_config.get("simulator", {}).get("config", {})
        node_hz = float(sim_cfg.get("node_frequency", 240.0))
        rate = Rate(node_hz, reset=True) if node_hz > 0 else None

        lcm_handles = [sim_node._lcm]
        for registry in (self.robot_ref, self.sensor_ref, self.object_ref):
            for component in registry.values():
                lc = getattr(component, "_lcm", None)
                if lc is not None:
                    lcm_handles.append(lc)

        log.info(
            "Newton backend: GUI viewer active - running simulation loop on main thread"
        )

        while not sim_node._done:
            if not self.viewer_manager or not self.viewer_manager.is_running():
                log.info("Newton backend: Viewer closed - stopping simulation loop")
                break

            try:
                for lc in lcm_handles:
                    lc.handle_timeout(0)
            except OSError as exc:
                log.warning(f"Newton backend: LCM error in viewer loop: {exc}")
                break

            self._spin_sim_components()
            sim_node._step_simulation()

            if rate is not None:
                rate.sleep()

        sim_node._done = True

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
        self.viewer_manager.render(
            self.state_current, self.contacts, self._simulation_time
        )

    def shutdown_backend(self) -> None:
        if hasattr(self, "viewer_manager"):
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
