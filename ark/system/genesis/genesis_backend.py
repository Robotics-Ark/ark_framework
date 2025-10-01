from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import cv2
import genesis as gs
import numpy as np

from ark.tools.log import log
from ark.system.simulation.simulator_backend import SimulatorBackend

from ark.system.genesis.genesis_multibody import GenesisMultiBody


def import_class_from_directory(path: Path) -> tuple[type[Any], Any | None]:
    """Load and return a class (and optional driver) from ``path``.

    The helper searches for ``<ClassName>.py`` inside ``path`` and imports the
    class with the same name.  When the module exposes a ``Drivers`` class a
    ``GENESIS_DRIVER`` attribute is returned alongside the main class.
    """

    class_name = path.name
    file_path = (path / f"{class_name}.py").resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    module_dir = str(file_path.parent)
    sys.path.insert(0, module_dir)

    try:
        spec = importlib.util.spec_from_file_location(class_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[class_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(class_name, None)
        sys.path.pop(0)

    drivers_attr: Any | None = None
    drivers_cls = getattr(module, "Drivers", None)
    if isinstance(drivers_cls, type):
        drivers_attr = getattr(drivers_cls, "GENESIS_DRIVER", None)

    class_candidates = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type) and obj.__module__ == module.__name__
    ]

    target_class = next(
        (cls for cls in class_candidates if cls.__name__ == class_name), None
    )
    if target_class is None:
        non_driver_classes = [
            cls for cls in class_candidates if cls.__name__ != "Drivers"
        ]
        if len(non_driver_classes) != 1:
            raise ValueError(
                f"Expected a single class definition in {file_path}, found {len(non_driver_classes)}."
            )
        target_class = non_driver_classes[0]

    return target_class, drivers_attr


class GenesisBackend(SimulatorBackend):
    """Backend wrapper around the Genesis client.

    This class handles scene creation, stepping the simulation and managing
    simulated components such as robots, objects and sensors.
    """

    def initialize(self) -> None:
        """!Initialize the Genesis world.

        The method creates the Genesis client, configures gravity and time step
        and loads all robots, objects and sensors defined in
        ``self.global_config``.  Optional frame capture settings are applied as
        well.
        """
        self.ready = False
        self._is_initialised = False
        self.scene: gs.Scene | None = None
        self.scene_ready: bool = False

        connection_mode = (
            self.global_config["simulator"]["config"]["connection_mode"]
        )
        show_viewer = connection_mode.upper() == "GUI"

        gs.init(backend=gs.cpu)

        gravity = self.global_config["simulator"]["config"].get(
            "gravity", [0.0, 0.0, -9.81]
        )
        timestep = 1.0 / self.global_config["simulator"]["config"].get(
            "sim_frequency", 100
        )

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=timestep, gravity=gravity),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.Plane())

        # Optional off-screen rendering
        self.save_render_config: dict[str, Any] | None = self.global_config[
            "simulator"
        ].get("save_render")
        if self.save_render_config:
            self.render_cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=False,
            )
            self.save_path = Path(
                self.save_render_config.get("save_path", "output/save_render")
            )
            self.save_path.mkdir(parents=True, exist_ok=True)

            remove_existing = self.save_render_config.get("remove_existing", True)
            if remove_existing:
                for child in self.save_path.iterdir():
                    if child.is_file():
                        child.unlink()
            self.save_interval = float(
                self.save_render_config.get("save_interval", 1 / 30)
            )
            self.overwrite_file = bool(
                self.save_render_config.get("overwrite_file", False)
            )
        else:
            self.render_cam = None
            self.save_path = None
            self.save_interval = 0.0
            self.overwrite_file = False

        # Setup robots
        if self.global_config.get("robots", None):
            for robot_name, robot_config in self.global_config["robots"].items():
                self.add_robot(robot_name, robot_config)

        # Setup objects
        if self.global_config.get("objects", None):
            for obj_name, obj_config in self.global_config["objects"].items():
                self.add_sim_component(obj_name, obj_config)

        # Sensors have to be set up last, as e.g. cameras might need
        # a parent to attach to
        if self.global_config.get("sensors", None):
            for sensor_name, sensor_config in self.global_config["sensors"].items():
                self.add_sensor(sensor_name, sensor_config)

        self.ready = True

    def is_ready(self) -> bool:
        """!Check whether the backend has finished initialization.

        @return ``True`` once all components were created and the simulator is
                ready for stepping.
        @rtype bool
        """
        return self.ready

    def set_gravity(self, gravity: tuple[float, float, float]) -> None:
        """!Set the world gravity.

        @param gravity Tuple ``(gx, gy, gz)`` specifying gravity in m/s^2.
        """
        raise NotImplementedError("Not required for Genesis")

    def set_time_step(self, time_step: float) -> None:
        """!Set the simulation timestep.

        @param time_step Length of a single simulation step in seconds.
        """
        raise NotImplementedError("Not required for Genesis")

    ##########################################################
    ####            ROBOTS, SENSORS AND OBJECTS           ####
    ##########################################################

    def add_robot(self, name: str, robot_config: dict[str, Any]) -> None:
        """!Instantiate and register a robot in the simulation.

        @param name Identifier for the robot.
        @param robot_config Robot specific configuration dictionary.
        """
        class_path = Path(robot_config["class_dir"])
        if class_path.is_file():
            class_path = class_path.parent

        robot_class, driver_entry = import_class_from_directory(class_path)

        if driver_entry is None:
            raise ValueError(
                f"Genesis driver not defined for robot '{name}' at {class_path}."
            )

        driver_cls = getattr(driver_entry, "value", driver_entry)
        if self.scene is None:
            raise RuntimeError("Genesis scene is not initialized.")

        driver = driver_cls(name, robot_config, self.scene)
        robot = robot_class(name=name, global_config=self.global_config, driver=driver)

        self.robot_ref[name] = robot

    def add_sim_component(
        self,
        name: str,
        obj_config: dict[str, Any],
    ) -> None:
        """!Add a generic simulated object.

        @param name Name of the object.
        @param obj_config Object specific configuration dictionary.
        """
        if self.scene is None:
            raise RuntimeError("Genesis scene is not initialized.")

        sim_component = GenesisMultiBody(
            name=name, client=self.scene, global_config=self.global_config
        )
        self.object_ref[name] = sim_component

    def add_sensor(self, name: str, sensor_config: dict[str, Any]) -> None:
        """!Instantiate and register a sensor.

        @param name Name of the sensor component.
        @param sensor_config Sensor configuration dictionary.
        """
        raise NotImplementedError("Sensors are not compatible with Genesis yet.")
        # Cameras are not supported on MacOS, Ubuntu Cameras are not working
        # Genesis-Embodied-AI/Genesis#1739

    def remove(self, name: str) -> None:
        """!Remove a component from the simulator.

        @param name Name of the robot, object or sensor to remove.
        """
        raise NotImplementedError("Genesis does not support removing components.")

    #######################################
    ####          SIMULATION           ####
    #######################################

    def _all_available(self) -> bool:
        """Return ``True`` when all registered components are active."""

        robots_ready = all(not robot._is_suspended for robot in self.robot_ref.values())
        objects_ready = all(not obj._is_suspended for obj in self.object_ref.values())
        sensors_ready = all(
            not sensor._is_suspended for sensor in self.sensor_ref.values()
        )
        return robots_ready and objects_ready and sensors_ready
    
    def save_render(self) -> None:
        """Add the latest render to save folder if rendering is configured."""

        if self.render_cam is None or self.save_path is None:
            return

        rgba = self.render_cam.render()
        time_us = int(1e6 * self._simulation_time)
        if self.overwrite_file:
            save_path = self.save_path / "render.png"
        else:
            save_path = self.save_path / f"{time_us}.png"
        # Convert renderer output to uint8 BGR image for OpenCV
        img = np.asarray(rgba)
        # Drop alpha channel if present
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        # Normalize to uint8 if needed (assume float in [0,1])
        if img.dtype != np.uint8:
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255.0).astype(np.uint8)
        # Convert RGB -> BGR for OpenCV
        img_bgr = img[..., ::-1]
        cv2.imwrite(str(save_path), img_bgr)


    def step(self) -> None:
        """!Advance the simulation by one timestep.

        The method updates all registered components, advances the physics
        engine and optionally saves renders when enabled.
        """
        if self.scene is None:
            raise RuntimeError("Genesis scene is not initialized.")

        if not self.scene_ready:
            self.scene.build()
            self.scene_ready = True

        if not self._all_available():
            log.warn("Skipping simulation step because a component is suspended.")
            return

        self._step_sim_components()
        self.scene.step()
        if self.save_render_config:
            self.save_render()

    def reset_simulator(self) -> None:
        """!Reset the entire simulator state.

        All robots, objects and sensors are destroyed and the backend is
        re-initialized using ``self.global_config``.
        """
        raise NotImplementedError("Reset simulator not implemented yet.")

    def get_current_time(self) -> float:
        """!Return the current simulation time.

        @return Elapsed simulation time in seconds.
        @rtype float
        """
        return self.scene.t

    def shutdown_backend(self) -> None:
        """!Disconnect all components and shut down the backend.

        This should be called at program termination to cleanly close the
        simulator and free all resources.
        """
        for robot in self.robot_ref.values():
            robot.kill_node()
        for obj in self.object_ref.values():
            obj.kill_node()
        for sensor in self.sensor_ref.values():
            sensor.kill_node()
