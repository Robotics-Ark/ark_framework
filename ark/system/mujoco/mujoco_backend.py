import importlib.util
import sys, ast, os
from pathlib import Path
from typing import Any, Optional

import mujoco
import mujoco.viewer

from ark.system.mujoco.mjcf_builder import MJCFBuilder

from ark.tools.log import log
from ark.system.simulation.simulator_backend import SimulatorBackend
from ark.system.mujoco.mujoco_multibody import MujocoMultiBody
from arktypes import *

import textwrap


def import_class_from_directory(path: Path) -> tuple[type, Optional[type]]:
    """!Load a class from ``path``.

    The helper searches for ``<ClassName>.py`` inside ``path`` and imports the
    class with the same name.  If a ``Drivers`` class is present in the module
    its ``MUJOCO_DRIVER`` attribute is returned alongside the main class.

    @param path Path to the directory containing the module.
    @return Tuple ``(cls, driver_cls)`` where ``driver_cls`` is ``None`` when no
            driver is defined.
    @rtype Tuple[type, Optional[type]]
    """
    # Extract the class name from the last part of the directory path (last directory name)
    class_name = path.name
    file_path = path / f"{class_name}.py"

    # get the full absolute path
    file_path = file_path.resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=file_path)

    # for imports
    module_dir = os.path.dirname(file_path)
    sys.path.insert(0, module_dir)

    # Extract class names from the AST
    class_names = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    ]

    # check if Sensor_Drivers is in the class_names
    if "Drivers" in class_names:
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location(class_names[0], file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[class_names[0]] = module
        spec.loader.exec_module(module)

        class_ = getattr(module, class_names[0])
        sys.path.pop(0)

        drivers = class_.MUJOCO_DRIVER
        class_names.remove("Drivers")

    # Retrieve the class from the module (has to be list of one)
    class_ = getattr(module, class_names[0])

    if len(class_names) != 1:
        raise ValueError(
            f"Expected exactly two class definition in {file_path}, but found {len(class_names)}."
        )

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(class_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[class_name] = module
    spec.loader.exec_module(module)

    # Retrieve the class from the module (has to be list of one)
    class_ = getattr(module, class_names[0])
    sys.path.pop(0)

    # Return the class
    return class_, drivers


class MujocoBackend(SimulatorBackend):

    def initialize(self) -> None:
        """!Initialize the MuJoCo simulation backend."""
        self.builder = MJCFBuilder("ARK Mujoco").set_compiler(
            angle="radian", meshdir="ark_mujoco_assets"
        )

        gravity = self.global_config["simulator"]["config"].get(
            "gravity", [0, 0, -9.81]
        )
        self.set_gravity(gravity)

        if self.global_config.get("objects", None):
            for object_name, object_config in self.global_config["objects"].items():
                self.add_sim_component(object_name, object_config)

        if self.global_config.get("robots", None):
            for robot_name, robot_config in self.global_config["robots"].items():
                self.add_robot(robot_name, robot_config)

        if self.global_config.get("sensors", None):
            for sensor_name, sensor_config in self.global_config["sensors"].items():
                self.add_sensor(sensor_name, sensor_config)

        self.builder.make_spawn_keyframe(name="spawn")
        xml_string = self.builder.to_string(pretty=True)

        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)

        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview"
        )
        self.renderer = mujoco.Renderer(self.model, 100, 100)

        if (
            self.global_config["simulator"]["config"]["connection_mode"].upper()
            == "GUI"
        ):
            self.headless = False
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )
            # self.viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
            # self.viewer.cam.distance = 3
            # self.viewer.cam.azimuth = 0
            # self.viewer.cam.elevation = 0
        else:
            self.headless = True

        for object_name in self.object_ref:
            self.object_ref[object_name].update_ids(self.model, self.data)

        for sensor_name in self.sensor_ref:
            self.sensor_ref[sensor_name]._driver.update_ids(self.model, self.data)

        for robot_name in self.robot_ref:
            self.robot_ref[robot_name]._driver.update_ids(self.model, self.data)

        self.timestep = 1 / self.global_config["simulator"]["config"].get(
            "sim_frequency", 500.0
        )

    def set_gravity(self, gravity: tuple[float, float, float]) -> None:
        """!Set the gravity vector for the simulation.

        @param gravity Gravity vector ``(x, y, z)`` in ``m/s^2``.
        @return ``None``
        """
        self.builder.set_option(gravity=gravity)

    def reset_simulator(self) -> None:
        """!Reset the simulator to the initial keyframe."""
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "spawn")
        if key_id < 0:
            raise ValueError("Keyframe 'spawn' not found")

        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        if self.model.nu > 0:
            self.data.ctrl[:] = 0.0
        if self.model.na > 0:
            self.data.act[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def add_robot(
        self,
        name: str,
        robot_config: dict[str, Any],
    ) -> None:
        """!Add a robot to the simulation.

        @param name Name of the robot.
        @param robot_config Configuration dictionary for the robot.
        @return ``None``
        """
        class_path = Path(robot_config["class_dir"])
        if class_path.is_file():
            class_path = class_path.parent
        RobotClass, DriverClass = import_class_from_directory(class_path)
        DriverClass = DriverClass.value

        driver = DriverClass(name, component_config=robot_config, builder=self.builder)
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
        """!Add a sensor to the simulation.

        @param name Name of the sensor.
        @param sensor_config Configuration dictionary for the sensor.
        @return ``None``
        """
        class_path = Path(sensor_config["class_dir"])
        if class_path.is_file():
            class_path = class_path.parent

        SensorClass, DriverClass = import_class_from_directory(class_path)
        DriverClass = DriverClass.value

        attached_body_id = None
        if sensor_config["sim_config"].get("attach", None):
            raise NotImplementedError(
                "Attaching sensors to bodies is not implemented for MuJoCo yet."
            )

        driver = DriverClass(
            name, sensor_config, attached_body_id, builder=self.builder
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
        """!Register a static simulation component.

        @param name Name of the component.
        @param obj_config Configuration dictionary for the component.
        @return ``None``
        """
        sim_component = MujocoMultiBody(
            name=name, builder=self.builder, global_config=self.global_config
        )
        self.object_ref[name] = sim_component

    def _all_available(self) -> bool:
        """!Check whether all registered components are active.

        @return ``True`` if no component is suspended.
        @rtype bool
        """
        for robot_name in self.robot_ref:
            if self.robot_ref[robot_name]._is_suspended:
                return False
        for object_name in self.object_ref:
            if self.object_ref[object_name]._is_suspended:
                return False
        return True

    def remove(self, name: str) -> None:
        """!Remove a component from the simulation."""
        raise NotImplementedError("Mujoco does not support removing objects once loaded into XML.")

    def step(self) -> None:
        """!Step the simulator forward by one time step."""
        if self._all_available():
            self._step_sim_components()
            mujoco.mj_step(self.model, self.data)

            if not self.headless:
                self.viewer.sync()

            self._simulation_time = self.data.time
        else:
            log.panda("Did not step")

    def shutdown_backend(self) -> None:
        """!Shut down the simulation backend."""
        if not self.headless:
            self.viewer.close()
