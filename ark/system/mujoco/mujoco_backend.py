"""@file mujoco_backend.py
@brief Backend implementation for running simulations in Mujoco.
"""

from curses import window
import importlib.util
import sys, ast, os
import math
import cv2
from pathlib import Path
from typing import Any, Optional, Dict
import glfw
import numpy as np

import mujoco
import mujoco.viewer

from ark.system.mujoco.mjcf_builder import MJCFBuilder, BodySpec


from ark.tools.log import log
from ark.system.simulation.simulator_backend import SimulatorBackend
from ark.system.pybullet.pybullet_robot_driver import BulletRobotDriver
from ark.system.pybullet.pybullet_camera_driver import BulletCameraDriver
from ark.system.mujoco.mujoco_multibody import MujocoMultiBody
from ark.system.driver.sensor_driver import SensorType
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
        self.builder  = MJCFBuilder("ARK Mujoco").set_compiler(
            angle="radian",
            meshdir="ark_mujoco_assets"
        )

        # ===== Set Gravity =====
        gravity = self.global_config["simulator"]["config"].get(
            "gravity", [0, 0, -9.81]
        )
        self.set_gravity(gravity)


        # ===== Set Objects =====
        if self.global_config.get("objects", None):
            for obj_name, obj_config in self.global_config["objects"].items():
                self.add_sim_component(obj_name, obj_config)


        self.builder.add_texture("grid", type="2d", builtin="checker", width=512, height=512)
        self.builder.add_material("mat_grid", texture="grid", reflectance="0.2")
        # self.builder.add_body("floor")
        # self.builder.add_geom("floor", type="plane", size=[5, 5, 0.1], material="mat_grid")


        self.builder.make_spawn_keyframe(name="spawn")
        xml_string = self.builder.to_string(pretty=True)

        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)

        # SET UP THE PHYSICS SIMULATOR WITH GUI/DIRECT
        if (
            self.global_config["simulator"]["config"]["connection_mode"].upper()
            == "GUI"
        ):
            self.headless = False
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )
        else:
            # Launch the viewer in passive mode (headless)
            self.headless = True

        print(
            "MujocoBackend initialized in headless mode."
            if self.headless
            else "MujocoBackend initialized in GUI mode."
        )

        self.timestep = 1 / self.global_config["simulator"]["config"].get(
            "sim_frequency", 240.0
        )

    def compile_model(self, xml: str):
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        return model, data

    def build_world(self, world_xml: dict) -> str:

        return f"""
            <mujoco model="cube_camera">
  <compiler angle="degree" coordinate="local" />
  <option timestep="0.002" />
  <size />
  <asset /> 
  <default />
  <worldbody>
    <body name="floor">
      <geom type="plane" size="1 1 0.1" />
    </body>
    <body name="cube" pos="0 0 0.05">
      <joint type="free" name="cube_root" />
      <geom type="box" size="0.1 0.05 0.05" density="1000" rgba="1.0 0.4 0.2 1.0" />
    </body>
  </worldbody>
  <actuator />
  <sensor />
  <tendon />
  <equality />
  <contact />
  <keyframe />
</mujoco>        
            """

    def set_gravity(self, gravity: tuple[float, float, float]) -> str:
        self.builder.set_option(gravity=gravity)

    def reset_simulator(self) -> None:
        raise NotImplementedError(
            "Resetting the Mujoco simulator is not implemented yet."
        )

    def add_robot(
        self,
        name: str,
        global_config: dict[str, Any],
    ) -> None:
        raise NotImplementedError(
            "MujocoBackend does not support adding robots directly. Use MujocoMultiBody instead."
        )

    def add_sensor(
        self,
        name: str,
        sensor_type: SensorType,
        sensor_config: dict[str, Any],
    ) -> tuple[str, str, str]:

        class_path = Path(sensor_config["class_dir"])
        if class_path.is_file():
            class_path = class_path.parent

        SensorClass, DriverClass = import_class_from_directory(class_path)
        DriverClass = DriverClass.value

        attached_body_id = None
        if sensor_config["sim_config"].get("attach", None):
            raise NotImplementedError(
                "Attaching sensors to bodies is not implemented for Mujoco yet."
            )

        driver = DriverClass(name, sensor_config, attached_body_id, client=None)
        sensor = SensorClass(
            name=name,
            driver=driver,
            global_config=self.global_config,
        )
        self.sensor_ref[name] = sensor
        xml_config = driver.get_xml_config()
        return xml_config

    def add_sim_component(
        self,
        name: str,
        obj_config: dict[str, Any],
    ) -> None:
        """
        Returns an MJCF <body>...</body> snippet for a single object
        based on a PyBullet-like config.
        """
        print(f"Adding simulation component: {name} with config: {obj_config}")
        
        sim_component = MujocoMultiBody(
            name=name, client=self.builder, global_config=self.global_config
        )
        self.object_ref[name] = sim_component

    def _all_available(self):
        """!Check whether all registered components are active.

        @return ``True`` if no component is suspended.
        @rtype bool
        """
        for robot in self.robot_ref:
            if self.robot_ref[robot]._is_suspended:
                return False
        for obj in self.object_ref:
            if self.object_ref[obj]._is_suspended:
                return False
        return True

    def remove(self, name: str) -> None:
        pass

    def step(self) -> None:
        # print("Stepping the Mujoco simulation.")
        """!Step the simulator forward by one time step."""
        if self._all_available():
            # step all the components
            # self._step_sim_components()

            # update the simulation
            mujoco.mj_step(self.model, self.data)

            # update the viewer
            if self.headless == False:
                self.viewer.sync()

            self._simulation_time = self.data.time

        else:
            log.panda("Did not step")
            pass
        # print("Mujoco simulation step completed.")

    def shutdown_backend(self) -> None:
        if not self.headless:
            # close the viewer
            self.viewer.close()
