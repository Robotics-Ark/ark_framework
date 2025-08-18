"""@file mujoco_backend.py
@brief Backend implementation for running simulations in Mujoco.
"""

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
        self.world_model_dict = {}
        # set gravity if specified in the global config

        gravity = self.global_config["simulator"]["config"].get(
            "gravity", [0, 0, -9.81]
        )
        self.world_model_dict["gravity"] = self.set_gravity(gravity)

        self.world_model_dict["objects"] = []
        self.world_model_dict["assets"] = []
        self.world_model_dict["defaults"] = []
        if self.global_config.get("objects", None):
            for obj_name, obj_config in self.global_config["objects"].items():
                asset_xml , body_xml, default_xml = self.add_sim_component(obj_name, obj_config)
                self.world_model_dict["objects"].append(
                    body_xml
                )
                if asset_xml:
                    self.world_model_dict["assets"].append(asset_xml)
                if default_xml:
                    self.world_model_dict["defaults"].append(default_xml)


        # setup model and data by initialsing the xml file
        world_xml = self.build_world(self.world_model_dict)
        print(f"World XML: {world_xml}")
        self.model, self.data = self.compile_model(world_xml)


        # SET UP THE PHYSICS SIMULATOR WITH GUI/DIRECT
        self.headless = True
        if self.global_config.get("connection_mode", "GUI") == "GUI":
            self.headless = False
            if not glfw.init():
                raise RuntimeError("Could not initialize GLFW")
            self.window = glfw.create_window(1200, 900, "Ark Mujoco Viewer", None, None)
            if not self.window:
                glfw.terminate()
                raise RuntimeError("Could not create GLFW window")
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)

            self.cam = mujoco.MjvCamera()
            self.opt = mujoco.MjvOption()
            self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
            self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
            mujoco.mjv_defaultFreeCamera(self.model, self.cam)
            self.cam.distance = max(2.0, np.linalg.norm(self.model.stat.extent) * 1.5)
        else:
            # Launch the viewer in passive mode (headless)
            raise NotImplementedError(
                "Headless mode is not implemented for Mujoco yet."
            )
        
        for obj_key in self.object_ref.keys():
            obj = self.object_ref[obj_key]
            obj.update_ids(self.model, self.data)

        for sensor_key in self.sensor_ref.keys():
            sensor = self.sensor_ref[sensor_key]
            sensor._driver.update_ids(self.model, self.data)


        self.timestep = 1 / self.global_config["simulator"]["config"].get(
            "sim_frequency", 240.0
        )

    def compile_model(self, xml: str):
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        return model, data

    def build_world(self, world_xml: dict) -> str:
        bodies = "\n".join(world_xml["objects"])
        gravity = world_xml["gravity"]
        assets = "\n".join(world_xml["assets"])
        defaults = "\n".join(world_xml["defaults"])

        return f"""
            <mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>           
            """

    def set_gravity(self, gravity: tuple[float, float, float]) -> str:
        return f"""
        <option gravity="{gravity[0]} {gravity[1]} {gravity[2]}"/>
        """

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
    ) -> tuple[str,str,str]:
        """
        Returns an MJCF <body>...</body> snippet for a single object
        based on a PyBullet-like config.
        """
        sim_component = MujocoMultiBody(
            name=name,
            client=None,
            global_config=self.global_config,
        )
        self.object_ref[name] = sim_component
        xml_config = sim_component.get_xml_config()
        return xml_config


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
        """!Step the simulator forward by one time step."""
        if self._all_available():
            # step all the components
            self._step_sim_components()

            # update the simulation 
            mujoco.mj_step(self.model, self.data)
            
            # update the viewer
            if self.headless == False: 
                mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
                fb_w, fb_h = glfw.get_framebuffer_size(self.window)
                viewport = mujoco.MjrRect(0, 0, fb_w, fb_h)
                mujoco.mjr_render(viewport, self.scene, self.ctx)
            

            self._simulation_time = self.data.time
            print(f"Simulation time: {self._simulation_time:.2f}s")

        else:
            log.panda("Did not step")
            pass

    def shutdown_backend(self) -> None:
        self.viewer.close()
