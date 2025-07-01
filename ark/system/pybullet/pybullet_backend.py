"""@file pybullet_backend.py
@brief Backend implementation for running simulations in PyBullet.
"""

import importlib.util
import sys, ast, os
import math
import cv2
from pathlib import Path
from typing import Any, Optional, Dict

import pybullet as p
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

from ark.tools.log import log
from ark.system.simulation.simulator_backend import SimulatorBackend
from ark.system.pybullet.pybullet_robot_driver import BulletRobotDriver
from ark.system.pybullet.pybullet_camera_driver import BulletCameraDriver
from ark.system.pybullet.pybullet_multibody import PyBulletMultiBody
from arktypes import *



def import_class_from_directory(path: Path) -> tuple[type, Optional[type]]:
    """!Load a class from ``path``.

    The helper searches for ``<ClassName>.py`` inside ``path`` and imports the
    class with the same name.  If a ``Drivers`` class is present in the module
    its ``PYBULLET_DRIVER`` attribute is returned alongside the main class.

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
    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    # check if Sensor_Drivers is in the class_names
    if "Drivers" in class_names:
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location(class_names[0], file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[class_names[0]] = module
        spec.loader.exec_module(module)

        class_ = getattr(module, class_names[0])
        sys.path.pop(0)

        drivers = class_.PYBULLET_DRIVER
        class_names.remove("Drivers")

    # Retrieve the class from the module (has to be list of one)
    class_ = getattr(module, class_names[0])


    if len(class_names) != 1:
        raise ValueError(f"Expected exactly two class definition in {file_path}, but found {len(class_names)}.")


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




class PyBulletBackend(SimulatorBackend):
    """Backend wrapper around the PyBullet client.

    This class handles scene creation, stepping the simulation and managing
    simulated components such as robots, objects and sensors.
    """

    def initialize(self) -> None:
        """!Initialize the PyBullet world.

        The method creates the Bullet client, configures gravity and time step
        and loads all robots, objects and sensors defined in
        ``self.global_config``.  Optional frame capture settings are applied as
        well.
        """
        self.ready = False
        self.client = self._connect_pybullet(self.global_config)
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Render images from Pybullet and save
        self.save_render_config = self.global_config["simulator"].get("save_render", None)
        if self.save_render_config is not None:
            self._rendered_time = -1.0
            self.save_path = Path(self.save_render_config.get("save_path", "output/save_render"))
            self.save_path.mkdir(parents=True, exist_ok=True)

            # Remove existing files
            remove_existing = self.save_render_config.get("remove_existing", True)
            if remove_existing:
                for child in self.save_path.iterdir():
                    if child.is_file():
                        child.unlink()

            # Get config
            default_extrinsics = {
                "look_at": [0, 0, 1.0],
                "distance": 3,
                "azimuth": 0,
                "elevation": 0
            }
            default_intrinsics = {
                "width": 640,
                "height": 480,
                "field_of_view": 60,
                "near_plane": 0.1,
                "far_plane": 100.0
            }
            self.save_interval = self.save_render_config.get("save_interval", 1 / 30)
            self.overwrite_file = self.save_render_config.get("overwrite_file", False)
            self.extrinsics = self.save_render_config.get("extrinsics", default_extrinsics)
            self.intrinsics = self.save_render_config.get("intrinsics", default_intrinsics)

        for additional_urdf_dir in self.global_config["simulator"]['config'].get("urdf_dirs", []):
            self.client.setAdditionalSearchPath(additional_urdf_dir)

        gravity = self.global_config["simulator"]['config'].get("gravity", [0, 0, -9.81])
        self.set_gravity(gravity)

        timestep = 1 / self.global_config["simulator"]['config'].get("sim_frequency", 240.0)
        self.set_time_step(timestep)

        # Setup robots
        if self.global_config.get("robots", None):
            for robot_name, robot_config in self.global_config["robots"].items():
                self.add_robot(robot_name,
                            robot_config)

        # Setup objects
        if self.global_config.get("objects", None):
            for obj_name, obj_config in self.global_config["objects"].items():
                self.add_sim_component(obj_name,
                                    obj_config)

        # Sensors have to be set up last, as e.g. cameras might need
        # a parent to attach to
        if self.global_config.get("sensors", None):
            for sensor_name, sensor_config in self.global_config["sensors"].items():
                self.add_sensor(sensor_name,
                                sensor_config)
        self.ready = True


    def is_ready(self) -> bool:
        """!Check whether the backend has finished initialization.

        @return ``True`` once all components were created and the simulator is
                ready for stepping.
        @rtype bool
        """
        return self.ready


    def _connect_pybullet(self, config: dict[str, Any]):
        """!Create and return the Bullet client.

        ``config`` must contain the ``connection_mode`` under the ``simulator``
        section.  Optionally ``mp4`` can be provided to enable video
        recording.

        @param config Global configuration dictionary.
        @return Initialized :class:`BulletClient` instance.
        @rtype BulletClient
        """
        kwargs = dict(options="")
        mp4 = config.get("mp4")
        if mp4:
            kwargs["options"] = f"--mp4={mp4}"
        connection_mode_str = config["simulator"]["config"]["connection_mode"].upper()
        connection_mode = getattr(p, connection_mode_str)
        return BulletClient(connection_mode, **kwargs)


    def set_gravity(self, gravity: tuple[float]) -> None:
        """!Set the world gravity.

        @param gravity Tuple ``(gx, gy, gz)`` specifying gravity in m/s^2.
        """
        self.client.setGravity(gravity[0], gravity[1], gravity[2])


    def set_time_step(self, time_step: float) -> None:
        """!Set the simulation timestep.

        @param time_step Length of a single simulation step in seconds.
        """
        self.client.setTimeStep(time_step)
        self._time_step = time_step


    ##########################################################
    ####            ROBOTS, SENSORS AND OBJECTS           ####
    ##########################################################

    def add_robot(self,
                  name: str,
                  robot_config: Dict[str, Any]):
        """!Instantiate and register a robot in the simulation.

        @param name Identifier for the robot.
        @param robot_config Robot specific configuration dictionary.
        """
        class_path = Path(robot_config["class_dir"])
        if class_path.is_file():
            class_path = class_path.parent
        RobotClass, DriverClass = import_class_from_directory(class_path)
        DriverClass = DriverClass.value
        driver = DriverClass(name,
                             robot_config,
                             self.client)
        robot = RobotClass(name = name,
                           global_config = self.global_config,
                           driver = driver)

        self.robot_ref[name] = robot

    def add_sim_component(self,
                          name: str,
                          obj_config: Dict[str, Any],
                          ) -> None:
        """!Add a generic simulated object.

        @param name Name of the object.
        @param obj_config Object specific configuration dictionary.
        """
        sim_component = PyBulletMultiBody(name=name,
                                          client=self.client,
                                          global_config=self.global_config)
        self.object_ref[name] = sim_component

    def add_sensor(self,
                   name: str,
                   sensor_config: Dict[str, Any]
                   ) -> None:
        """!Instantiate and register a sensor.

        @param name Name of the sensor component.
        @param sensor_config Sensor configuration dictionary.
        """
        sensor_type = sensor_config["type"]
        class_path = Path(sensor_config["class_dir"])
        if class_path.is_file():
            class_path = class_path.parent

        SensorClass, DriverClass = import_class_from_directory(class_path)
        DriverClass = DriverClass.value

        attached_body_id = None
        if sensor_config["sim_config"].get("attach", None):

            print(self.global_config["objects"].keys())
            # search through robots and objects to find attach link if needed
            if sensor_config["sim_config"]["attach"]["parent_name"] in self.global_config["robots"].keys():
                attached_body_id = self.robot_ref[sensor_config["sim_config"]["attach"]["parent_name"]]._driver.ref_body_id
            elif sensor_config["sim_config"]["attach"]["parent_name"] in self.global_config["objects"].keys():
                attached_body_id = self.object_ref[sensor_config["sim_config"]["attach"]["parent_name"]].ref_body_id
            else:
                log.error(f"Parent to attach sensor " + name + " to does not exist !")
        driver = DriverClass(name,
                             sensor_config,
                             attached_body_id,
                             self.client)
        sensor = SensorClass(name = name,
                             driver = driver,
                             global_config = self.global_config,
                             )

        self.sensor_ref[name] = sensor

    def remove(self, name: str) -> None:
        """!Remove a component from the simulator.

        @param name Name of the robot, object or sensor to remove.
        """
        if name in self.robot_ref:
            self.robot_ref[name].shutdown()
            del self.robot_ref[name]
        elif name in self.sensor_ref:
            self.sensor_ref[name].shutdown()
            del self.obsensor_refject_ref[name]
        elif name in self.object_ref:
            self.object_ref[name].shutdown()
            del self.object_ref[name]
        else:
            log.warning("Could not remove " + name + ", it does not exist.")
            return
        log.ok("Deleted " + name + " !")


    #######################################
    ####          SIMULATION           ####
    #######################################

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

    def step(self) -> None:
        """!Advance the simulation by one timestep.

        The method updates all registered components, advances the physics
        engine and optionally saves renders when enabled.
        """
        if self._all_available():
            self._step_sim_components()
            self.client.stepSimulation()
            self._simulation_time += self._time_step

            if self.save_render_config is not None:
                if (self._simulation_time - self._rendered_time) > self.save_interval:
                    self.save_render()
                    self._rendered_time = self._simulation_time

        else:
            log.panda("Did not step")
            pass

    def save_render(self):
        """!Render the scene and write the image to disk.

        The image is saved either as ``render.png`` when overwriting or with the
        current simulation time as filename when not.
        """
        # Calculate camera extrinsic matrix
        look_at = self.extrinsics["look_at"]
        azimuth = math.radians(self.extrinsics["azimuth"])
        distance = self.extrinsics["distance"]

        x = look_at[0] + distance * math.cos(azimuth)
        y = look_at[1] + distance * math.sin(azimuth)
        z = look_at[2] + self.extrinsics["elevation"]

        view_matrix =p.computeViewMatrix(
            cameraEyePosition=[x, y, z],
            cameraTargetPosition=look_at,
            cameraUpVector=[0, 0, 1]
        )

        # Calculate intrinsic matrix
        width = self.intrinsics["width"]
        height = self.intrinsics["height"]
        aspect = width / height
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.intrinsics["field_of_view"],
            aspect=aspect,
            nearVal=self.intrinsics["near_plane"],
            farVal=self.intrinsics["far_plane"]
        )

        # Render the image
        image = p.getCameraImage(width, height, viewMatrix=view_matrix, projectionMatrix=projection_matrix)

        # image[2] contains the color image (RGBA) as a numpy array
        rgba = image[2]

        # Save image
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGB2BGR)
        time_us = int(1e6 * self._simulation_time)

        if self.overwrite_file:
            save_path = self.save_path / "render.png"
        else:
            save_path = self.save_path / f"{time_us}.png"
        cv2.imwrite(str(save_path), bgra)

    def reset_simulator(self) -> None:
        """!Reset the entire simulator state.

        All robots, objects and sensors are destroyed and the backend is
        re-initialized using ``self.global_config``.
        """
        log.error("Reset Simulator function is not ready yet !")
        for robot in self.robot_ref:
            self.robot_ref[robot].kill_node()

        for obj in self.object_ref:
            self.object_ref[obj].kill_node()

        for sensor in self.sensor_ref:
            self.sensor_ref[sensor].kill_node()

        self.client.disconnect()
        self._simulation_time = 0.0
        self.initialize()

        if self.save_render_config is not None:
            self._rendered_time = -1.0

        log.ok("Simulator reset complete.")

    def get_current_time(self) -> float:
        """!Return the current simulation time.

        @return Elapsed simulation time in seconds.
        @rtype float
        """
        # https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12438
        return self._simulation_time

    def shutdown_backend(self):
        """!Disconnect all components and shut down the backend.

        This should be called at program termination to cleanly close the
        simulator and free all resources.
        """
        self.client.disconnect()
        for robot in self.robot_ref:
            self.robot_ref[robot].kill_node()
        for obj in self.object_ref:
            self.object_ref[obj].kill_node()
        for sensor in self.sensor_ref:
            self.sensor_ref[sensor].kill_node()
