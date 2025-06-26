
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional

from ark.tools.log import log
from ark.system.driver.sensor_driver import SensorType
from ark.system.component.robot import Robot
from ark.system.component.sensor import Sensor
from ark.system.component.sim_component import SimComponent

    
class SimulatorBackend(ABC):
    """
    TODO
    Handles global simulator stuff
    """

    def __init__(self, global_config: Dict[str, Any]) -> None:
        """
        Initializes the simulator with the given configuration.

        Args:
            config (dict): The configuration dictionary for the simulator initialization.
        """
        self.robot_ref: Dict[str, Robot] = {}  # Key is robot name, value is config dict
        self.object_ref: Dict[str, SimComponent] = {}  # Key is object name, value is config dict
        self.sensor_ref: Dict[str, Sensor] = {}  # Key is sensor name, value is config dict
        self.ready: bool = False
        self._simulation_time: float = 0.0
        self.global_config = global_config
        self.initialize()
        self.ready = True
        
    def is_ready(self) -> bool:
        """Returns whether the simulator is ready for interaction."""
        return self.ready

    #########################
    ##    Initialization   ##
    #########################

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the simulator with the given configuration.

        Args:
            config (dict): The configuration dictionary.
        """
        ...
    
    @abstractmethod
    def set_gravity(self, gravity: tuple[float, float, float]) -> None:
        """
        Set the gravity in the simulator environment.

        Args:
            gravity (tuple): A tuple (x, y, z) representing the gravity vector.
        """
        ...


    @abstractmethod
    def reset_simulator(self) -> None:
        """Reset the entire simulator state, including scene and objects."""
        ...

    @abstractmethod
    def add_robot(
        self,
        name: str,
        global_config: dict[str, Any],
    ) -> None:
        """
        Add a robot to the simulator.

        Args:
            name (str): The name of the robot.
            config (dict): Configuration parameters for the robot.
        """
        ...

    @abstractmethod
    def add_sensor(
        self,
        name: str,
        sensor_type: SensorType,
        global_config: dict[str, Any],
    ) -> None:
        """
        Add a sensor to the simulator.

        Args:
            name (str): The name of the sensor.
            sensor_type (SensorType): The type of sensor (e.g., CAMERA, FORCE_TORQUE).
            config (dict): Configuration parameters for the sensor.
        """
        ...
    
    @abstractmethod
    def add_sim_component(
        self,
        name: str,
        type: str,
        global_config: dict[str, Any],
    ) -> None:
        """
        Add an object to the simulator.

        Args:
            name (str): The name of the object.
            type (str): The type of object (e.g., "cube", "sphere").
            config (dict): Configuration parameters for the object.
        """
        ...


    @abstractmethod
    def remove(self, name: str) -> None:
        """
        Remove a robot, sensor, or object from the simulator.

        Args:
            name (str): The name of the object to remove.
        """
        ...

    @abstractmethod
    def step(self) -> None:
        """Advance the simulator by one time step."""
        ...
        
    @abstractmethod
    def shutdown_backend(self) -> None:
        """Shutdown the simulator, cleaning up resources."""
        pass

        
    def _step_sim_components(self) -> None:
        """
     
        """
        for robot in self.robot_ref:
            if not self.robot_ref[robot]._is_suspended:
                self.robot_ref[robot].step_component()
                self.robot_ref[robot].control_robot()
        for obj in self.object_ref:
            self.object_ref[obj].step_component()
        for sensor in self.sensor_ref:
            self.sensor_ref[sensor].step_component()


    def _spin_sim_components(self) -> None:
        """
        TODO

        """
        for robot in self.robot_ref:
            self.robot_ref[robot].manual_spin()
        for obj in self.object_ref:
            self.object_ref[obj].manual_spin()
        for sensor in self.sensor_ref:
            self.sensor_ref[sensor].manual_spin()

            


