"""Abstract interface for simulation backends."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional

from ark.tools.log import log
from ark.system.driver.sensor_driver import SensorType
from ark.system.component.robot import Robot
from ark.system.component.sensor import Sensor
from ark.system.component.sim_component import SimComponent

    
class SimulatorBackend(ABC):
    """Base class for all simulator backends.

    The backend manages robots, sensors and other simulated components and
    exposes a minimal interface for stepping and resetting the simulation
    environment.
    """

    def __init__(self, global_config: Dict[str, Any]) -> None:
        """!Create and initialize the backend.

        @param global_config Dictionary describing the complete simulator
               configuration.
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
        """!Check if the backend finished initialization."""
        return self.ready

    #########################
    ##    Initialization   ##
    #########################

    @abstractmethod
    def initialize(self) -> None:
        """!Initialize the simulator implementation."""
        ...
    
    @abstractmethod
    def set_gravity(self, gravity: tuple[float, float, float]) -> None:
        """!Set the gravity vector used by the simulator.

        @param gravity Tuple ``(x, y, z)`` representing the gravity vector.
        """
        ...


    @abstractmethod
    def reset_simulator(self) -> None:
        """!Reset the entire simulator state."""
        ...

    @abstractmethod
    def add_robot(
        self,
        name: str,
        global_config: dict[str, Any],
    ) -> None:
        """!Add a robot to the simulation.

        @param name Name of the robot.
        @param global_config Configuration dictionary for the robot.
        """
        ...

    @abstractmethod
    def add_sensor(
        self,
        name: str,
        sensor_type: SensorType,
        global_config: dict[str, Any],
    ) -> None:
        """!Add a sensor to the simulation.

        @param name Name of the sensor.
        @param sensor_type Type of the sensor.
        @param global_config Configuration dictionary for the sensor.
        """
        ...
    
    @abstractmethod
    def add_sim_component(
        self,
        name: str,
        type: str,
        global_config: dict[str, Any],
    ) -> None:
        """!Add a generic simulation object.

        @param name Name of the object.
        @param type Type identifier (e.g. ``"cube"``).
        @param global_config Configuration dictionary for the object.
        """
        ...


    @abstractmethod
    def remove(self, name: str) -> None:
        """!Remove a robot, sensor or object by name.

        @param name Name of the component to remove.
        """
        ...

    @abstractmethod
    def step(self) -> None:
        """!Advance the simulator by one timestep."""
        ...
        
    @abstractmethod
    def shutdown_backend(self) -> None:
        """!Shut down the simulator and free resources."""
        pass

        
    def _step_sim_components(self) -> None:
        """!Step all registered components."""
        for robot in self.robot_ref:
            if not self.robot_ref[robot]._is_suspended:
                self.robot_ref[robot].step_component()
                self.robot_ref[robot].control_robot()
        for obj in self.object_ref:
            self.object_ref[obj].step_component()
        for sensor in self.sensor_ref:
            self.sensor_ref[sensor].step_component()


    def _spin_sim_components(self) -> None:
        """!Spin components in manual mode."""
        for robot in self.robot_ref:
            self.robot_ref[robot].manual_spin()
        for obj in self.object_ref:
            self.object_ref[obj].manual_spin()
        for sensor in self.sensor_ref:
            self.sensor_ref[sensor].manual_spin()

            


