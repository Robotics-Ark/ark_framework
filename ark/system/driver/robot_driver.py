
"""! Driver interfaces for controlling robots.

This module defines :class:`RobotDriver` and :class:`SimRobotDriver` which act
as adapters between :class:`Robot` components and a specific simulator or real
hardware backend.  Methods provide access to joint states and control commands.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Dict, List

import pybullet as p

from ark.tools.log import log
from ark.system.driver.component_driver import ComponentDriver




class ControlType(Enum):
    """! Enumeration of supported joint control modes."""

    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    FIXED = "fixed"


class RobotDriver(ComponentDriver):
    """
    TODO
    Defines base gateway, responsible for exchanging information between our component classes and a backend
    Should absorb everything that is specific to any simulator or real system 
    (driver will handle differences between real systems)
    """

    def __init__(self,
                 component_name: str,
                 component_config: Dict[str, Any] = None,
                 sim: bool = True,
                 ) -> None:
        """! Initialise the driver with optional configuration and sim flag."""

        super().__init__(component_name=component_name,
                         component_config=component_config,
                         sim=sim)

    #####################
    ##    get infos    ##
    #####################

    @abstractmethod
    def check_torque_status(self) -> bool:
        """! Return ``True`` if the robot's actuators are currently torqued."""
        pass

    @abstractmethod
    def pass_joint_positions(self, joints: List[str]) -> Dict[str, float]:
        """! Return positions for the provided joint names."""
        pass

    @abstractmethod
    def pass_joint_velocities(self, joints: List[str]) -> Dict[str, float]:
        """! Return velocities for the provided joint names."""
        pass

    @abstractmethod
    def pass_joint_efforts(self, joints: List[str]) -> Dict[str, float]:
        """! Return efforts for the provided joint names."""
        pass

    #####################
    ##     control     ##
    #####################

    @abstractmethod
    def pass_joint_group_control_cmd(self, control_mode: str, cmd: Dict[str, float], **kwargs) -> None:
        """! Send a control command for a group of joints."""
        pass



class SimRobotDriver(RobotDriver, ABC):
    """
    TODO
    
    """

    def __init__(self,
                 component_name: str,
                 component_config: Dict[str, Any] = None,
                 sim: bool = True,
                 ) -> None:
        """! Initialise a simulation-only robot driver."""
        super().__init__(component_name, component_config, True)

    @abstractmethod
    def sim_reset(self,
                  base_pos : List[float],
                  base_orn : List[float],
                  init_pos : List[float]) -> None:
        """! Reset the simulated robot to the given base pose and joint state."""
        ...

    def shutdown_driver(self) -> None:
        """! Shutdown hooks for simulators that require cleanup."""
        pass
