
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Dict, List

import pybullet as p

from ark.tools.log import log
from ark.system.driver.component_driver import ComponentDriver




class ControlType(Enum):
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
        
        super().__init__(component_name=component_name,
                         component_config=component_config,
                         sim=sim)

    #####################
    ##    get infos    ##
    #####################

    @abstractmethod
    def check_torque_status(self) -> bool:
        pass

    @abstractmethod
    def pass_joint_positions(self, joints: List[str]) -> Dict[str, float]:
        pass

    @abstractmethod
    def pass_joint_velocities(self, joints: List[str]) -> Dict[str, float]:
        pass

    @abstractmethod
    def pass_joint_efforts(self, joints: List[str]) -> Dict[str, float]:
        pass

    #####################
    ##     control     ##
    #####################

    @abstractmethod
    def pass_joint_group_control_cmd(self, control_mode: str, cmd: Dict[str, float], **kwargs) -> None:
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
        super().__init__(component_name, component_config, True)

    @abstractmethod
    def sim_reset(self, 
                  base_pos : List[float],
                  base_orn : List[float],
                  init_pos : List[float]) -> None:
        ...

    def shutdown_driver(self) -> None:
        # Nothing to handle here
        pass
