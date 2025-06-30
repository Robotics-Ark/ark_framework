
"""! Sensor driver base definitions.

This module contains abstract base classes for sensor drivers used throughout
the ARK framework. Drivers handle backend-specific details for sensors such as
cameras or LiDARs.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Dict, List

from ark.tools.log import log
from ark.system.driver.component_driver import ComponentDriver

import numpy as np


class SensorType(Enum):
    """! Enumeration of supported sensor types."""

    CAMERA = "camera"
    FORCE_TORQUE = "force_torque"
    
    
    
class SensorDriver(ComponentDriver, ABC):
    """! Abstract driver interface for sensors.

    Concrete sensor drivers inherit from this class and implement the required
    methods to acquire data from a simulator or hardware backend.
    """

    def __init__(self,
                 component_name: str,
                 component_config: Dict[str, Any] = None,
                 sim: bool = True,
                 ) -> None:
        """! Initialize the sensor driver.

        @param component_name Name of the sensor component.
        @param component_config Configuration dictionary or path.
        @param sim True if running in simulation mode.
        """

        super().__init__(component_name, component_config, sim)



class CameraDriver(SensorDriver, ABC):
    """! Base class for camera sensor drivers."""

    def __init__(self,
                 component_name: str,
                 component_config: Dict[str, Any] = None,
                 sim: bool = True,
                 ) -> None:
        """! Initialize the camera driver."""

        super().__init__(component_name, component_config, sim)
        
    @abstractmethod
    def get_images(self) -> Dict[str, np.ndarray]:
        """! Retrieve images from the camera."""

        ...

class LiDARDriver(SensorDriver, ABC):
    """!
    Abstract base class for LiDAR sensor drivers.

    Defines the required interface for retrieving LiDAR scan data.
    """

    def __init__(self, 
                 component_name: str,
                 component_config: Dict[str, Any] = None,
                 sim: bool = True,
                 ) -> None:
        """!
        Initialize the LiDAR driver.

        @param component_name Name of the LiDAR component.
        @param component_config Configuration dictionary.
        @param sim True if running in simulation mode.
        """
        super().__init__(component_name, component_config, sim)
        
    @abstractmethod
    def get_scan(self) -> Dict[str, np.ndarray]:
        """!
        Retrieve a LiDAR scan.

        @return Dictionary containing:
            - "angles": 1D NumPy array of angles (in radians) in the LiDAR's reference frame.
            - "ranges": 1D NumPy array of range values (in meters).

        Angles and ranges must be aligned such that each angle corresponds to the respective range index.
        """
        ...
