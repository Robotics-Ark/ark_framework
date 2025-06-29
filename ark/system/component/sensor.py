
"""! Base class for sensor components.

Sensors extend :class:`SimToRealComponent` and provide hooks for reading data
from either simulation or real hardware.  Concrete sensor classes implement
`get_sensor_data` and `pack_data` to publish measurements to the rest of the
system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ark.system.component.base_component import SimToRealComponent
from ark.system.driver.sensor_driver import SensorDriver
from ark.tools.log import log
from typing import Any, Optional, Dict, Tuple, List, Union
from pathlib import Path
import os
import yaml

from arktypes import flag_t

class Sensor(SimToRealComponent, ABC):
    """
    Abstract base class for sensor system components.

    This class serves as a template for creating sensor in the ark sim-to-real pipeline. 
    Subclasses must implement ...
    """


    def __init__(self, name: str, 
                       global_config: Dict[str, Any] = None,
                       driver: Optional[SensorDriver] = None,
                       ) -> None:
        
        """
        Initialize the SystemComponent instance.

        Args:
            name (str): The name of the component, initializes name 
        
        Raises:
            ValueError: If the provided 'name' is empty or invalid.
        """
        
        super().__init__(name, global_config, driver) # handles self.name, self.sim
        self.sensor_config = self._load_config_section(global_config=global_config, name=name, type="sensors")

        # if runing a real system
        if not self.sim:
            try:
                self.freq = self.sensor_config["frequency"]
            except:
                log.warning(f"No frequency provided for sensor '{self.name}', using default !")
                self.freq = 240
            self.create_stepper(self.freq, self.step_component)
            
        self.create_service(self.reset_service_name, flag_t, flag_t, self.reset_component)

    @abstractmethod
    def get_sensor_data(self) -> Any:
        """! Simulate the sensor's behavior."""
        

    @abstractmethod
    def pack_data(self, data: Any):
        """! Pack the sensor data into a lcm_type to be published."""
        

    # # OVERRIDE
    # def shutdown(self) -> None:
    #     # kill driver (close ports, ...)
    #     self._driver.shutdown_driver()
    #     # kill all communication
    #     super().shutdown()

    def reset_component(self) -> None:
        """! Reset the sensor state if required by the implementation."""
        pass

    def step_component(self):
        """! Acquire sensor data and publish it over the network."""
        data = self.get_sensor_data()
        packed = self.pack_data(data)
        self.component_multi_publisher.publish(packed)