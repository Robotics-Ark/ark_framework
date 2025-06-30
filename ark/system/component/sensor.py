
"""Base classes for sensor components."""

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
    """Base class for sensors used in the framework."""


    def __init__(self, name: str,
                       global_config: Dict[str, Any] = None,
                       driver: Optional[SensorDriver] = None,
                       ) -> None:
        """Create a sensor component.

        @param name  Name of the sensor.
        @param global_config  Global configuration dictionary.
        @param driver  Optional :class:`SensorDriver` implementation.
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
        """Acquire data from the sensor or its simulation."""
        

    @abstractmethod
    def pack_data(self, data: Any):
        """Transform raw sensor data into the message format."""
        

    # # OVERRIDE
    # def shutdown(self) -> None:
    #     # kill driver (close ports, ...)
    #     self._driver.shutdown_driver()
    #     # kill all communication
    #     super().shutdown()

    def reset_component(self) -> None:
        """Reset the sensor state if necessary."""

    def step_component(self):
        """Retrieve, pack and publish sensor data."""
        data = self.get_sensor_data()
        packed = self.pack_data(data)
        self.component_multi_publisher.publish(packed)