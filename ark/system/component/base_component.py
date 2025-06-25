
from abc import ABC, abstractmethod
from typing import Any, Dict, Union
from pathlib import Path
import os
import yaml

from arktypes import flag_t
from ark.client.comm_infrastructure.hybrid_node import HybridNode
from ark.system.driver.component_driver import ComponentDriver
from ark.tools.log import log


class BaseComponent(HybridNode, ABC):
    """
    Abstract base class for system components: robots, sensors, and simulation objects.

    This class serves as a template for creating any component in the ark sim-to-real pipeline. 
    Subclasses must implement the 'reset' method to define how to components return to their initial state.
    """


    def __init__(self, 
                 name: str, 
                 global_config: Union[str, Dict[str, Any], Path],
                 ) -> None:
        """
        Initialize the SystemComponent instance.

        Args:
            name (str): The name of the component, initializes name 
        
        Raises:
            ValueError: If the provided 'name' is empty or invalid.
        """
        if not name:
            raise ValueError("Name must be a non-empty string (unique in your system).")
        super().__init__(name, global_config)
        self.name = name # node_name and name are the same
        self._is_suspended = False  # TODO do we still need this ?
        


    @abstractmethod
    def pack_data(self) -> None:
        """
        Packs the data to be sent to the client.

        This method should be implemented by subclasses to define specific behavior 
        for packing data to be sent to the client.
        """

    def component_channels_init(self, channels) -> None:
        """
        Initialize the component's channels.

        This method should be implemented by subclasses to define specific behavior 
        for initializing the component's channels.
        """
        self.component_multi_publisher = self.create_multi_channel_publisher(channels)

    @abstractmethod
    def step_component(self) -> None:
        """
        Handles the component's functionality, e.g., processing sensor data, controlling robot, etc.

        This method should be implemented by subclasses to define specific behavior 
        for the component.
        """
        ...


    @abstractmethod
    def reset_component(self, channel, msg) -> None:
        """
        Resets the state of the component (e.g., robot, sensor, simulation object).

        This method should be implemented by subclasses to define specific behavior 
        for resetting states, configurations, or other parameters that are important 
        for the component.
        """
        ...       
            
        
        
        
class SimToRealComponent(BaseComponent, ABC):
    
    def __init__(self, 
                 name: str, 
                 global_config: Union[str, Dict[str, Any], Path],
                 driver: ComponentDriver = None,
                 ) -> None:
        
        super().__init__(name, global_config)
        self._driver = driver
        self.sim = self._driver.is_sim()
        
        # initialize service for reset of any component
        self.reset_service_name = self.name + "/reset/"
        if self.sim:
            self.reset_service_name = self.reset_service_name + "sim/" 
      
        
    # Override killing the node to also shutwodn the driver, freeing up ports etc.
    def kill_node(self) -> None:
        # kill driver (close ports, ...)
        self._driver.shutdown_driver()
        # kill all communication
        super().kill_node()