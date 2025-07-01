
"""Base classes for simulation objects."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from ark.tools.log import log
from ark.system.component.base_component import BaseComponent
from arktypes import flag_t, rigid_body_state_t
    
    
class SimComponent(BaseComponent, ABC):
    """Base class for simulated rigid bodies."""


    def __init__(self,
                 name: str,
                 global_config: Dict[str, Any] = None
                 ) -> None:
        """Create a simulation component.

        @param name  Name of the object.
        @param global_config  Global configuration dictionary.
        """
        super().__init__(name = name, 
                         global_config = global_config)
        # extract this components configuration from the global configuration
        self.config = self._load_config_section(global_config=global_config, name=name, type="objects")
        # whether this should publish state information 
        self.publish_ground_truth = self.config["publish_ground_truth"] 
        # initialize service for reset of any component
        self.reset_service_name = self.name + "/reset/sim" 
        
        self.create_service(self.reset_service_name, rigid_body_state_t, flag_t, self.reset_component)

  
    def step_component(self):
        """Gather object state and publish it if required."""
        if self.publish_ground_truth:
            data = self.get_object_data()
            packed = self.pack_data(data)
            self.component_multi_publisher.publish(packed)
        
    @abstractmethod
    def pack_data(self) -> None:
        """Pack object data into the message format."""

    @abstractmethod
    def get_object_data(self) -> Any:
        """Retrieve the current state of the simulated object."""

    
    
