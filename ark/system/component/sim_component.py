
"""! Common functionality for simulated objects.

`SimComponent` extends :class:`BaseComponent` with utilities for publishing
ground truth state in simulation.  Subclasses implement data packing and object
specific state retrieval.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from ark.tools.log import log
from ark.system.component.base_component import BaseComponent
from arktypes import flag_t, rigid_body_state_t
    
    
class SimComponent(BaseComponent, ABC):
    """
    TODO
    """


    def __init__(self, 
                 name: str,  
                 global_config: Dict[str, Any] = None
                 ) -> None:
        """
        TODO
        """
        super().__init__(name = name, 
                         global_config = global_config)
        # extract this components configuration from the global configuration
        self.config = self._load_config_section(global_config=global_config, name=name, type="objects")
        # whether this should publish state information 
        self.publish_ground_truth = self.config["publish_ground_truth"] 
        # initialize service for reset of any component
        self.reset_service_name = self.name + "/reset/sim/" 
        
        self.create_service(self.reset_service_name, rigid_body_state_t, flag_t, self.reset_component)

  
    def step_component(self):
        """! Publish ground truth state if configured to do so."""
        if self.publish_ground_truth:
            data = self.get_object_data()
            packed = self.pack_data(data)
            self.component_multi_publisher.publish(packed)
        
    @abstractmethod
    def pack_data(self) -> None:
        """
        Packs the data to be sent to the client.

        This method should be implemented by subclasses to define specific behavior 
        for packing data to be sent to the client.
        """

    @abstractmethod
    def get_object_data(self) -> Any:
        """
        Get the data from the object.

        This method should be implemented by subclasses to define specific behavior 
        for getting data from the object.
        """

    
    
