
"""Base classes for ARK system components.

This module defines the :class:`BaseComponent` and
:class:`SimToRealComponent` classes which form the foundation of all
robot, sensor and simulation objects in the framework.
"""

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
    """Base class for all components in the ARK system.

    @brief Provides common functionality for robots, sensors and
    simulated objects.

    Concrete implementations must provide methods for packing data,
    stepping the component and resetting it to a well defined state.
    """


    def __init__(self,
                 name: str,
                 global_config: Union[str, Dict[str, Any], Path],
                 ) -> None:
        """Construct a new component.

        @param name  Unique name of the component.
        @param global_config  Global configuration or path to a YAML
        configuration file.
        @throws ValueError if ``name`` is empty.
        """
        if not name:
            raise ValueError("Name must be a non-empty string (unique in your system).")
        super().__init__(name, global_config)
        self.name = name # node_name and name are the same
        self._is_suspended = False  # TODO do we still need this ?
        


    @abstractmethod
    def pack_data(self) -> None:
        """Pack the component data for publishing.

        Subclasses use this hook to convert raw state information into
        the LCM types that are sent to the client.
        """

    def component_channels_init(self, channels: Dict[str, type]) -> None:
        """Create publishers for the specified channels.

        @param channels  Iterable of channel names that the component
        should publish on.
        """
        self.component_multi_publisher = self.create_multi_channel_publisher(channels)

    @abstractmethod
    def step_component(self) -> None:
        """Perform a single update of the component state.

        Called periodically by the communication layer, this method is
        responsible for gathering data from the driver and publishing it
        to the appropriate channels.
        """
        ...


    @abstractmethod
    def reset_component(self, channel, msg) -> None:
        """Reset the component to a known state.

        @param channel  Channel name for the reset request.
        @param msg  Optional message containing reset parameters.
        """
        ...       
            
        
        
        
class SimToRealComponent(BaseComponent, ABC):
    """Component with a driver that may run in simulation or on real hardware."""

    def __init__(self,
                 name: str,
                 global_config: Union[str, Dict[str, Any], Path],
                 driver: ComponentDriver = None,
                 ) -> None:
        """Create a component connected to a driver.

        @param name  Name of the component.
        @param global_config  Configuration dictionary or path.
        @param driver  Driver instance that provides low level access.
        """

        super().__init__(name, global_config)
        self._driver = driver
        self.sim = self._driver.is_sim()
        
        # initialize service for reset of any component
        self.reset_service_name = self.name + "/reset"
        if self.sim:
            self.reset_service_name = self.reset_service_name + "/sim" 
      
        
    # Override killing the node to also shutdown the driver, freeing up ports etc.
    def kill_node(self) -> None:
        """Shut down the node and associated driver."""
        # kill driver (close ports, ...)
        self._driver.shutdown_driver()
        # kill all communication
        super().kill_node()