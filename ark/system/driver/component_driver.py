
"""! Abstract driver layer for all components.

Drivers handle the backend specific communication for robots, sensors and other
simulation objects.  They provide unified methods so that higher level
components can interact with different simulators or hardware without caring
about their implementation details.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from pathlib import Path
import os
import yaml

from ark.tools.log import log

class ComponentDriver(ABC):
    """
    Abstract base class for a driver that facilitates communication between 
    component classes and a backend (e.g., simulator or hardware). This class 
    should handle backend-specific details.

    Attributes:
        component_name (str): The name of the component using this driver.
        component_config (Dict[str, Any], optional): Configuration settings 
            for the component. Defaults to None.
    """

    def __init__(self, 
                 component_name: str, 
                 component_config: Any = None,
                 sim: bool = True,
                 ) -> None:
        """
        Initializes the ComponentDriver.

        Args:
            component_name (str): The name of the component using this driver.
            component_config (Dict[str, Any], optional): Configuration settings 
                for the component using this driver. Defaults to None.
        """
        self.component_name = component_name
        
        if not isinstance(component_config, dict):
            self.config = self._load_single_section(component_config, component_name)
        else:
            self.config = component_config
        self.sim = sim


    def _load_single_section(self, component_config, component_name):
        """! Load configuration for a single component from a YAML file."""
        # handle path object vs string
        if isinstance(component_config, str):
            component_config = Path(component_config)
        elif not component_config.exists():
            log.error("Given configuration file path does not exist.")
            
        if not component_config.is_absolute():
            component_config = component_config.resolve()
        
        config_path = str(component_config)
        with open(config_path, 'r') as file:
            cfg = yaml.safe_load(file)
        section_config = {}
        for section_name in ["robots", "sensors", "objects"]:
            for item in cfg.get(section_name, []):
                if isinstance(item, dict):  # If it's an inline configuration
                    subconfig = item
                elif isinstance(item, str) and item.endswith('.yaml'):  # If it's a path to an external file
                    if os.path.isabs(item):  # Check if the path is absolute
                        external_path = item
                    else:  # Relative path, use the directory of the main config file
                        external_path = os.path.join(os.path.dirname(config_path), item)
                    # Load the YAML file and return its content
                    with open(external_path, 'r') as file:
                        subconfig = yaml.safe_load(file)
                else:
                    log.error(f"Invalid entry in '{section_name}': {item}. Please provide either a config or a path to another config.")
                    continue  # Skip invalid entries
                
                if subconfig["name"] == component_name:
                    section_config = subconfig["config"]
        if not section_config:
            log.error(f"Could not find configuration for {component_name} in {config_path}")
        return section_config
    
    def is_sim(self):
        """! Return ``True`` if this driver is used in simulation."""
        return self.sim

    @abstractmethod
    def shutdown_driver(self) -> None:
        """
        Abstract method to shut down the driver. This should handle resource 
        cleanup, closing connections, and other shutdown procedures.
        """
        pass
