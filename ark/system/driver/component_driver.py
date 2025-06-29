
"""! Base component driver definitions.

This module contains the :class:`ComponentDriver` abstract base class used by
all ARK drivers. It includes helper functionality for loading configuration
files and common attributes shared by concrete drivers.
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
        """! Initialize the driver.

        @param component_name Name of the component using this driver.
        @param component_config Path or dictionary with configuration for the
               component using this driver.
        @param sim Set to ``True`` if running in simulation mode.
        """
        self.component_name = component_name
        
        if not isinstance(component_config, dict):
            self.config = self._load_single_section(component_config, component_name)
        else:
            self.config = component_config
        self.sim = sim


    def _load_single_section(self, component_config, component_name):
        """! Load the configuration of a single component from a YAML file.

        This helper parses a global configuration file and extracts the
        subsection corresponding to ``component_name``.

        @param component_config Path to a YAML file or a ``Path`` object
               pointing to the configuration file.
        @param component_name Name of the component whose configuration should
               be loaded.
        @return Dictionary containing the configuration for the component.
        """

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
        """! Return whether this driver is running in simulation mode.

        @return ``True`` if the driver targets a simulator, ``False`` otherwise.
        """

        return self.sim

    @abstractmethod
    def shutdown_driver(self) -> None:
        """! Shut down the driver and release all resources.

        Concrete drivers should override this method to close connections or
        stop any background tasks started by the driver.
        """
        pass
