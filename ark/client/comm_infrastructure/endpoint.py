
import lcm
from lcm import LCM
#from typing import Any, Optional, Dict, Tuple, List, Union
from pathlib import Path
#import os
import yaml
from ark.tools.log import log
#import socket
import os

class EndPoint:

    def __init__(self, global_config) -> None:
        """
        Initializes an Endpoint object to interact with the registry and set up LCM communication.

        Parameters:
        - registry_host (str): The host address of the registry. Default is "127.0.0.1" (localhost).
        - registry_port (int): The port number for the registry. Default is 1234.
        - lcm_network_bounces (int): The Time To Live (TTL) value for LCM multicast messages. Default is 1.

        This constructor sets up the registry host and port as instance variables,
        and configures the LCM system using a multicast address and the provided TTL.
        """
        
        # self.network_config = {
        #     "registry_host": "127.0.0.1",#"10.206.165.77",
        #     "registry_port": 1234,
        #     "lcm_network_bounces": 1 #was 1
        # }
        self._load_network_config(global_config)
        self.registry_host = self.network_config.get("registry_host", "127.0.0.1")
        self.registry_port = self.network_config.get("registry_port", 1234)
        self.lcm_network_bounces = self.network_config.get("lcm_network_bounces", 1)
        udpm = f"udpm://239.255.76.67:7667?ttl={self.lcm_network_bounces}"
        self._lcm: LCM = lcm.LCM(udpm)


    def _load_network_config(self, global_config: str | Path | dict | None) -> None:
        """
        Loads and updates the network configuration for the current instance from the provided input.

        This method accepts a string representing the path to a YAML file, a `Path` object pointing
        to a YAML file, a dictionary containing network configuration data, or `None`. It then attempts
        to extract and store the 'network' configuration in `self.network_config`. If the file or
        configuration is missing or invalid, default system settings are used and logged appropriately.

        Args:
            global_config (str | Path | dict | None):
                - If a string or `Path`, it should point to a valid YAML configuration file containing a
                'network' key.
                - If a dictionary, it should include a 'network' key with relevant configuration values.
                - If `None`, the method will log a warning and use default settings.

        Returns:
            None or dict:
                - In most cases, the method updates `self.network_config` in place and returns `None`.
                - If the YAML file cannot be read, an empty dictionary `{}` is returned early,
                signaling an error in reading the file.

        Raises:
            None: This function does not explicitly raise any exceptions. Errors are logged instead.

        Side Effects:
            - Modifies the `self.network_config` attribute of the instance.
            - Logs warnings or errors if configuration data is missing or invalid.
        """
        self.network_config = {}
        # extract network part of the global config 
        if isinstance(global_config, str):
            global_config = Path(global_config)  # Convert string to a Path object

            # Check if the given path exists
            if not global_config.exists():
                log.error("Given configuration file path does not exist. Using default system configuration.")
                return  # Exit the function if the file does not exist

            # Resolve relative paths to absolute paths
            elif not global_config.is_absolute():
                global_config = global_config.resolve()

        # If global_config is now a Path object, treat it as a configuration file
        if isinstance(global_config, Path):
            config_path = str(global_config)  # Convert Path to string

            try:
                # Attempt to open and read the YAML configuration file
                with open(config_path, 'r') as file:
                    cfg = yaml.safe_load(file) or {}  # Load YAML content, default to an empty dictionary if None
            except Exception as e:
                log.error(f"Error reading config file {config_path}: {e}. Using default system configuration.")
                return {} # Exit on failure to read file
            
            try:
                # Extract and update the 'system' configuration if it exists in the loaded YAML
                if "network" in cfg:
                    self.network_config.update(cfg.get("network", self.network_config))
                else:
                    log.warn(f"Couldn't find system in config. Using default system configuration.")
                return  # Successfully updated configuration
            except Exception as e:
                log.error(f"Invalid entry in 'system' for. Using default system configuration.")
                return  # Exit if there's an error updating the config

        # If global_config is a dictionary, assume it directly contains configuration values
        elif isinstance(global_config, dict):
            try:
                # check if system exists in the global_config
                if "network" in global_config:
                    self.network_config.update(global_config.get("network"))
                else:
                    log.warn(f"Couldn't find system in config. Using default system configuration.")
            except Exception as e:
                log.warn(f"Couldn't find system in config. Using default system configuration.")

        # If no configuration is provided (None), log a warning and use the default config
        elif global_config is None:
            log.warn(f"No global configuration provided. Using default system configuration.")

        # If global_config is of an unsupported type, log an error and use the default config
        else: 
            log.error(f"Invalid global configuration type: {type(global_config)}. Using default system configuration.")