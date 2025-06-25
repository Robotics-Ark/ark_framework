
from pathlib import Path
from abc import ABC, abstractmethod 
from typing import Dict, Any
import os
import yaml
import sys
from ark.client.comm_infrastructure.base_node import BaseNode
from ark.system.pybullet.pybullet_backend import PyBulletBackend
from ark.tools.log import log
from arktypes import flag_t

import pdb


class SimulatorNode(BaseNode, ABC):
    def __init__(self, global_config):
        self._load_config(global_config)
        self.name = self.global_config["simulator"].get("name", "simulator")

        super().__init__(self.name, global_config=global_config)
        
        log.info("Initializing SimulatorNode called " + self.name + " with id " + self.node_id + " ...")

        # Setup backend 
        self.backend_type = self.global_config["simulator"]["backend_type"]
        if self.backend_type == "pybullet":
            self.backend = PyBulletBackend(self.global_config)
        elif self.backend_type == "mujoco":
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported backend '{self.backend_type}'")
        
        # to initialize a scene with objects that dont need to publish, e.g. for visuals 
        self.initialize_scene()
        
        ## Reset Backend Service
        reset_service_name = self.name + "/backend/reset/sim"
        self.create_service(reset_service_name, flag_t, flag_t, self._reset_backend)
        
        freq = self.global_config["simulator"]["config"].get("node_frequency", 240.0)
        self.create_stepper(freq, self._step_simulation) 
        
        
    def _load_config(self, global_config) -> None:
        
        if not global_config:
            raise ValueError("Please provide a global configuration file.")
        
        if isinstance(global_config, str):
            global_config = Path(global_config)
        
        if not global_config.exists():
            raise ValueError("Given configuration file path does not exist, currently: " + str(global_config))
            
        if not global_config.is_absolute():
            global_config = global_config.resolve()
        
        config_path = str(global_config)
        with open(config_path, 'r') as file:
            cfg = yaml.safe_load(file)   

        # assert that the config is a dict
        if not isinstance(cfg, dict):
            raise ValueError("The configuration file must be a valid dictionary.")

        # merge with subconfigs
        config = {}
        try:
            config["network"] = cfg["network"]
        except KeyError as e:
            config["network"] = None
        try:
            config["simulator"] = cfg["simulator"]
        except KeyError as e:
            raise ValueError("Please provide at least name and backend_type under simulation in your config file.")

        try:
            config["robots"] = self._load_section(cfg, config_path, "robots")
        except KeyError as e:
            config["robots"] = {}
        try:
            config["sensors"] = self._load_section(cfg, config_path, "sensors")
        except KeyError as e:
            config["sensors"] = {}
        try:
            config["objects"] = self._load_section(cfg, config_path, "objects")
        except KeyError as e:
            config["objects"] = {}
    
        log.ok("Config file under " + config_path + " loaded successfully.")
        self.global_config = config


    def _load_section(self, cfg: dict[str, Any], config_path: str, section_name: str) -> dict[str, Any]:
        """
        Generic function to load a section from the config (e.g., robots, sensors, or objects).
        It handles both inline configurations and paths to external YAML files.

        @param cfg: The main configuration dictionary.
        @param config_path: The path to the main configuration file.
        @param section_name: The name of the section to load (e.g., "robots", "sensors", or "objects").

        @return: A dictionary containing the loaded configuration for the specified section.
        """
        # { "name" : { ... } },
        #   "name" : { ... } } 
        section_config = {}
        for item in cfg.get(section_name) or []:
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
            
            section_config[subconfig["name"]] = subconfig["config"]

        return section_config

    
    def _reset_backend(self, channel, msg):
        self.backend.reset_simulator()
        return flag_t()
    

    def _step_simulation(self) -> None:
        self.step()
        self.backend.step()

    @abstractmethod 
    def initialize_scene(self) -> None:
        pass
    
    @abstractmethod
    def step(self) -> None:
        # enables custom behavior like printing out information etc.
        pass

    # OVERRIDE
    def spin(self) -> None:
        """!
        Runs the node’s main loop, handling LCM messages continuously until the node is finished.

        The loop calls `self._lcm.handle()` to process incoming messages. If an OSError is encountered,
        the loop will stop and the node will shut down.
        """
        while not self._done:
            try:
                self._lcm.handle_timeout(0)
                self.backend._spin_sim_components()
            except OSError as e:
                log.warning(f"LCM threw OSError {e}")
                self._done = True

    # OVERRIDE
    def kill_node(self) -> None:
        # kill driver (close ports, ...)
        self.backend.shutdown_backend()
        # kill all communication
        super().kill_node()
