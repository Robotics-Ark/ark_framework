"""Simulation node base implementation.

This module provides :class:`SimulatorNode` which serves as the entry point
for launching and controlling a simulator instance.  It loads a global
configuration, instantiates the desired backend and offers utilities for
managing the simulation lifecycle.  Concrete simulations should derive from
this class and implement :func:`initialize_scene` and :func:`step`.
"""

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
    """Base class for simulator nodes.

    A :class:`SimulatorNode` wraps a simulation backend and exposes LCM
    services for stepping and resetting the simulation.  Subclasses are
    expected to implement :func:`initialize_scene` to construct the initial
    environment and :func:`step` to execute custom logic on every simulation
    tick.
    """

    def __init__(self, global_config):
        """!Construct the simulator node.

        The constructor loads the global configuration, instantiates the
        backend and sets up basic services for stepping and resetting the
        simulator.

        @param global_config Path to the configuration YAML file or a loaded
               configuration dictionary.
        """
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
        """!Load and merge the global configuration.

        The configuration may either be provided as a path to a YAML file or
        already loaded into a dictionary.  Included sub-configurations for
        robots, sensors and objects are resolved and merged.

        @param global_config Path to the configuration file or configuration
               dictionary.
        """

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
        """!Load a sub‑configuration section.

        Sections may either be specified inline within the main configuration
        file or given as paths to external YAML files.  The returned dictionary
        maps component names to their configuration dictionaries.

        @param cfg The top level configuration dictionary.
        @param config_path Absolute path to the loaded configuration file.
        @param section_name Name of the section to load (``"robots"``,
               ``"sensors"`` or ``"objects"``).
        @return Dictionary containing the merged configuration for the section.
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
        """!Service callback resetting the backend."""
        self.backend.reset_simulator()
        return flag_t()
    

    def _step_simulation(self) -> None:
        """!Advance the simulation by one step and call :func:`step`."""
        self.step()
        self.backend.step()

    @abstractmethod 
    def initialize_scene(self) -> None:
        """!Create the initial simulation scene."""
        pass
    
    @abstractmethod
    def step(self) -> None:
        """!Hook executed every simulation step."""
        pass

    # OVERRIDE
    def spin(self) -> None:
        """!Run the node's main loop.

        The loop processes incoming LCM messages and forwards control to the
        backend for spinning all components.  It terminates when an
        ``OSError`` occurs or :attr:`_done` is set to ``True``.
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
        """!Shut down the node and the underlying backend."""
        self.backend.shutdown_backend()
        super().kill_node()
