"""Simulation node base implementation.

This module provides :class:`SimulatorNode` which serves as the entry point
for launching and controlling a simulator instance.  It loads a global
configuration, instantiates the desired backend and offers utilities for
managing the simulation lifecycle.  Concrete simulations should derive from
this class and implement :func:`initialize_scene` and :func:`step`.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ark.client.comm_infrastructure.base_node import BaseNode
from ark.tools.log import log
from ark.utils.utils import ConfigPath
from arktypes import flag_t


class SimulatorNode(BaseNode, ABC):
    """Base class for simulator nodes.

    A :class:`SimulatorNode` wraps a simulation backend and exposes LCM
    services for stepping and resetting the simulation.  Subclasses are
    expected to implement :func:`initialize_scene` to construct the initial
    environment and :func:`step` to execute custom logic on every simulation
    tick.
    """

    def __init__(
        self,
        global_config,
        observation_channels: dict[str, type] | None = None,
        action_channels: dict[str, type] | None = None,
        namespace: str = "ark",
    ):
        """!Construct the simulator node.

        The constructor loads the global configuration, instantiates the
        backend and sets up basic services for stepping and resetting the
        simulator.

        @param global_config Path to the configuration YAML file or a loaded
               configuration dictionary.
        """
        self._load_config(global_config)
        self.name = self.global_config["simulator"].get("name", "simulator")

        self.global_config["observation_channels"] = observation_channels
        self.global_config["action_channels"] = action_channels
        self.global_config["namespace"] = namespace

        super().__init__(self.name, global_config=global_config)

        log.info(
            "Initializing SimulatorNode called "
            + self.name
            + " with id "
            + self.node_id
            + " ..."
        )

        # Setup backend
        self.backend_type = self.global_config["simulator"]["backend_type"]
        if self.backend_type == "pybullet":
            from ark.system.pybullet.pybullet_backend import PyBulletBackend

            self.backend = PyBulletBackend(self.global_config)
        elif self.backend_type == "mujoco":
            from ark.system.mujoco.mujoco_backend import MujocoBackend

            self.backend = MujocoBackend(self.global_config)
        elif self.backend_type in ["isaacsim", "isaac_sim", "isaac"]:
            from ark.system.isaac.isaac_backend import IsaacSimBackend

            self.backend = IsaacSimBackend(self.global_config)
        else:
            raise ValueError(f"Unsupported backend '{self.backend_type}'")

        # to initialize a scene with objects that dont need to publish, e.g. for visuals
        self.initialize_scene()
        self.step_physics()

        ## Reset Backend Service
        reset_service_name = f"{namespace}/" + self.name + "/backend/reset/sim"
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
            global_config = ConfigPath(global_config)
        elif isinstance(global_config, Path):
            global_config = ConfigPath(str(global_config))
        if not global_config.exists():
            raise ValueError(
                "Given configuration file path does not exist, currently: "
                + global_config.str
            )

        if not global_config.is_absolute():
            global_config = global_config.resolve()

        cfg = global_config.read_yaml()

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
            raise ValueError(
                "Please provide at least name and backend_type under simulation in your config file."
            )

        try:
            config["robots"] = self._load_section(cfg, global_config, "robots")
        except KeyError as e:
            config["robots"] = {}
        try:
            config["sensors"] = self._load_section(cfg, global_config, "sensors")
        except KeyError as e:
            config["sensors"] = {}
        try:
            config["objects"] = self._load_section(cfg, global_config, "objects")
        except KeyError as e:
            config["objects"] = {}

        log.ok("Config file under " + global_config.str + " loaded successfully.")
        self.global_config = config

    def _load_section(
        self, cfg: dict[str, Any], config_path: str | ConfigPath, section_name: str
    ) -> dict[str, Any]:
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
            elif isinstance(item, str) and item.endswith(
                ".yaml"
            ):  # If it's a path to an external file
                if os.path.isabs(item):  # Check if the path is absolute
                    external_path = ConfigPath(item)
                else:  # Relative path, use the directory of the main config file
                    external_path = config_path.parent / item
                # Load the YAML file and return its content
                subconfig = external_path.read_yaml()
            else:
                log.error(
                    f"Invalid entry in '{section_name}': {item}. Please provide either a config or a path to another config."
                )
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

    def step_physics(self, num_steps: int = 25, call_step_hook: bool = False) -> None:
        """
        Advance the physics backend
        Args:
            num_steps: Number of physics ticks to run.
            call_step_hook: If True, also invoke the subclass step() each tick.

        Returns:
            None
        """
        for _ in range(max(0, num_steps)):
            if call_step_hook:
                self.step()
            self.backend.step()

    def initialize_scene(self) -> None:
        """!Create the initial simulation scene."""
        pass

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
