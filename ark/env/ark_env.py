
import time
from typing import Optional, Callable, Any, Tuple, Dict, List, Union
from pathlib import Path
import yaml
import os

from gymnasium import Env

from arktypes import float_t, robot_init_t, flag_t, rigid_body_state_t
from ark.tools.log import log
from ark.client.comm_infrastructure.instance_node import InstanceNode
from ark.env.spaces import ActionSpace, ObservationSpace
from ark.client.comm_handler.service import send_service_request

from abc import ABC, abstractmethod


class ArkEnv(Env, InstanceNode, ABC):
    """!ArkEnv base class.

    This environment integrates the Ark system with the :mod:`gymnasium` API.  It
    handles action publishing, observation retrieval and exposes helper utilities
    for resetting parts of the system.  Sub‑classes are expected to implement the
    packing/unpacking logic for messages as well as the reward and termination
    functions.

    @param environment_name Name of the environment (also the node name).
    @type environment_name str
    @param action_channels Channels on which actions will be published.
    @type action_channels List[Tuple[str, type]]
    @param observation_channels Channels on which observations will be received.
    @type observation_channels List[Tuple[str, type]]
    @param global_config Path or dictionary describing the complete Noah system
        configuration.  If ``None`` a warning is emitted and only minimal
        functionality is available.
    @type global_config Union[str, Dict[str, Any], Path]
    @param sim Set ``True`` when running in simulation mode.
    @type sim bool
    """

    def __init__(
        self,
        environment_name: str,
        action_channels: Dict[str, type],
        observation_channels: Dict[str, type],
        global_config: Union[str, Dict[str, Any], Path]=None,
        sim=True) -> None:
        """!Construct the environment.

        The constructor sets up the internal communication channels and creates
        the action and observation spaces.  The configuration can either be
        provided as a path to a YAML file or as a dictionary already loaded in
        memory.

        @param environment_name Name of the environment node.
        @param action_channels Dictionary mapping channel names to LCM
               types for actions.
        @type action_channels Dict[str, type]
        @param observation_channels Dictionary mapping channel names to LCM
               types for observations.
        @type observation_channels Dict[str, type]
        @param global_config Optional path or dictionary describing the system.
        @param sim If ``True`` the environment interacts with the simulator.
        """
        super().__init__(environment_name, global_config)
        
        self._load_config(global_config) # creates self.global_config
        self.sim = sim
        # Create the action space
        self.action_space = ActionSpace(action_channels, self.action_packing, self._lcm)
        self.observation_space = ObservationSpace(observation_channels, self.observation_unpacking, self._lcm)

        self._multi_comm_handlers.append(self.action_space.action_space_publisher)
        self._multi_comm_handlers.append(self.observation_space.observation_space_listener)

        self.prev_state = None

    @abstractmethod
    def action_packing(self, action: Any) -> Dict[str, Any]:
        '''!Serialize an action.

        This method converts the high level action passed to :func:`step` into
        a dictionary that can be published over LCM.  The dictionary keys are
        channel names and the values are already packed LCM messages.

        @param action The high level action provided by the agent.
        @return A mapping from channel names to packed LCM messages.
        @rtype Dict[str, Any]
        '''
        raise NotImplementedError
    
    @abstractmethod
    def observation_unpacking(self, observation_dict: Dict[str, Any]) -> Any:
        '''!Deserialize observations.

        ``observation_dict`` contains the raw LCM messages keyed by channel
        name.  Implementations should convert these messages into a convenient
        format for the agent.

        @param observation_dict Raw messages indexed by channel name.
        @return Processed observation in an arbitrary format.
        @rtype Any
        '''
        raise NotImplementedError
    
    @abstractmethod
    def terminated_truncated_info(self, state: Any, action: Any, next_state: Any) -> Tuple[bool, bool, Any]:
        '''!Evaluate episode status.

        Determines whether the episode has terminated or been truncated after
        transitioning from ``state`` to ``next_state`` by ``action``.  When
        :func:`reset` is called ``action`` and ``next_state`` will be ``None``.

        @param state Previous environment state.
        @param action Action taken to reach ``next_state``.
        @param next_state New state after the action.
        @return Tuple of termination flag, truncation flag and optional info.
        @rtype Tuple[bool, bool, Any]
        '''
        return False, False, None
    
    @abstractmethod
    def reward(self, state: Any, action: Any, next_state: Any) -> float:
        '''!Compute the reward for a transition.

        Sub‑classes must implement the task specific reward computation given
        a state transition.

        @param state Environment state before the action.
        @param action Action taken by the agent.
        @param next_state Environment state after the action.
        @return Scalar reward.
        @rtype float
        '''
        raise NotImplementedError
    
    @abstractmethod
    def reset_objects(self):
        """!Reset all objects in the environment."""
        raise NotImplementedError
    
    def reset(self, **kwargs) -> Tuple[Any, Any]:
        '''!Reset the environment.

        This method resets all user defined objects by calling
        :func:`reset_objects` and waits until fresh observations are available.
        The returned information tuple contains the termination and truncation
        flags as produced by :func:`terminated_truncated_info`.

        @return Observation after reset and information tuple.
        @rtype Tuple[Any, Any]
        '''
        #self.suspend_node()
        self.reset_objects(**kwargs)
        self.observation_space.is_ready = False
        #self.restart_node()
        # if self.sim:
        #     self.reset_backend()
        self.observation_space.wait_until_observation_space_is_ready()
        obs = self.observation_space.get_observation()
        info = self.terminated_truncated_info(obs, None, None)
        self.prev_state = obs

        return obs, info

    def reset_backend(self):
        """!Reset the simulation backend."""
        raise NotImplementedError("This feature is to be added soon.")
        # service_name = self.global_config["simulator"]["name"] + "/backend/reset/sim"
        # self.send_service_request(
        #     service_name=service_name,
        #     request=flag_t(),
        #     response_type=flag_t,
        # )
    
    def reset_component(self, name: str, **kwargs):
        """!Reset a single component.

        Depending on ``name`` this method sends a reset service request to a
        robot or object defined in the configuration.

        @param name Identifier of the component to reset.
        @param kwargs Optional parameters such as ``base_position`` or
               ``initial_configuration`` used to override the configuration.
        """
        if self.global_config is None:
            log.error("No configuration file provided, so no objects can be found. Please provide a valid configuration file.")
            return
        # search through config
        #if name in [robot["name"] for robot in self.global_config["robots"]]:
        if name in self.global_config["robots"]:
            
            service_name = name + "/reset/"
            if self.sim:
                service_name = service_name + 'sim'
                
            request = robot_init_t()
            request.name = name
            request.position = kwargs.get("base_position", self.global_config["robots"][name]["base_position"])
            request.orientation = kwargs.get("base_orientation", self.global_config["robots"][name]["base_orientation"])
            q_init = kwargs.get("initial_configuration", self.global_config["robots"][name]["initial_configuration"])
            request.n = len(q_init)
            request.q_init = q_init
                        
        #elif name in [sensor["name"] for sensor in self.global_config["sensors"]]:
        elif name in self.global_config["sensors"]:
            log.error(f"Can't reset a sensor (called for {name}).")
            
        #elif name in [obj["name"] for obj in self.global_config["objects"]]:
        elif name in self.global_config["objects"]:
            service_name = name + "/reset/"
            if self.sim:
                service_name = service_name + 'sim'
            
            request = rigid_body_state_t()
            request.name = name
            request.position = kwargs.get("base_position", self.global_config["objects"][name]["base_position"])
            request.orientation = kwargs.get("base_orientation", self.global_config["objects"][name]["base_orientation"])
 
            # TODO for now we only work with position init, may add velocity in the future
            request.lin_velocity = kwargs.get("base_velocity", [0.0, 0.0, 0.0])
            request.ang_velocity = kwargs.get("base_angular_velocity", [0.0, 0.0, 0.0])

        else:
            log.error(f"Component {name} not part of the system.")
        
        response = self.send_service_request(service_name=service_name, 
                                             request=request, 
                                             response_type=flag_t)
        
        
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Any]:
        """!Advance the environment by one step.

        The provided ``action`` is packed and published.  The function then
        waits for a new observation, computes the reward and termination flags
        and returns all gathered information.

        @param action Action provided by the agent.
        @return Tuple of observation, reward, termination flag, truncation flag
                and an optional info object.
        @rtype Tuple[Any, float, bool, bool, Any]
        """
        if self.prev_state == None: 
            raise ValueError("Please call reset() before calling step().")
        
        self.action_space.pack_and_publish(action)

        # Wait for the observation space to be ready
        self.observation_space.wait_until_observation_space_is_ready()

        # Get the observation
        obs = self.observation_space.get_observation()
        reward = self.reward(self.prev_state, action, obs)
        terminated, truncated, info = self.terminated_truncated_info(self.prev_state, action, obs)
        self.prev_state = obs

        # Return the observation (excluding termination and truncation flags), reward, and flags
        return obs, reward, terminated, truncated, info

    def _load_config(self, global_config) -> None:
        """!Load and merge the environment configuration.

        The configuration can be provided as a path to a YAML file or as an
        already parsed dictionary.  Sections describing robots, sensors and
        objects may themselves reference additional YAML files which are loaded
        and merged.

        @param global_config Path or dictionary to parse.
        """
        if isinstance(global_config, str):
            global_config = Path(global_config)
        elif global_config is None:
            log.warning("No configuration file provided. Using default configuration.")
            # Assign a default empty configuration
            self.global_config = None
            return
        elif not global_config.exists():
            log.error("Given configuration file path does not exist.")
            return  # Early return if file doesn't exist

        if global_config is not None and not global_config.is_absolute():
            global_config = global_config.resolve()

        if global_config is not None:
            config_path = str(global_config)
            with open(config_path, 'r') as file:
                cfg = yaml.safe_load(file)   

        # merge with subconfigs
        config = {}
        try:
            config["network"] = cfg.get("network", None)
        except:
            config["network"] = None
        try:
            config["simulator"] = cfg.get("simulator", None)
        except:
            log.error("Please provide at least name and backend_type under simulation in your config file.")
        
        # Load robots, sensors, and objects 
        config["robots"] = self._load_section(cfg, config_path, "robots") if cfg.get("robots") else {}
        config["sensors"] = self._load_section(cfg, config_path, "sensors") if cfg.get("sensors") else {}
        config["objects"] = self._load_section(cfg, config_path, "objects") if cfg.get("objects") else {}

        log.info(f"Config file under {config_path if global_config else 'default configuration'} loaded successfully.")
        self.global_config = config
        
        
    def _load_section(self, cfg, config_path, section_name):
        """!Load a sub-section from the configuration.

        Sections can either be provided inline in ``cfg`` or as a path to an
        additional YAML file.  This helper returns a dictionary mapping component
        names to their configuration dictionaries.

        @param cfg Parsed configuration dictionary.
        @param config_path Path to the root configuration file, used to resolve
               relative includes.
        @param section_name Section within ``cfg`` to load.
        @return Dictionary with component names as keys and their configurations
                as values.
        """
        # { "name" : { ... } },
        #   "name" : { ... } } 
        section_config = {}

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
            
            section_config[subconfig["name"]] = subconfig["config"]

        return section_config
