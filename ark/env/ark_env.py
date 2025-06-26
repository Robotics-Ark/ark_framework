
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
    """!
    A custom environment class for interacting with the Noah system. This class inherits
    from `gym.Env` and integrates with the Noah framework to handle observations, actions,
    rewards, and environment resets.

    @param environment_name: The name of the environment (also a node)
    @type: str
    @param action channels: Channel names where actions will be published.
    @type action_channels: List[str, type of LCM]
    @param observation channels: Channel names where observations will be listened
    @type observation_channels: List[str, type of LCM]
    @param global_config: Contain the graph, networking, simulation setup and node configurations
    @type global_config: Union
    @param sim: Whether run ArkEnv in the simulation
    @type param: bool
    """

    def __init__(
        self,
        environment_name: str,
        action_channels: List[Tuple[str, type]],
        observation_channels: List[Tuple[str, type]],
        global_config: Union[str, Dict[str, Any], Path]=None,
        sim=True) -> None:
        """!
        Initializes the Noah environment.
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
        '''
            Takes any input passed to the `step` function and returns a dictionary where:
            - Each key corresponds to an action channel.
            - Each value is the packed LCM message associated with that action channel.
            @rtype: Dict[str, Any]
        '''
        raise NotImplementedError
    
    @abstractmethod
    def observation_unpacking(self, observation_dict: Dict[str, Any]) -> Any:
        '''
            Returns a dictionary containing all previous messages received on the observation channels.
            - Each key represents the name of the channel from which the message was received.
            - Each value is the corresponding packed LCM message for that channel.

            The format of the returned dictionary is flexible and can be customized based on the desired structure.
            @rtype: Any
        '''
        raise NotImplementedError
    
    @abstractmethod
    def terminated_truncated_info(self, state: Any, action: Any, next_state: Any) -> Tuple[bool, bool, Any]:
        '''
        Returns a tuple containing the termination flag, the truncation flag, and any additional information.
        Note: when reset is called action and next state will be None
        @rtype: Tuple[bool, bool, Any]
        '''
        return False, False, None
    
    @abstractmethod
    def reward(self, state: Any, action: Any, next_state: Any) -> float:
        '''
        Returns the reward for the given state, action, and next state.
        @rtype: float
        '''
        raise NotImplementedError
    
    @abstractmethod
    def reset_objects(self):
        raise NotImplementedError
    
    def reset(self):
        '''
        Returns a tuple (obs, info) where ``info`` contains termination and
        truncation data.
        @rtype: Tuple[Any, Any]
        '''
        #self.suspend_node()
        self.reset_objects()
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
        service_name = self.global_config["simulation"]["name"] + "/backend/reset/sim"
        response = self.send_service_request(service_name=service_name, 
                                             request=flag_t(), 
                                             response_type=flag_t)
    
    def reset_component(self, name: str, **kwargs):
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
            q_init = kwargs.get("initial_position", self.global_config["robots"][name]["initial_position"])
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
        """!
        Takes an action in the environment, steps forward, and returns the resulting
        observation, reward, termination status, truncation status and info.

        @param action: The action to take in the environment.
        @type action: Any
        @param reward_function: An optional function that takes the new state and returns a reward.
        @type reward_function: Any, Any, Any
        @return: A tuple containing the observation, reward, termination flag, truncation flag, info
        @rtype: Tuple[Any, float, bool, bool, Any]
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
        """
        Generic function to load a section from the config (e.g., robots, sensors, or objects).
        It handles both inline configurations and paths to external YAML files.
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