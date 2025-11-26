import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ark.client.comm_infrastructure.instance_node import InstanceNode
from ark.env.spaces import ActionSpace, ObservationSpace
from ark.tools.log import log
from ark.utils.communication_utils import (
    build_action_space,
    build_observation_space,
    get_channel_types,
    _dynamic_observation_unpacker,
    _dynamic_action_packer,
    namespace_channels,
)
from ark.utils.data_utils import generate_flat_dict
from ark.utils.utils import ConfigPath
from arktypes import robot_init_t, flag_t, rigid_body_state_t
from gymnasium import Env


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
    @type action_channels list[tuple[str, type]]
    @param observation_channels Channels on which observations will be received.
    @type observation_channels list[tuple[str, type]]
    @param global_config Path or dictionary describing the complete Noah system
        configuration.  If ``None`` a warning is emitted and only minimal
        functionality is available.
    @type global_config Union[str, dict[str, Any], Path]
    @param sim Set ``True`` when running in simulation mode.
    @type sim bool
    """

    def __init__(
        self,
        environment_name: str,
        channel_schema: str,
        global_config: str,
        sim=True,
        namespace: str = "ark",
    ) -> None:
        """!Construct the environment.

        The constructor sets up the internal communication channels and creates
        the action and observation spaces.  The configuration can either be
        provided as a path to a YAML file or as a dictionary already loaded in
        memory.

        @param environment_name Name of the environment node.
        @param action_channels Dictionary mapping channel names to LCM
               types for actions.
        @type action_channels dict[str, type]
        @param observation_channels Dictionary mapping channel names to LCM
               types for observations.
        @type observation_channels dict[str, type]
        @param global_config Optional path or dictionary describing the system.
        @param sim If ``True`` the environment interacts with the simulator.
        """
        super().__init__(
            environment_name, global_config
        )  # TODO check why global config needed here

        schema = ConfigPath(channel_schema).read_yaml()

        # Derive observation and action channel types from schema
        obs_chans = get_channel_types(schema=schema, channel_type="observation_space")
        act_chans = get_channel_types(schema=schema, channel_type="action_space")

        # Namespace channels with rank
        observation_channels = namespace_channels(
            channels=obs_chans, namespace=namespace
        )
        action_channels = namespace_channels(channels=act_chans, namespace=namespace)

        self._flatten_action_space = schema["env"]["flatten_action_space"]
        self._flatten_obs_space = schema["env"]["flatten_obs_space"]

        self.sim = sim
        self.namespace = namespace
        self.prev_state = None

        # Create the action space
        self.ark_action_space = ActionSpace(
            action_channels, self.action_packing, self._lcm
        )
        self.ark_observation_space = ObservationSpace(
            observation_channels, self.observation_unpacking, self._lcm
        )

        self._multi_comm_handlers.append(self.ark_action_space.action_space_publisher)
        self._multi_comm_handlers.append(
            self.ark_observation_space.observation_space_listener
        )

        self._load_config(global_config)  # creates self.global_config

        # Build Gym-style observation / action spaces from schema
        self.observation_space = build_observation_space(
            schema=schema, flatten_obs_space=self._flatten_obs_space
        )
        self.action_space = build_action_space(schema=schema)

        self._obs_unpacker = _dynamic_observation_unpacker(
            schema, namespace=self.namespace
        )
        self._action_packer = _dynamic_action_packer(schema, namespace=self.namespace)

        # Reward and Termination Conditions
        self._termination_conditions = self._create_termination_conditions()
        self._reward_functions = self._create_reward_functions()

    def action_packing(self, action):
        """
        Packs the action into a task_space_command_t format.

        Expected layout:
            [EE_X, EE_Y, EE_Z, EE_QX, EE_QY, EE_QZ, EE_QW, Gripper]
        """
        return self._action_packer(action)

    def observation_unpacking(self, observation_dict):
        """
        Unpack raw LCM observations into a compact dict used by the agent.

        """
        obs = self._obs_unpacker(observation_dict)
        if self._flatten_obs_space:
            obs = generate_flat_dict(obs)
        return obs

    @abstractmethod
    def _create_termination_conditions(self): ...

    @abstractmethod
    def _create_reward_functions(self): ...

    @abstractmethod
    def reset_objects(self):
        """!Reset all objects in the environment."""
        raise NotImplementedError

    def reset(self, **kwargs) -> tuple[Any, Any]:
        """!Reset the environment.

        This method resets all user defined objects by calling
        :func:`reset_objects` and waits until fresh observations are available.
        The returned information tuple contains the termination and truncation
        flags as produced by :func:`terminated_truncated_info`.

        @return Observation after reset and information tuple.
        @rtype tuple[Any, Any]
        """
        if not self.ark_observation_space.is_ready:
            self.ark_observation_space.wait_until_observation_space_is_ready()
        self.reset_objects()
        self.ark_observation_space.is_ready = False
        self.ark_observation_space.wait_until_observation_space_is_ready()
        obs = self.ark_observation_space.get_observation()
        # Reset per-episode state for reward / termination functions
        for termination in self._termination_conditions.values():
            termination.reset()
        for reward_fn in self._reward_functions.values():
            reward_fn.reset(initial_obs=obs)

        self.prev_state = obs

        return obs, {}

    def reset_backend(self):
        """!Reset the simulation backend."""
        raise NotImplementedError("This feature is to be added soon.")

    def reset_component(self, name: str, **kwargs):
        """!Reset a single component.

        Depending on ``name`` this method sends a reset service request to a
        robot or object defined in the configuration.

        @param name Identifier of the component to reset.
        @param kwargs Optional parameters such as ``base_position`` or
               ``initial_configuration`` used to override the configuration.
        """
        if self.global_config is None:
            log.error(
                "No configuration file provided, so no objects can be found. Please provide a valid configuration file."
            )
            return
        # search through config
        # if name in [robot["name"] for robot in self.global_config["robots"]]:
        if name in self.global_config["robots"]:

            service_name = f"{self.namespace}/" + name + "/reset/"
            if self.sim:
                service_name = service_name + "sim"

            request = robot_init_t()
            request.name = name
            request.position = kwargs.get(
                "base_position", self.global_config["robots"][name]["base_position"]
            )
            request.orientation = kwargs.get(
                "base_orientation",
                self.global_config["robots"][name]["base_orientation"],
            )
            q_init = kwargs.get(
                "initial_configuration",
                self.global_config["robots"][name]["initial_configuration"],
            )
            request.n = len(q_init)
            request.q_init = q_init

        elif name in self.global_config["sensors"]:
            log.error(f"Can't reset a sensor (called for {name}).")

        # elif name in [obj["name"] for obj in self.global_config["objects"]]:
        elif name in self.global_config["objects"]:
            service_name = f"{self.namespace}/" + name + "/reset/"
            if self.sim:
                service_name = service_name + "sim"

            request = rigid_body_state_t()
            request.name = name
            request.position = kwargs.get(
                "base_position", self.global_config["objects"][name]["base_position"]
            )
            request.orientation = kwargs.get(
                "base_orientation",
                self.global_config["objects"][name]["base_orientation"],
            )

            # TODO for now we only work with position init, may add velocity in the future
            request.lin_velocity = kwargs.get("base_velocity", [0.0, 0.0, 0.0])
            request.ang_velocity = kwargs.get("base_angular_velocity", [0.0, 0.0, 0.0])

        else:
            log.error(f"Component {name} not part of the system.")

        _ = self.send_service_request(
            service_name=service_name, request=request, response_type=flag_t
        )

    def _step_termination(self, obs, action, info=None):
        """
        Step and aggregate termination conditions

        Args:
            env (Environment): Environment instance
            action (n-array): 1D flattened array of actions executed by all agents in the environment
            info (None or dict): Any info to return

        Returns:
            2-tuple:
                - float: aggregated termination at the current timestep
                - dict: any information passed through this function or generated by this function
        """
        # Get all dones and successes from individual termination conditions
        dones = []
        successes = []
        info = dict() if info is None else info
        if "termination_conditions" not in info:
            info["termination_conditions"] = dict()
        for name, termination_condition in self._termination_conditions.items():
            d, s = termination_condition.step(obs=obs, action=action)
            dones.append(d)
            successes.append(s)
            info["termination_conditions"][name] = {
                "done": d,
                "success": s,
            }
        # Any True found corresponds to a done / success
        done = sum(dones) > 0
        success = sum(successes) > 0

        # Populate info
        info["success"] = success
        return done, info

    def _step_reward(self, obs, action, info=None):
        """
        Step and aggregate reward functions

        Args:
            env (Environment): Environment instance
            action (n-array): 1D flattened array of actions executed by all agents in the environment
            info (None or dict): Any info to return

        Returns:
            2-tuple:
                - float: aggregated reward at the current timestep
                - dict: any information passed through this function or generated by this function
        """
        # Make sure info is a dict
        total_info = dict() if info is None else info
        # We'll also store individual reward split as well
        breakdown_dict = dict()
        # Aggregate rewards over all reward functions
        total_reward = 0.0
        for reward_name, reward_function in self._reward_functions.items():
            reward, reward_info = reward_function.step(obs=obs, action=action)
            total_reward += reward
            breakdown_dict[reward_name] = reward
            total_info[reward_name] = reward_info

        # Store breakdown dict
        total_info["reward_breakdown"] = breakdown_dict

        return total_reward, total_info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, Any]:
        """!Advance the environment by one step.

        The provided ``action`` is packed and published.  The function then
        waits for a new observation, computes the reward and termination flags
        and returns all gathered information.

        @param action Action provided by the agent.
        @return tuple of observation, reward, termination flag, truncation flag
                and an optional info object.
        @rtype tuple[Any, float, bool, bool, Any]
        """
        if self.prev_state == None:
            raise ValueError("Please call reset() before calling step().")

        self.ark_action_space.pack_and_publish(action)

        # Wait for the observation space to be ready
        self.ark_observation_space.wait_until_observation_space_is_ready()

        # Get the observation
        obs = self.ark_observation_space.get_observation()

        # Calculate reward
        done, done_info = self._step_termination(obs=obs, action=action)
        reward, reward_info = self._step_reward(obs=obs, action=action)
        truncated = True if done and not done_info["success"] else False

        info = {
            "reward": reward_info,
            "done": done_info,
        }

        self.prev_state = obs

        if done or truncated:
            print(f"Episode terminated, {done}, {truncated}")
        return obs, reward, done, truncated, info

    def _load_config(self, global_config: str | ConfigPath) -> None:
        """!Load and merge the environment configuration.

        The configuration can be provided as a path to a YAML file or as an
        already parsed dictionary.  Sections describing robots, sensors and
        objects may themselves reference additional YAML files which are loaded
        and merged.

        @param global_config Path or dictionary to parse.
        """
        if global_config is None:
            log.warning("No configuration file provided. Using default configuration.")
            self.global_config = None
            return
        if isinstance(global_config, str):
            global_config = ConfigPath(global_config)
        if isinstance(global_config, Path):
            global_config = ConfigPath(str(global_config))

        if isinstance(global_config, ConfigPath) and not global_config.exists():
            log.error(
                f"Given configuration file path does not exist: {global_config.str}"
            )
            return

        if isinstance(global_config, ConfigPath) and not global_config.is_absolute():
            global_config = global_config.resolve()

        cfg = global_config.read_yaml()

        config = {
            "network": cfg.get("network", None) if isinstance(cfg, dict) else None,
            "simulator": cfg.get("simulator", None) if isinstance(cfg, dict) else None,
            "robots": (
                self._load_section(cfg, global_config, "robots")
                if cfg.get("robots")
                else {}
            ),
            "sensors": (
                self._load_section(cfg, global_config, "sensors")
                if cfg.get("sensors")
                else {}
            ),
            "objects": (
                self._load_section(cfg, global_config, "objects")
                if cfg.get("objects")
                else {}
            ),
        }

        if not config["simulator"]:
            log.error(
                "Please provide at least name and backend_type under 'simulator' in your config file."
            )

        log.info(
            f"Config file under {global_config.str if global_config else 'default configuration'} loaded successfully."
        )
        self.global_config = config

    def _load_section(
        self, cfg: [str, Any], config_path: ConfigPath, section_name: str
    ) -> dict[str, Any]:
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
        section_config: dict[str, Any] = {}

        for item in cfg.get(section_name, []):
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

    def close(self):
        """!Gracefully shut down communications and background threads."""
        self.suspend_communications(services=True)
        spin_thread = getattr(self, "spin_thread", None)
        if spin_thread and spin_thread.is_alive():
            spin_thread.join(timeout=1.0)
