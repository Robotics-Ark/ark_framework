from __future__ import annotations

import uuid
from multiprocessing import Process
from typing import Callable, Type, Any

from ark.system.simulation.simulator_node import SimulatorNode
from ark.utils.communication_utils import (
    get_channel_types,
    namespace_channels,
)
from ark.utils.utils import ConfigPath
from gymnasium import Env
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv


def run_simulator_proc(
    global_config: str,
    observation_channels: dict[str, Any],
    action_channels: dict[str, Any],
    namespace: str,
) -> None:
    """
    Launch a simulator node as a blocking process.
    Args:
        global_config: Path to the global configuration file for the simulator.
        observation_channels: Dictionary defining observation channels and their types.
        action_channels: Dictionary defining action channels and their types.
        namespace: Unique namespace for the simulator instance to avoid channel conflicts.

    Returns:
        None

    """
    node = SimulatorNode(
        global_config=global_config,
        observation_channels=observation_channels,
        action_channels=action_channels,
        namespace=namespace,
    )
    node.spin()


def _make_env_thunk(
    env_cls: Type[Env],
    namespace: str,
    channel_schema: str,
    global_config: str,
    sim: bool,
) -> Callable[[], Env]:
    """
    Create a thunk (callable) that initializes a new environment instance.
    Args:
        env_cls: The environment class to instantiate.
        namespace: Unique namespace for the environment instance.
        channel_schema: Schema defining observation and action channels.
        global_config: Path to the global configuration file.
        sim: Whether to connect the environment to a simulator process.

    Returns:

    """

    def _init() -> Env:
        return env_cls(
            namespace=namespace,
            channel_schema=channel_schema,
            global_config=global_config,
            sim=sim,
        )

    return _init


def make_sim(channel_schema: str, global_config: str, namespace: str) -> Process:
    """
    Spawn a simulator process for a specific namespace and channel configuration.
    Args:
        channel_schema: Path to the YAML schema defining observation and action channels.
        global_config: Path to the global configuration for the simulator.
        namespace: Unique namespace for the simulator instance.

    Returns:
        A daemon Process object running the simulator node.

    """
    schema = ConfigPath(channel_schema).read_yaml()

    # Derive observation and action channel types from schema
    obs_chans = get_channel_types(schema=schema, channel_type="observation_space")
    act_chans = get_channel_types(schema=schema, channel_type="action_space")
    # Namespace channels
    observation_channels = namespace_channels(channels=obs_chans, namespace=namespace)
    action_channels = namespace_channels(channels=act_chans, namespace=namespace)

    sim_proc = Process(
        target=run_simulator_proc,
        args=(global_config, observation_channels, action_channels, namespace),
        daemon=True,
    )
    sim_proc.start()
    return sim_proc


def _cleanup_sim_procs(sim_procs: list[Process]) -> None:
    """
    Terminate and join any running simulator processes.
    Args:
        sim_procs: List of simulator Process objects to clean up.

    Returns:
        None.

    """
    for proc in sim_procs:
        if proc.is_alive():
            proc.terminate()
    for proc in sim_procs:
        if proc.is_alive():
            proc.join(timeout=1.0)


def _attach_cleanup(env: VectorEnv, sim_procs: list[Process]) -> VectorEnv:
    """
    Wrap an environment's close method to also terminate simulator processes.
    Args:
        env: The vectorized environment to wrap.
        sim_procs: List of simulator Process objects associated with this environment.

    Returns:
        The same environment instance with a modified close method that ensures
        simulator processes are cleaned up.

    """
    original_close = getattr(env, "close", None)

    def _close():
        try:
            if callable(original_close):
                original_close()
        finally:
            _cleanup_sim_procs(sim_procs)

    env.close = _close
    env._sim_procs = sim_procs
    return env


def make_vector_env(
    env_cls: Type[Env],
    num_envs: int,
    channel_schema: str,
    global_config: str,
    sim: bool = True,
    asynchronous: bool = True,
) -> VectorEnv:
    """
    Create a vectorized environment with optional simulator processes.
    Args:
        env_cls: The environment class to instantiate.
        num_envs: Number of environment instances to create (must be >= 1).
        channel_schema: Path to the YAML schema defining observation and action channels.
        global_config: Path to the global configuration file.
        sim: Whether to connect the environment to a simulator process.
        asynchronous: Whether to use asynchronous (AsyncVectorEnv) or synchronous (SyncVectorEnv).

    Returns:
        A vectorized environment instance.
    """

    if num_envs <= 0:
        raise ValueError("num_envs must be >= 1")

    thunks = []
    sim_procs = []
    for rank in range(num_envs):
        namespace = uuid.uuid4().hex[:8]

        if sim:
            sim_proc = make_sim(
                channel_schema=channel_schema,
                global_config=global_config,
                namespace=namespace,
            )
            sim_procs.append(sim_proc)

        thunks.append(
            _make_env_thunk(env_cls, namespace, channel_schema, global_config, sim)
        )

        env: VectorEnv = (
            AsyncVectorEnv(thunks) if asynchronous else SyncVectorEnv(thunks)
        )

        if sim_procs:
            env = _attach_cleanup(env, sim_procs)

        return env
