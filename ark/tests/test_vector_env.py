import argparse
import os
from pathlib import Path

import numpy as np

from ark.env.ark_env import ArkEnv

from ark.env.franka_env import FrankaEnv
from ark.env.vector_env import make_vector_env, make_sim
from ark.utils.communication_utils import (
    build_action_space,
    build_observation_space,
    get_channel_types,
    _dynamic_observation_unpacker,
)
from ark.utils.utils import ConfigPath


class DemoEnv(ArkEnv):
    def __init__(self, channel_schema, global_config, namespace: str, sim: bool = True):
        super().__init__(
            environment_name="demo_env",
            channel_schema=channel_schema,
            global_config=global_config,
            namespace=namespace,
            sim=sim,
        )

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset()

        print("\n[DEBUG] reset() obs shapes for env:", self.namespace)
        # self._print_obs_shapes(obs)

        return obs, info

    def _print_obs_shapes(self, obs):
        for k, v in obs.items():
            try:
                arr = np.asarray(v)
                print(f"  {k:35s} shape={arr.shape} dtype={arr.dtype}")
            except Exception as e:
                print(f"  {k:35s} ERROR converting: {e}")

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info

    def reset_objects(self):
        self.reset_component("cube")
        self.reset_component("target")
        self.reset_component("franka")

    @staticmethod
    def _create_reward_functions():
        return {}

    @staticmethod
    def _create_termination_conditions():
        return {}


def run_franka_vector_demo(
    channel_schema: str, config_path: str, num_envs: int = 2, num_steps: int = 5
) -> None:
    """
    Simple driver to instantiate a vectorized FrankaEnv batch via
    make_vector_env and print observations, rewards and done flags
    for a few steps to verify behavior.

    This is intended as an integration smoke-test rather than a pure
    unit test and assumes the Ark backend is available.
    """
    print(f"Creating {num_envs} FrankaEnv instances...")
    # make_sim(
    #     channel_schema=channel_schema,
    #     global_config=config_path,
    #     namespace="test",  # namespace,
    # )
    #
    # env = DemoEnv(
    #     namespace="test",
    #     channel_schema=channel_schema,
    #     global_config=config_path,
    #     sim=True,
    # )
    # obs, info = env.reset()
    # for k, v in obs.items():
    #     print(k, v.shape)
    #
    # make_sim(
    #     channel_schema=channel_schema,
    #     global_config=config_path,
    #     namespace="test2",
    # )
    #
    # env = DemoEnv(
    #     namespace="test2",
    #     channel_schema=channel_schema,
    #     global_config=config_path,
    #     sim=True,
    # )
    # obs, info = env.reset()
    # for k, v in obs.items():
    #     print(k, v.shape)

    vec_env = make_vector_env(
        DemoEnv,
        num_envs=num_envs,
        channel_schema=channel_schema,
        global_config=config_path,
        sim=True,
        asynchronous=False,
    )

    obs, info = vec_env.reset()
    # print("Initial obs:")
    # for i in range(vec_env.num_envs):
    #     per_env_obs = {k: v[i] for k, v in obs.items()}
    #     print(f"  env[{i}] initial obs:", per_env_obs)

    for step in range(num_steps):
        # Build distinct actions per env so we can trace them
        actions = vec_env.action_space.sample()
        if isinstance(actions, np.ndarray):
            # For Box spaces: ensure each env has a different first element
            for i in range(vec_env.num_envs):
                actions[i, ...] = i

        obs, rewards, terminated, truncated, infos = vec_env.step(actions)

        print(f"\nStep {step}:")
        for i in range(vec_env.num_envs):
            per_env_obs = {k: v[i] for k, v in obs.items()}
            print(f"  env[{i}]:")
            # print("    action   :", actions[i])
            # print("    obs      :", per_env_obs)
            # print("    reward   :", rewards[i])
            # print("    terminated:", terminated[i], "truncated:", truncated[i])

    print("Final reset")
    obs, info = vec_env.reset()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test for make_vector_env using FrankaEnv instances. "
            "Assumes Ark comms and simulation are running."
        )
    )
    parser.add_argument(
        "--channel-schema",
        type=str,
        default="ark_framework/ark/configs/franka_panda.yaml",
        help="Path to RL channel schema YAML (with observation_space/action_space).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="ark_diffusion_policies_on_franka/diffusion_policy/config/global_config.yaml",
        help="Path to Ark global_config.yaml.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of parallel Franka envs."
    )
    parser.add_argument(
        "--num-steps", type=int, default=5, help="Number of rollout steps to print."
    )

    args = parser.parse_args()

    channel_schema = os.path.abspath(args.channel_schema)
    config_path = os.path.abspath(args.config_path)

    if not Path(channel_schema).exists():
        raise FileNotFoundError(f"Channel schema not found: {channel_schema}")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config path not found: {config_path}")

    run_franka_vector_demo(
        channel_schema=channel_schema,
        config_path=config_path,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
    )


if __name__ == "__main__":
    main()
