from ark.env.franka_env import FrankaEnv
from gymnasium import spaces


def test_franka_env_spaces_construction(tmp_path):
    # Minimal synthetic schema matching build_observation_space / build_action_space
    schema_yaml = tmp_path / "schema.yaml"
    schema_yaml.write_text(
        """
observation:
  proprio:
    - from: franka/joint_states/sim
      using: joint_state
      dim: [7]
  sensors:
    - name: IntelRealSense
      select: ["rgb"]
      image_height: 64
      image_width: 64

action:
  action:
    - from: franka/cartesian_command/sim
      using: task_space_command
      dim: [8]

observation_space:
  proprio:
    - using: joint_state
      dim: [7]
  sensors:
    - name: IntelRealSense
      select: ["rgb"]
      image_height: 64
      image_width: 64

action_space:
  action:
    - using: task_space_command
      dim: [8]
"""
    )

    # For this unit test we don't rely on a real running Ark system.
    # Point config_path to an existing global_config.yaml if available,
    # or to an empty file; ArkEnv will warn but construction should succeed.
    dummy_cfg = tmp_path / "global_config.yaml"
    dummy_cfg.write_text("network: {}\nrobots: {}\nsensors: {}\nobjects: {}\n")

    env = FrankaEnv(
        channel_schema=str(schema_yaml),
        global_config=str(dummy_cfg),
    )

    # Observation space is a Dict with 'proprio' and 'sensors'
    assert isinstance(env.observation_space, spaces.Dict)
    assert "proprio" in env.observation_space.spaces
    assert "sensors" in env.observation_space.spaces

    proprio_space = env.observation_space.spaces["proprio"]
    assert isinstance(proprio_space, spaces.Box)
    assert proprio_space.shape == (7,)

    # Action space is a Dict with 'proprio' key holding a Box of dim 8
    assert isinstance(env.action_space, spaces.Dict)
    assert "proprio" in env.action_space.spaces
    act_space = env.action_space.spaces["proprio"]
    assert isinstance(act_space, spaces.Box)
    assert act_space.shape == (8,)

    # Sampling should work and return correct shapes
    obs_sample = env.observation_space.sample()
    act_sample = env.action_space.sample()

    assert "proprio" in obs_sample
    assert obs_sample["proprio"].shape == (7,)
    assert act_sample["proprio"].shape == (8,)
