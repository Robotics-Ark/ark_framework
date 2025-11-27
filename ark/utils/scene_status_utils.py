from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ObjectState:
    """
    Lightweight carrier for object pose used by reward/termination logic.

    Works with both nested and flattened observation dictionaries by looking
    for common field names.
    """

    name: str
    position: np.ndarray
    orientation: np.ndarray | None = None

    def distance_to(self, other: "ObjectState") -> float:
        return float(np.linalg.norm(self.position - other.position))

    @staticmethod
    def from_observation(obs: dict[str, Any], name: str):
        """
        Try to extract an object's position/orientation from the observation dict.
        Supports:
        - nested: obs[name] = {"position": ..., "orientation": ...}
        - flattened: keys like f"{name}::position", f"{name}::orientation"
        """

        pos = ori = None

        # Flattened observation dict (most common in this repo)
        if f"objects::{name}::position" in obs:
            pos = obs.get(f"objects::{name}::position")
            ori = obs.get(f"objects::{name}::orientation")
        # Nested observation dict
        elif "objects" in obs and name in obs["objects"]:
            obj = obs["objects"][name]
            pos = obj.get("position")
            ori = obj.get("orientation")

        if pos is None:
            return None
        pos_arr = np.asarray(pos, dtype=np.float32).reshape(-1)
        ori_arr = None
        if ori is not None:
            ori_arr = np.asarray(ori, dtype=np.float32).reshape(-1)
        return ObjectState(name=name, position=pos_arr, orientation=ori_arr)


@dataclass
class RobotState:
    """
    Lightweight carrier for object pose used by reward/termination logic.

    Works with both nested and flattened observation dictionaries by looking
    for common field names.
    """

    position: np.ndarray
    orientation: np.ndarray
    joint_positions: np.ndarray

    @staticmethod
    def from_observation(obs: dict[str, Any]):
        """
        Extract the robot pose from either flattened or nested observations.
        """
        pos = ori = joints = None

        # Flattened keys (after generate_flat_dict)
        if "proprio::pose::position" in obs:
            pos = obs.get("proprio::pose::position")
            ori = obs.get("proprio::pose::orientation")

        if "proprio::joint_state::position" in obs:
            joints = obs.get("proprio::joint_state::position")

        if "proprio" in obs:
            proprio = obs["proprio"]

            # Nested pose
            if "pose" in proprio:
                pose = proprio["pose"]
                pos = pos or pose.get("position")
                ori = ori or pose.get("orientation")

            # Nested joint state
            if "joint_state" in proprio:
                js = proprio["joint_state"]
                joints = joints or js.get("position")

        if pos is None or ori is None:
            return None
        pos_arr = np.asarray(pos, dtype=np.float32).reshape(-1)
        ori_arr = np.asarray(ori, dtype=np.float32).reshape(-1)
        joints_arr = np.asarray(joints, dtype=np.float32).reshape(-1)

        return RobotState(
            position=pos_arr, orientation=ori_arr, joint_positions=joints_arr
        )

    def get_position_orientation(self):
        return self.position, self.orientation

    def get_position(self):
        return self.position

    def get_current_joint_states(self):
        return self.joint_positions.copy()


def task_space_action_from_obs(
    obs: dict[str, Any], action_dim: int, num_envs: int = 1
) -> np.ndarray:
    """
    Build an initial task-space action (xyz + quat + gripper) from the reset observation.
    Falls back to zeros if position/orientation are missing.
    """
    init = np.zeros((num_envs, action_dim), dtype=np.float32)
    if not isinstance(obs, dict):
        return init

    pos = obs.get("proprio::pose::position")
    ori = obs.get("proprio::pose::orientation")

    if (pos is None or ori is None) and "proprio" in obs:
        proprio = obs["proprio"]
        pose = proprio.get("pose", {})
        pos = pos if pos is not None else pose.get("position")
        ori = ori if ori is not None else pose.get("orientation")

    if pos is None or ori is None:
        return init

    pos_arr = np.asarray(pos, dtype=np.float32).reshape(num_envs, -1)
    ori_arr = np.asarray(ori, dtype=np.float32).reshape(num_envs, -1)

    pos_len = min(3, pos_arr.shape[1], action_dim)
    init[:, :pos_len] = pos_arr[:, :pos_len]

    ori_start = pos_len
    ori_len = min(4, ori_arr.shape[1], action_dim - ori_start)
    if ori_len > 0:
        init[:, ori_start : ori_start + ori_len] = ori_arr[:, :ori_len]

    return init
