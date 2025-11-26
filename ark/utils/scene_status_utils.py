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

    @staticmethod
    def from_observation(obs: dict[str, Any]):
        """
        Extract the robot pose from either flattened or nested observations.
        """
        pos = ori = None

        # Flattened keys (after generate_flat_dict)
        if "proprio::pose::position" in obs:
            pos = obs.get("proprio::pose::position")
            ori = obs.get("proprio::pose::orientation")
        # Nested observation dict
        elif "proprio" in obs and "pose" in obs["proprio"]:
            pose = obs["proprio"]["pose"]
            pos = pose.get("position")
            ori = pose.get("orientation")

        if pos is None or ori is None:
            return None
        pos_arr = np.asarray(pos, dtype=np.float32).reshape(-1)
        ori_arr = np.asarray(ori, dtype=np.float32).reshape(-1)

        return RobotState(position=pos_arr, orientation=ori_arr)

    def get_position_orientation(self):
        return self.position, self.orientation

    def get_position(self):
        return self.position
