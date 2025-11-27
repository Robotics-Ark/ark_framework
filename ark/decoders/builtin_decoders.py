from typing import Any

import numpy as np
from ark.decoders.registry import register_decoder

from arktypes.utils.unpack import image, depth

OBS_SCHEMA = {
    "joint_state": ["position", "velocity", "effort"],
    "pose": ["position", "orientation"],
    "rgbd": ["rgb", "depth"],
    "rigid_body_state": [
        "position",
        "orientation",
        "linear_velocity",
        "angular_velocity",
    ],
}


@register_decoder("joint_state")
def decode_joint_state(msg) -> dict[str, Any]:
    """
    Decode a joint_state message into a dictionary of numpy arrays.
    Args:
        msg: A message object with attributes `position`, `velocity`, and `effort`.

    Returns:
        A dictionary containing:
        - "position": numpy array of joint positions (float32)
        - "velocity": numpy array of joint velocities (float32)
        - "effort": numpy array of joint efforts/torques (float32)

    """
    return {
        "position": np.asarray(msg.position, dtype=np.float32),
        "velocity": np.asarray(msg.velocity, dtype=np.float32),
        "effort": np.asarray(msg.effort, dtype=np.float32),
    }


@register_decoder("pose")
def decode_pose(msg) -> dict[str, Any]:
    """
    Decode a pose message into a dictionary of numpy arrays.
    Args:
        msg: A message object with attributes `position` and `orientation`.

    Returns:
        A dictionary containing:
        - "position": numpy array of position coordinates (float32)
        - "orientation": numpy array of orientation quaternion (float32)

    """
    return {
        "position": np.asarray(msg.position, dtype=np.float32),
        "orientation": np.asarray(msg.orientation, dtype=np.float32),
    }


@register_decoder("rgbd")
def decode_rgbd(msg) -> dict[str, Any]:
    """
    Decode a rgbd message into a dictionary of numpy arrays.
    Args:
        msg: A message object with attributes `image` and `depth`.

    Returns:
        A dictionary containing:
        - "rgb": image data
        - "depth": depth data

    """
    return {
        "rgb": image(msg.image),
        "depth": depth(msg.depth),  # TODO check is this optional
    }


@register_decoder("rigid_body_state")
def decode_rgbd(msg) -> dict[str, Any]:
    """
    Decode a rigid_body_state message into a dictionary of numpy arrays.
    Args:
        msg: A message object with attributes `position`, `orientation`, `lin_velocity`, and `ang_velocity`.

    Returns:
        A dictionary containing:
        - "position": numpy array of body position (float32)
        - "orientation": numpy array of body orientation quaternion (float32)
        - "linear_velocity": numpy array of linear velocity (float32)
        - "angular_velocity": numpy array of angular velocity (float32)

    """
    return {
        "position": np.asarray(msg.position, dtype=np.float32),
        "orientation": np.asarray(msg.orientation, dtype=np.float32),
        "linear_velocity": np.asarray(msg.lin_velocity, dtype=np.float32),
        "angular_velocity": np.asarray(msg.ang_velocity, dtype=np.float32),
    }
