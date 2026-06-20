import numpy as np
from typing import Any
from numpy.typing import NDArray
from dataclasses import dataclass
from gymnasium.spaces import Box, Dict as GymDict, MultiBinary


@dataclass
class Limits:
    lower: NDArray[Any]
    upper: NDArray[Any]


class JointState(GymDict):

    __slots__ = ("dof", "joint_names")

    def __init__(
        self,
        joint_names,
        position_limits: Limits,
        velocity_limits: Limits | None = None,
        effort_limits: Limits | None = None,
        ext_torque_limits: Limits | None = None,
        dtype: type[np.floating[Any]] | type[np.integer[Any]] = np.float32,
        seed: int | np.random.Generator | None = None,
    ):

        if dtype not in (np.float32, np.float64):
            raise TypeError("JointState dtype must be float32 or float64.")

        self.dof = len(joint_names)
        self.joint_names = joint_names
        spaces = {
            "position": Box(
                position_limits.lower,
                position_limits.upper,
                shape=(self.dof,),
                dtype=dtype,
                seed=seed,
            )
        }
        if velocity_limits is not None:
            spaces["velocity"] = Box(
                velocity_limits.lower,
                velocity_limits.upper,
                shape=(self.dof,),
                dtype=dtype,
                seed=seed,
            )
        if effort_limits is not None:
            spaces["effort"] = Box(
                effort_limits.lower,
                effort_limits.upper,
                shape=(self.dof,),
                dtype=dtype,
                seed=seed,
            )
        if ext_torque_limits is not None:
            spaces["ext_torque"] = Box(
                ext_torque_limits.lower,
                ext_torque_limits.upper,
                shape=(self.dof,),
                dtype=dtype,
                seed=seed,
            )
        super().__init__(spaces, seed=seed)

    def __repr__(self) -> str:
        return f"JointState(joint_names={self.joint_names}, spaces={self.spaces})"


class Controller(GymDict):
    """Represents the state of a controller, such as a joystick or gamepad."""

    __slots__ = ("n_axes", "n_buttons")

    def __init__(
        self,
        n_axes: int = 0,
        n_buttons: int = 0,
        axis_dtype: type[np.floating[Any]] = np.float32,
        seed: int | np.random.Generator | None = None,
    ):
        if n_axes < 0:
            raise ValueError("n_axes must be non-negative.")
        if n_buttons < 0:
            raise ValueError("n_buttons must be non-negative.")
        self.n_axes = n_axes
        self.n_buttons = n_buttons
        spaces = {}
        if n_axes > 0:
            if axis_dtype not in (np.float32, np.float64):
                raise TypeError("Joystick axis_dtype must be float32 or float64.")
            spaces["axes"] = Box(
                -1.0, 1.0, shape=(n_axes,), dtype=axis_dtype, seed=seed
            )
        if n_buttons > 0:
            spaces["buttons"] = MultiBinary(n_buttons, seed=seed)
        super().__init__(spaces, seed=seed)

    def __repr__(self):
        inside = ""
        if self.n_axes > 0:
            inside += f"n_axes={self.n_axes}, axis_dtype={self.spaces['axes'].dtype}"
        if self.n_buttons > 0:
            if inside:
                inside += ", "
            inside += f"n_buttons={self.n_buttons}"
        return f"Controller({inside})"
