import numpy as np
from gymnasium import Space
from numpy.typing import NDArray
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as GymDict
from typing import SupportsFloat, Any, Sequence
from scipy.spatial.transform import (
    Rotation as ScipyRotation,
    RigidTransform as ScipyRigidTransform,
)


class Translation(Box):
    __slots__ = ()

    def __init__(
        self,
        low: SupportsFloat | NDArray[Any] = -np.inf,
        high: SupportsFloat | NDArray[Any] = np.inf,
        dtype: type[np.floating[Any]] = np.float32,
        seed: int | np.random.Generator | None = None,
    ):
        dtype = np.dtype(dtype)
        if dtype not in (np.float32, np.float64):
            raise TypeError("Translation dtype must be numpy.float32 or numpy.float64.")
        super().__init__(low=low, high=high, shape=(3,), dtype=dtype, seed=seed)

    def __repr__(self) -> str:
        return f"Translation(low={self.low}, high={self.high}, dtype={self.dtype})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Translation):
            return False
        return self.low == other.low and self.high == other.high


class Rotation(Space[NDArray[Any]]):

    def __init__(
        self,
        dtype: type[np.floating[Any]] = np.float32,
        seed: int | np.random.Generator | None = None,
    ):
        dtype = np.dtype(dtype)
        if dtype not in (np.float32, np.float64):
            raise TypeError("RotationSpace dtype must be float32 or float64.")
        super().__init__(shape=(4,), dtype=dtype, seed=seed)

    @property
    def shape(self) -> tuple[int, ...]:
        return (4,)

    @property
    def is_np_flattenable(self):
        return True

    def sample(self) -> NDArray[Any]:
        r = ScipyRotation.random(random_state=self.np_random).as_quat()
        r: NDArray = r.astype(self.dtype, copy=False)
        return r / np.linalg.norm(r)  # ensure unit quaternion

    def contains(self, x: Any, tolerance: float = 1e-5) -> bool:

        if isinstance(x, ScipyRotation):
            x = x.as_quat()
        elif not isinstance(x, np.ndarray):
            return False

        quat = np.asarray(x, dtype=self.dtype)

        if quat.shape != self.shape or not np.all(np.isfinite(quat)):
            return False

        return abs(np.linalg.norm(quat) - 1.0) <= tolerance

    def to_jsonable(
        self, sample_n: Sequence[NDArray[Any] | ScipyRotation]
    ) -> list[list]:
        return [
            s.as_quat().tolist() if isinstance(s, ScipyRotation) else s.tolist()
            for s in sample_n
        ]

    def from_jsonable(self, sample_n: Sequence[float]) -> list[NDArray[Any]]:
        return [np.asarray(s, dtype=self.dtype) for s in sample_n]

    def __repr__(self) -> str:
        return f"Rotation(dtype={self.dtype})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Rotation)


class RigidTransform(GymDict):
    __slots__ = ("tolerance",)

    def __init__(
        self,
        translation_low: SupportsFloat | NDArray[Any] = -np.inf,
        translation_high: SupportsFloat | NDArray[Any] = np.inf,
        dtype: type[np.floating[Any]] = np.float32,
        seed: int | np.random.Generator | None = None,
    ):

        super().__init__(
            {
                "translation": Translation(
                    translation_low, translation_high, dtype, seed
                ),
                "rotation": Rotation(dtype, seed),
            }
        )

    def contains(
        self,
        x: dict | NDArray | ScipyRigidTransform,
        rot_tolerance: float = 1e-5,
    ) -> bool:
        if isinstance(x, dict):
            try:
                t, r = x["translation"], x["rotation"]
            except KeyError:
                return False
        elif isinstance(x, np.ndarray):
            if x.shape == (7,):
                t, r = x[:3], x[3:]
            elif x.shape == (4, 4):
                t, r = x[:3, 3], ScipyRotation.from_matrix(x[:3, :3]).as_quat()
            else:
                return False
        elif isinstance(x, ScipyRigidTransform):
            t, r = x.translation, x.rotation.as_quat()
        else:
            return False

        contains_t = self["translation"].contains(t)
        contains_r = self["rotation"].contains(r, tolerance=rot_tolerance)

        return contains_t and contains_r

    def sample(self) -> dict[str, NDArray[Any]]:
        return {
            "translation": self["translation"].sample(),
            "rotation": self["rotation"].sample(),
        }

    def __repr__(self) -> str:
        return f"RigidTransform(translation={self['translation']}, rotation={self['rotation']})"
