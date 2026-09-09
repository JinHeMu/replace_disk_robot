"""Small, explicit data contracts shared by algorithms and adapters."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


Vector = NDArray[np.float64]


def _vector(value: ArrayLike, size: int, name: str) -> Vector:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    return array.copy()


@dataclass(frozen=True)
class Pose:
    """Rigid pose with explicit parent frame and wxyz quaternion."""

    frame_id: str
    position_m: Vector
    quaternion_wxyz: Vector

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        position = _vector(self.position_m, 3, "position_m")
        quaternion = _vector(self.quaternion_wxyz, 4, "quaternion_wxyz")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-12:
            raise ValueError("quaternion_wxyz must have non-zero norm")
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "quaternion_wxyz", quaternion / norm)


@dataclass(frozen=True)
class Wrench:
    """Force and torque expressed in one named frame."""

    frame_id: str
    force_n: Vector
    torque_nm: Vector

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        object.__setattr__(self, "force_n", _vector(self.force_n, 3, "force_n"))
        object.__setattr__(self, "torque_nm", _vector(self.torque_nm, 3, "torque_nm"))

    def as_vector(self) -> Vector:
        return np.concatenate((self.force_n, self.torque_nm))


@dataclass(frozen=True)
class JointState:
    """Named joint state; order is part of the contract."""

    names: tuple[str, ...]
    position_rad: Vector

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("joint names must not be empty")
        object.__setattr__(
            self, "position_rad", _vector(self.position_rad, len(self.names), "position_rad")
        )


@dataclass(frozen=True)
class TrajectoryPoint:
    """A planner output point; no interpolation semantics are hidden here."""

    time_from_start_s: float
    position_rad: Vector

    def __post_init__(self) -> None:
        if not np.isfinite(self.time_from_start_s) or self.time_from_start_s < 0:
            raise ValueError("time_from_start_s must be finite and non-negative")
        array = np.asarray(self.position_rad, dtype=float)
        if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError("position_rad must be a non-empty finite vector")
        object.__setattr__(self, "position_rad", array.copy())


@dataclass(frozen=True)
class CartesianJog:
    """Linear velocity in base_frame; intrinsic angular velocity in current TCP axes.

    TCP X points forward, Y points left, Z points up at this scene's init.
    The explicit mixed convention keeps vertical translation upright while roll
    always follows the tool. Units are m/s and rad/s, never displacement per key.
    """
    base_frame: str
    linear_m_s: np.ndarray
    angular_rad_s: np.ndarray

    def __post_init__(self):
        if not self.base_frame:
            raise ValueError('base_frame must not be empty')
        object.__setattr__(self, 'linear_m_s', _vector(self.linear_m_s, 3, 'linear_m_s'))
        object.__setattr__(self, 'angular_rad_s', _vector(self.angular_rad_s, 3, 'angular_rad_s'))

    @classmethod
    def zero(cls, base_frame='world'):
        return cls(base_frame, np.zeros(3), np.zeros(3))
