"""Backend-neutral six-axis admittance control.

The controller turns a measured external wrench into a compliant Cartesian
pose offset. It is deliberately independent of MuJoCo, ROS, IK and hardware:
callers must supply both the nominal pose and wrench in one explicit frame,
and remain responsible for frame transforms, IK, collision checks, limits and
the final command output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.rotation import multiply as quat_multiply
from ..core.types import AdmittanceState, Pose, Wrench


def _six(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return a finite six-vector, broadcasting a scalar to all components."""
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(6, float(array))
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite scalar or vector with shape (6,)")
    return array.copy()


def _quaternion_from_rotation_vector(rotation_vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a wxyz quaternion for an axis-angle vector in radians."""
    theta = float(np.linalg.norm(rotation_vector))
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotation_vector / theta
    half_angle = 0.5 * theta
    return np.r_[np.cos(half_angle), axis * np.sin(half_angle)]


@dataclass(frozen=True)
class AdmittanceConfig:
    """Diagonal six-axis admittance parameters in one named frame.

    ``mass``, ``damping``, ``stiffness``, ``max_offset`` and
    ``max_velocity`` may be scalars or six-vectors. Components 0..2 are
    translational; components 3..5 are rotational. SI units are expected:

    - mass: kg for translation, kg*m^2 for rotation
    - damping: N*s/m for translation, N*m*s/rad for rotation
    - stiffness: N/m for translation, N*m/rad for rotation
    - offset: m for translation, rad for rotation
    - velocity: m/s for translation, rad/s for rotation

    The offset is expressed in ``frame_id`` and the returned corrected pose
    uses the same frame.
    """

    frame_id: str
    mass: ArrayLike
    damping: ArrayLike
    stiffness: ArrayLike
    max_offset: ArrayLike
    max_velocity: ArrayLike
    max_dt_s: float = 0.05

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        mass = _six(self.mass, "mass")
        damping = _six(self.damping, "damping")
        stiffness = _six(self.stiffness, "stiffness")
        max_offset = _six(self.max_offset, "max_offset")
        max_velocity = _six(self.max_velocity, "max_velocity")
        if np.any(mass <= 0):
            raise ValueError("mass must be positive")
        if np.any(damping <= 0):
            raise ValueError("damping must be positive")
        if np.any(stiffness < 0):
            raise ValueError("stiffness must be non-negative")
        if np.any(max_offset <= 0):
            raise ValueError("max_offset must be positive")
        if np.any(max_velocity <= 0):
            raise ValueError("max_velocity must be positive")
        if not np.isfinite(self.max_dt_s) or self.max_dt_s <= 0:
            raise ValueError("max_dt_s must be finite and positive")
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "stiffness", stiffness)
        object.__setattr__(self, "max_offset", max_offset)
        object.__setattr__(self, "max_velocity", max_velocity)


class AdmittanceController:
    """Stateful diagonal admittance controller.

    The discrete update is semi-implicit Euler::

        a = M^-1 (F_ext - D v - K x)
        v_next = v + a dt
        x_next = x + v_next dt

    where ``x`` is the six-axis offset and ``v`` its velocity. ``F_ext`` is
    the external wrench supplied by the caller in the controller frame. The
    returned pose is ``nominal`` plus ``x``; it is not a joint command.
    """

    def __init__(self, config: AdmittanceConfig) -> None:
        if not isinstance(config, AdmittanceConfig):
            raise TypeError("config must be an AdmittanceConfig")
        self.config = config
        self._offset = np.zeros(6)
        self._velocity = np.zeros(6)
        self._initialized = False

    def _check_pose(self, nominal: Pose) -> None:
        if not isinstance(nominal, Pose):
            raise TypeError("nominal must be a Pose")
        if nominal.frame_id != self.config.frame_id:
            raise ValueError(
                f"nominal frame {nominal.frame_id!r} does not match "
                f"controller frame {self.config.frame_id!r}"
            )

    def _check_wrench(self, wrench: Wrench) -> None:
        if not isinstance(wrench, Wrench):
            raise TypeError("wrench must be a Wrench")
        if wrench.frame_id != self.config.frame_id:
            raise ValueError(
                f"wrench frame {wrench.frame_id!r} does not match "
                f"controller frame {self.config.frame_id!r}"
            )

    def reset(self, nominal: Pose) -> None:
        """Reset offset and velocity to zero for ``nominal``."""
        self._check_pose(nominal)
        self._offset = np.zeros(6)
        self._velocity = np.zeros(6)
        self._initialized = True

    def state(self) -> AdmittanceState:
        """Return a copy of the current offset and velocity."""
        return AdmittanceState(
            self.config.frame_id,
            self._offset.copy(),
            self._velocity.copy(),
        )

    def update(self, nominal: Pose, wrench: Wrench, dt_s: float) -> Pose:
        """Integrate one step and return the corrected Cartesian pose.

        ``nominal`` and ``wrench`` must already be expressed in the configured
        frame. ``wrench`` is the external load that should move the tool in
        the positive offset direction; callers using a sensor whose sign
        convention differs must negate it before calling this method.
        """
        self._check_pose(nominal)
        self._check_wrench(wrench)
        if not np.isfinite(dt_s) or dt_s <= 0 or dt_s > self.config.max_dt_s:
            raise ValueError(
                f"dt_s must be finite and in (0, {self.config.max_dt_s}]"
            )
        if not self._initialized:
            self.reset(nominal)

        wrench_vector = wrench.as_vector()
        acceleration = (
            wrench_vector
            - self.config.damping * self._velocity
            - self.config.stiffness * self._offset
        ) / self.config.mass

        # Limit the velocity *before* integrating it into the offset; otherwise
        # one step may still move farther than max_velocity * dt_s.
        velocity = np.clip(
            self._velocity + acceleration * dt_s,
            -self.config.max_velocity,
            self.config.max_velocity,
        )
        offset = self._offset + velocity * dt_s
        clipped_offset = np.clip(
            offset,
            -self.config.max_offset,
            self.config.max_offset,
        )

        # Anti-windup: do not keep integrating outward once a component hits
        # its offset limit. Inward velocity is preserved so the tool can return.
        outward = (
            ((offset > self.config.max_offset) & (velocity > 0))
            | ((offset < -self.config.max_offset) & (velocity < 0))
        )
        velocity = np.where(outward, 0.0, velocity)

        self._offset = clipped_offset
        self._velocity = velocity

        return self._pose_from_offset(nominal, clipped_offset)

    @staticmethod
    def _pose_from_offset(nominal: Pose, offset: NDArray[np.float64]) -> Pose:
        position = nominal.position_m + offset[:3]
        delta_quaternion = _quaternion_from_rotation_vector(offset[3:])
        quaternion = quat_multiply(delta_quaternion, nominal.quaternion_wxyz)
        quaternion /= np.linalg.norm(quaternion)
        return Pose(nominal.frame_id, position, quaternion)
