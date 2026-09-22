"""Keyboard/joystick nominal pose generation combined with admittance.

The missing link between :class:`~replace_disk_robot.control.key_control.KeyControl`
and :class:`~replace_disk_robot.control.admittance.AdmittanceController` is the
nominal Cartesian pose:

* held keys integrate a nominal pose;
* releasing the keys stops the nominal pose but keeps compliance active;
* the admittance controller adds an unbounded offset to that pose.

The class is backend-neutral.  Callers provide an already-transformed external
wrench and the measured TCP pose in the same named frame; IK, collision checks,
workspace limits and joint limits stay in
:class:`~replace_disk_robot.control.servo.CartesianServo`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.ports import AdmittanceControllerPort
from ..core.rotation import (
    multiply,
    quaternion_from_rotation_vector,
    rotation_matrix,
)
from ..core.types import AdmittanceState, CartesianJog, Pose, Wrench


@dataclass(frozen=True)
class MotionReferenceConfig:
    """Axis switches and timing for compliant keyboard motion.

    ``enabled_axes`` is a six-element mask ``[x, y, z, rx, ry, rz]``. Disabled
    wrench axes are zeroed before the admittance controller, which lets the
    first deployment run translation-only compatibility while keeping keyboard
    rotation commands available.  Nominal pose and admittance offset are not
    clamped by this class.
    """

    enabled_axes: ArrayLike = (True,) * 6
    max_dt_s: float = 0.05

    def __post_init__(self) -> None:
        axes = np.asarray(self.enabled_axes, dtype=bool)
        if axes.shape != (6,):
            raise ValueError("enabled_axes must be a boolean 6-vector")
        if not np.isfinite(self.max_dt_s) or self.max_dt_s <= 0.0:
            raise ValueError("max_dt_s must be finite and positive")
        object.__setattr__(self, "enabled_axes", axes.copy())


class KeyboardAdmittanceController:
    """Generate ``nominal -> corrected`` Cartesian poses for a jog input.

    The input ``jog`` follows the historical Servo convention:

    * ``linear_m_s`` in the shared base/output frame;
    * ``angular_rad_s`` intrinsic to the *nominal TCP* axes.

    ``update`` integrates the nominal pose, limits its lead over the measured
    pose, masks disabled admittance axes, and returns the admittance-corrected
    pose that a pose-tracking servo should follow.
    """

    def __init__(
        self,
        admittance: AdmittanceControllerPort,
        initial_pose: Pose,
        config: MotionReferenceConfig | None = None,
    ) -> None:
        missing = [
            name
            for name in ("update", "reset", "state")
            if not hasattr(admittance, name)
        ]
        if missing:
            raise TypeError(
                "admittance must implement update/reset/state; "
                f"missing {missing}"
            )
        if not isinstance(initial_pose, Pose):
            raise TypeError("initial_pose must be a Pose")
        if config is not None and not isinstance(config, MotionReferenceConfig):
            raise TypeError("config must be a MotionReferenceConfig")
        self.admittance = admittance
        self.config = config or MotionReferenceConfig()
        self._frame_id = initial_pose.frame_id
        self._nominal = initial_pose
        self._corrected = initial_pose
        self._enabled_axes = self.config.enabled_axes.copy()
        self.reset(initial_pose)

    @property
    def frame_id(self) -> str:
        return self._frame_id

    @property
    def nominal_pose(self) -> Pose:
        return self._nominal

    @property
    def corrected_pose(self) -> Pose:
        return self._corrected

    @property
    def enabled_axes(self) -> NDArray[np.bool_]:
        return self._enabled_axes.copy()

    def set_enabled_axes(self, axes: ArrayLike) -> None:
        """Replace the runtime admittance axis mask without touching state."""
        array = np.asarray(axes, dtype=bool)
        if array.shape != (6,):
            raise ValueError("enabled_axes must be a boolean 6-vector")
        self._enabled_axes = array.copy()

    def state(self) -> AdmittanceState:
        return self.admittance.state()

    def reset(self, measured_pose: Pose) -> None:
        """Reset nominal and admittance state to the measured pose."""
        self._check_frame(measured_pose, "measured_pose")
        self._nominal = measured_pose
        self._corrected = measured_pose
        self.admittance.reset(measured_pose)

    def update(
        self,
        jog: CartesianJog,
        external_wrench: Wrench,
        measured_pose: Pose,
        dt_s: float,
    ) -> Pose:
        """Return the corrected pose for one control step."""
        if not isinstance(jog, CartesianJog):
            raise TypeError("jog must be a CartesianJog")
        if not isinstance(external_wrench, Wrench):
            raise TypeError("external_wrench must be a Wrench")
        self._check_frame(measured_pose, "measured_pose")
        if jog.base_frame != self._frame_id:
            raise ValueError(
                f"jog frame {jog.base_frame!r} does not match controller frame "
                f"{self._frame_id!r}"
            )
        if external_wrench.frame_id != self._frame_id:
            raise ValueError(
                f"wrench frame {external_wrench.frame_id!r} does not match "
                f"controller frame {self._frame_id!r}"
            )
        if not np.isfinite(dt_s) or dt_s <= 0.0 or dt_s > self.config.max_dt_s:
            raise ValueError(
                f"dt_s must be finite and in (0, {self.config.max_dt_s}]"
            )

        candidate = self._advance(self._nominal, jog, dt_s)
        self._nominal = candidate
        masked_wrench = self._mask_wrench(external_wrench)
        self._corrected = self.admittance.update(candidate, masked_wrench, dt_s)
        return self._corrected

    def _check_frame(self, pose: Pose, name: str) -> None:
        if not isinstance(pose, Pose):
            raise TypeError(f"{name} must be a Pose")
        if pose.frame_id != self._frame_id:
            raise ValueError(
                f"{name} frame {pose.frame_id!r} does not match controller "
                f"frame {self._frame_id!r}"
            )

    @staticmethod
    def _advance(nominal: Pose, jog: CartesianJog, dt_s: float) -> Pose:
        rotation = rotation_matrix(nominal.quaternion_wxyz)
        position = nominal.position_m + np.asarray(jog.linear_m_s) * dt_s
        world_angular_velocity = rotation @ np.asarray(jog.angular_rad_s)
        delta = quaternion_from_rotation_vector(world_angular_velocity * dt_s)
        quaternion = multiply(delta, nominal.quaternion_wxyz)
        quaternion = quaternion / np.linalg.norm(quaternion)
        return Pose(nominal.frame_id, position, quaternion)

    def _mask_wrench(self, wrench: Wrench) -> Wrench:
        if np.all(self._enabled_axes):
            return wrench
        value = wrench.as_vector() * self._enabled_axes
        return Wrench(wrench.frame_id, value[:3], value[3:])
