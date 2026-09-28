"""Frame transforms, filtering and deadbands for measured wrenches.

``AdmittanceController`` deliberately requires a wrench expressed in a known
frame with the sign of an *external load*.  Raw sensors usually do neither:

* a wrist F/T sensor reports the load it applies to its child link, which is
  the opposite of the external load acting on the tool;
* the sensor frame is neither the robot base nor the controlled TCP;
* wrench readings are noisy and have a small offset drift.

This module keeps that plumbing outside the admittance model.  It contains no
MuJoCo, ROS or hardware import so it can be unit-tested with plain arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..core.rotation import rotation_matrix
from ..core.types import Pose, Wrench


def _non_negative_scalar(value: ArrayLike, name: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim != 0 or not np.isfinite(array) or float(array) < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return float(array)


def _vector(value: ArrayLike, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {size}-vector")
    return array.copy()


@dataclass(frozen=True)
class SensorCompensationConfig:
    """Raw sensor calibration with gravity expressed in the pose parent frame."""

    frame_id: str
    load_sign: float = -1.0
    gravity_frame_id: str = "world"
    gravity_m_s2: ArrayLike = (0.0, 0.0, -9.80665)
    payload_mass_kg: float = 0.0
    payload_com_sensor_m: ArrayLike = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        if self.load_sign not in (-1.0, 1.0):
            raise ValueError("load_sign must be +1 or -1")
        if not self.gravity_frame_id:
            raise ValueError("gravity_frame_id must not be empty")
        object.__setattr__(self, "gravity_m_s2", _vector(self.gravity_m_s2, 3, "gravity_m_s2"))
        object.__setattr__(self, "payload_mass_kg", _non_negative_scalar(self.payload_mass_kg, "payload_mass_kg"))
        object.__setattr__(self, "payload_com_sensor_m", _vector(self.payload_com_sensor_m, 3, "payload_com_sensor_m"))


class SensorWrenchCompensator:
    """Subtract sensor bias and predicted payload gravity in sensor axes.

    Tare requires no contact and separates static bias from the gravity
    predicted at the tare pose. A payload change requires another tare.
    """

    def __init__(self, config: SensorCompensationConfig) -> None:
        if not isinstance(config, SensorCompensationConfig):
            raise TypeError("config must be SensorCompensationConfig")
        self.config = config
        self.bias_sensor = np.zeros(6)

    def _gravity(self, sensor_pose: Pose, mass: float, com: ArrayLike) -> np.ndarray:
        gravity_sensor = rotation_matrix(sensor_pose.quaternion_wxyz).T @ self.config.gravity_m_s2
        force = self.config.load_sign * mass * gravity_sensor
        return np.r_[force, np.cross(_vector(com, 3, "payload_com_sensor_m"), force)]

    def _inputs(self, wrench: Wrench, sensor_pose: Pose, payload_mass_kg, payload_com_sensor_m):
        if wrench.frame_id != self.config.frame_id:
            raise ValueError(f"wrench frame {wrench.frame_id!r} must be {self.config.frame_id!r}")
        if sensor_pose.frame_id != self.config.gravity_frame_id:
            raise ValueError(
                f"sensor pose frame {sensor_pose.frame_id!r} must be "
                f"{self.config.gravity_frame_id!r}"
            )
        mass = self.config.payload_mass_kg if payload_mass_kg is None else _non_negative_scalar(payload_mass_kg, "payload_mass_kg")
        com = self.config.payload_com_sensor_m if payload_com_sensor_m is None else payload_com_sensor_m
        return self._gravity(sensor_pose, mass, com)

    def tare(self, wrench: Wrench, sensor_pose: Pose, *, payload_mass_kg=None, payload_com_sensor_m=None) -> None:
        predicted = self._inputs(wrench, sensor_pose, payload_mass_kg, payload_com_sensor_m)
        self.bias_sensor = wrench.as_vector() - predicted

    def compensate(self, wrench: Wrench, sensor_pose: Pose, *, payload_mass_kg=None, payload_com_sensor_m=None) -> Wrench:
        predicted = self._inputs(wrench, sensor_pose, payload_mass_kg, payload_com_sensor_m)
        corrected = wrench.as_vector() - self.bias_sensor - predicted
        return Wrench(self.config.frame_id, corrected[:3], corrected[3:])


@dataclass(frozen=True)
class WrenchProcessorConfig:
    """Configuration for :class:`WrenchProcessor`.

    ``load_sign`` converts a sensor convention into an external load:

    * ``+1``: the sensor already reads the external load acting on the tool;
    * ``-1``: the sensor reads the parent-on-child/wrist load, as MuJoCo's
      ``force``/``torque`` sensors do.

    Filtering is a first-order low-pass in the output frame.  Set
    ``filter_alpha=1.0`` to disable it.  Deadbands are applied per component
    after filtering; zero disables a deadband.
    """

    frame_id: str
    load_sign: float = -1.0
    filter_alpha: float = 0.2
    force_deadband_n: ArrayLike = 0.0
    torque_deadband_nm: ArrayLike = 0.0

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        if not np.isfinite(self.load_sign) or self.load_sign not in (-1.0, 1.0):
            raise ValueError("load_sign must be either +1 or -1")
        if not np.isfinite(self.filter_alpha) or not 0.0 < self.filter_alpha <= 1.0:
            raise ValueError("filter_alpha must be finite and in (0, 1]")
        object.__setattr__(
            self,
            "force_deadband_n",
            _non_negative_scalar(self.force_deadband_n, "force_deadband_n"),
        )
        object.__setattr__(
            self,
            "torque_deadband_nm",
            _non_negative_scalar(self.torque_deadband_nm, "torque_deadband_nm"),
        )


def external_wrench_at_point(
    wrench: Wrench,
    sensor_pose_in_frame: Pose,
    tcp_position_in_frame: ArrayLike,
    *,
    load_sign: float = -1.0,
) -> Wrench:
    """Express ``wrench`` as an external load about a TCP in one output frame.

    ``sensor_pose_in_frame`` is the pose of the sensor origin/orientation and
    ``tcp_position_in_frame`` the point about which the equivalent wrench should
    act, both in the same output frame.  The returned wrench uses the output
    frame of ``sensor_pose_in_frame``.

    The moment shift follows from static equilibrium::

        f_ext     = sign * R f_sensor
        tau_ext   = sign * (R tau_sensor + (p_sensor - p_tcp) x R f_sensor)
    """

    if not isinstance(wrench, Wrench):
        raise TypeError("wrench must be a Wrench")
    if not isinstance(sensor_pose_in_frame, Pose):
        raise TypeError("sensor_pose_in_frame must be a Pose")
    if not np.isfinite(load_sign) or load_sign not in (-1.0, 1.0):
        raise ValueError("load_sign must be either +1 or -1")

    point = np.asarray(tcp_position_in_frame, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("tcp_position_in_frame must be a finite 3-vector")

    rotation = rotation_matrix(sensor_pose_in_frame.quaternion_wxyz)
    force = rotation @ wrench.force_n
    torque_about_sensor = rotation @ wrench.torque_nm
    lever = sensor_pose_in_frame.position_m - point
    torque = torque_about_sensor + np.cross(lever, force)
    return Wrench(
        sensor_pose_in_frame.frame_id,
        load_sign * force,
        load_sign * torque,
    )


def external_wrench_at_tcp(
    wrench: Wrench,
    sensor_pose_in_base: Pose,
    tcp_pose_in_base: Pose,
    tcp_frame_id: str,
    *,
    load_sign: float = -1.0,
) -> Wrench:
    """Shift moment to TCP origin and express force and moment in TCP axes."""
    if sensor_pose_in_base.frame_id != tcp_pose_in_base.frame_id:
        raise ValueError("sensor and TCP poses must have the same parent frame")
    if not tcp_frame_id:
        raise ValueError("tcp_frame_id must not be empty")
    in_base = external_wrench_at_point(
        wrench, sensor_pose_in_base, tcp_pose_in_base.position_m,
        load_sign=load_sign,
    )
    tcp_from_base = rotation_matrix(tcp_pose_in_base.quaternion_wxyz).T
    return Wrench(
        tcp_frame_id,
        tcp_from_base @ in_base.force_n,
        tcp_from_base @ in_base.torque_nm,
    )


class WrenchProcessor:
    """Stateful external-load filter in one explicit output frame."""

    def __init__(self, config: WrenchProcessorConfig) -> None:
        if not isinstance(config, WrenchProcessorConfig):
            raise TypeError("config must be a WrenchProcessorConfig")
        self.config = config
        self._filtered = np.zeros(6)
        self._previous_tcp_pose: Pose | None = None

    def reset(self) -> None:
        """Clear the low-pass state.  The next update seeds the filter."""
        self._filtered[:] = 0.0
        self._previous_tcp_pose = None

    def state(self) -> Wrench:
        """Return the last filtered external wrench."""
        return Wrench(
            self.config.frame_id,
            self._filtered[:3].copy(),
            self._filtered[3:].copy(),
        )

    def update(
        self,
        wrench: Wrench,
        sensor_pose_in_frame: Pose,
        tcp_position_in_frame: ArrayLike,
    ) -> Wrench:
        """Transform, filter and deadband one sensor reading."""
        if sensor_pose_in_frame.frame_id != self.config.frame_id:
            raise ValueError(
                f"sensor pose frame {sensor_pose_in_frame.frame_id!r} does not "
                f"match processor frame {self.config.frame_id!r}"
            )
        external = external_wrench_at_point(
            wrench,
            sensor_pose_in_frame,
            tcp_position_in_frame,
            load_sign=self.config.load_sign,
        )
        self._previous_tcp_pose = None
        return self._filter(external)

    def update_tcp(self, wrench: Wrench, sensor_pose_in_base: Pose, tcp_pose_in_base: Pose) -> Wrench:
        """Transform to TCP, then filter and apply deadbands in TCP axes."""
        external = external_wrench_at_tcp(
            wrench, sensor_pose_in_base, tcp_pose_in_base,
            self.config.frame_id, load_sign=self.config.load_sign,
        )
        return self._filter(external, tcp_pose_in_base)

    def _filter(self, external: Wrench, tcp_pose: Pose | None = None) -> Wrench:
        value = external.as_vector()
        previous = self._filtered.copy()
        if tcp_pose is not None and self._previous_tcp_pose is not None:
            old = self._previous_tcp_pose
            if old.frame_id != tcp_pose.frame_id:
                raise ValueError("TCP parent frame changed; reset the wrench filter")
            base_from_old = rotation_matrix(old.quaternion_wxyz)
            new_from_base = rotation_matrix(tcp_pose.quaternion_wxyz).T
            force_base = base_from_old @ previous[:3]
            moment_base = base_from_old @ previous[3:]
            moment_base += np.cross(old.position_m - tcp_pose.position_m, force_base)
            previous = np.r_[new_from_base @ force_base, new_from_base @ moment_base]
        alpha = float(self.config.filter_alpha)
        self._filtered = alpha * value + (1.0 - alpha) * previous
        self._previous_tcp_pose = tcp_pose
        output = self._filtered.copy()
        force_deadband = float(self.config.force_deadband_n)
        torque_deadband = float(self.config.torque_deadband_nm)
        if force_deadband > 0.0:
            output[:3][np.abs(output[:3]) < force_deadband] = 0.0
        if torque_deadband > 0.0:
            output[3:][np.abs(output[3:]) < torque_deadband] = 0.0
        return Wrench(self.config.frame_id, output[:3], output[3:])
