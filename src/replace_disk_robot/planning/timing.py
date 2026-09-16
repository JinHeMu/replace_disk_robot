"""Time allocation and interpolation for joint-space trajectories.

P2 uses a conservative first version:

* each pair of adjacent waypoints becomes one quintic segment;
* each segment starts and ends with zero joint velocity and acceleration;
* segment duration is chosen from joint velocity and acceleration limits.

The resulting :class:`TimedTrajectory` can be sampled at arbitrary times, which
means collision and limit checks can be performed on the actual curve that the
executor will follow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core import CollisionCheckerPort, JointState, TrajectoryPoint
from .collision import is_path_collision_free


_QUINTIC_PEAK_VELOCITY = 1.875
_QUINTIC_PEAK_ACCELERATION = 5.773502691896258


def _limit_vector(value: ArrayLike, n_joints: int, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(n_joints, float(array))
    if array.shape != (n_joints,):
        raise ValueError(f"{name} must have shape ({n_joints},) or be a scalar")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    return array.copy()


@dataclass(frozen=True)
class TimedTrajectory:
    """Piecewise quintic joint trajectory with zero boundary derivatives."""

    joint_names: tuple[str, ...]
    times_s: NDArray[np.float64]
    positions_rad: NDArray[np.float64]
    velocity_limits_rad_s: NDArray[np.float64] | None = None
    acceleration_limits_rad_s2: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=float)
        positions = np.asarray(self.positions_rad, dtype=float)
        if positions.ndim != 2 or positions.shape[0] < 2:
            raise ValueError("positions_rad must have shape (N, n_joints), N >= 2")
        if times.shape != (positions.shape[0],):
            raise ValueError("times_s must have shape (N,)")
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            raise ValueError("times_s must be finite and strictly increasing")
        if positions.shape[1] != len(self.joint_names):
            raise ValueError("positions_rad column count must match joint_names")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions_rad must be finite")

        object.__setattr__(self, "times_s", times.copy())
        object.__setattr__(self, "positions_rad", positions.copy())
        if self.velocity_limits_rad_s is not None:
            object.__setattr__(
                self,
                "velocity_limits_rad_s",
                _limit_vector(
                    self.velocity_limits_rad_s,
                    positions.shape[1],
                    "velocity_limits_rad_s",
                ),
            )
        if self.acceleration_limits_rad_s2 is not None:
            object.__setattr__(
                self,
                "acceleration_limits_rad_s2",
                _limit_vector(
                    self.acceleration_limits_rad_s2,
                    positions.shape[1],
                    "acceleration_limits_rad_s2",
                ),
            )

    @property
    def duration_s(self) -> float:
        return float(self.times_s[-1] - self.times_s[0])

    @property
    def start_time_s(self) -> float:
        return float(self.times_s[0])

    def _segment(self, t: float) -> tuple[int, float]:
        n_segments = self.positions_rad.shape[0] - 1
        if t <= self.times_s[0]:
            return 0, 0.0
        if t >= self.times_s[-1]:
            return n_segments - 1, 1.0
        index = int(np.searchsorted(self.times_s, t, side="right") - 1)
        index = min(max(index, 0), n_segments - 1)
        segment_dt = float(self.times_s[index + 1] - self.times_s[index])
        return index, float((t - self.times_s[index]) / segment_dt)

    def sample(self, t: float) -> NDArray[np.float64]:
        index, s = self._segment(float(t))
        s = float(np.clip(s, 0.0, 1.0))
        q0 = self.positions_rad[index]
        q1 = self.positions_rad[index + 1]
        blend = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        return q0 + (q1 - q0) * blend

    def sample_joint_state(self, t: float) -> JointState:
        return JointState(self.joint_names, self.sample(t))

    def sample_velocity(self, t: float) -> NDArray[np.float64]:
        index, s = self._segment(float(t))
        segment_dt = float(self.times_s[index + 1] - self.times_s[index])
        d_blend = 30.0 * s**2 * (1.0 - s) ** 2
        return (self.positions_rad[index + 1] - self.positions_rad[index]) * (
            d_blend / segment_dt
        )

    def sample_acceleration(self, t: float) -> NDArray[np.float64]:
        index, s = self._segment(float(t))
        segment_dt_sq = float(
            (self.times_s[index + 1] - self.times_s[index]) ** 2
        )
        dd_blend = 60.0 * s - 180.0 * s**2 + 120.0 * s**3
        return (self.positions_rad[index + 1] - self.positions_rad[index]) * (
            dd_blend / segment_dt_sq
        )


def assign_quintic_timing(
    waypoints: Sequence[TrajectoryPoint],
    joint_names: tuple[str, ...],
    *,
    velocity_limits_rad_s: ArrayLike,
    acceleration_limits_rad_s2: ArrayLike,
    min_segment_time_s: float = 0.05,
    max_segment_time_s: float | None = None,
) -> TimedTrajectory:
    """Assign per-segment quintic durations from joint limits.

    A segment between ``q0`` and ``q1`` uses the smooth quintic blend whose
    peak velocity is ``1.875 * |q1 - q0| / T`` and whose peak acceleration is
    ``5.7735 * |q1 - q0| / T**2``.  Each segment is therefore stopped at both
    ends.
    """
    if len(waypoints) < 2:
        raise ValueError("need at least two waypoints to assign timing")
    if not joint_names:
        raise ValueError("joint_names must not be empty")
    if min_segment_time_s <= 0.0:
        raise ValueError("min_segment_time_s must be positive")
    if max_segment_time_s is not None and max_segment_time_s < min_segment_time_s:
        raise ValueError("max_segment_time_s must be >= min_segment_time_s")

    positions = np.array([point.position_rad for point in waypoints], dtype=float)
    if positions.ndim != 2 or positions.shape[1] != len(joint_names):
        raise ValueError("waypoint joint count must match joint_names")
    if not np.all(np.isfinite(positions)):
        raise ValueError("waypoints must contain finite joint values")

    velocity_limits = _limit_vector(
        velocity_limits_rad_s,
        positions.shape[1],
        "velocity_limits_rad_s",
    )
    acceleration_limits = _limit_vector(
        acceleration_limits_rad_s2,
        positions.shape[1],
        "acceleration_limits_rad_s2",
    )

    times = [0.0]
    for q0, q1 in zip(positions[:-1], positions[1:]):
        delta = np.abs(q1 - q0)
        segment_time = float(min_segment_time_s)
        active = delta > 1e-12
        if np.any(active):
            time_from_velocity = _QUINTIC_PEAK_VELOCITY * delta / velocity_limits
            time_from_acceleration = np.sqrt(
                _QUINTIC_PEAK_ACCELERATION * delta / acceleration_limits
            )
            segment_time = max(
                segment_time,
                float(np.max(time_from_velocity[active])),
                float(np.max(time_from_acceleration[active])),
            )
        if max_segment_time_s is not None and segment_time > max_segment_time_s:
            raise ValueError(
                "joint limits require a longer segment than max_segment_time_s"
            )
        times.append(times[-1] + segment_time)

    return TimedTrajectory(
        joint_names=tuple(joint_names),
        times_s=np.asarray(times, dtype=float),
        positions_rad=positions,
        velocity_limits_rad_s=velocity_limits,
        acceleration_limits_rad_s2=acceleration_limits,
    )


def validate_timed_trajectory(
    trajectory: TimedTrajectory,
    *,
    collision_checker: CollisionCheckerPort | None = None,
    joint_lower_limits_rad: ArrayLike | None = None,
    joint_upper_limits_rad: ArrayLike | None = None,
    samples_per_segment: int = 40,
) -> dict[str, float]:
    """Validate joint limits, velocity/acceleration limits and the actual curve.

    Returns a dict with maximum observed velocity/acceleration and minimum
    clearance.  Raises ``ValueError`` on a hard violation.
    """
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least 2")

    lower = (
        None
        if joint_lower_limits_rad is None
        else np.asarray(joint_lower_limits_rad, dtype=float)
    )
    upper = (
        None
        if joint_upper_limits_rad is None
        else np.asarray(joint_upper_limits_rad, dtype=float)
    )
    if (lower is None) != (upper is None):
        raise ValueError("joint lower/upper limits must be provided together")
    n_joints = trajectory.positions_rad.shape[1]
    if lower is not None and (lower.shape != (n_joints,) or upper.shape != (n_joints,)):
        raise ValueError("joint limits must match trajectory joint count")

    max_velocity = 0.0
    max_acceleration = 0.0
    min_clearance = float("inf")
    previous_state: JointState | None = None

    for segment_index in range(len(trajectory.times_s) - 1):
        t0 = float(trajectory.times_s[segment_index])
        t1 = float(trajectory.times_s[segment_index + 1])
        for t in np.linspace(t0, t1, samples_per_segment + 1):
            t_float = float(t)
            state = trajectory.sample_joint_state(t_float)
            if lower is not None:
                if np.any(state.position_rad < lower) or np.any(
                    state.position_rad > upper
                ):
                    raise ValueError(
                        "timed trajectory violates configured joint limits"
                    )

            velocity = trajectory.sample_velocity(t_float)
            acceleration = trajectory.sample_acceleration(t_float)
            max_velocity = max(max_velocity, float(np.max(np.abs(velocity))))
            max_acceleration = max(
                max_acceleration,
                float(np.max(np.abs(acceleration))),
            )
            if trajectory.velocity_limits_rad_s is not None and np.any(
                np.abs(velocity) > trajectory.velocity_limits_rad_s + 1e-9
            ):
                raise ValueError("timed trajectory exceeds velocity limits")
            if trajectory.acceleration_limits_rad_s2 is not None and np.any(
                np.abs(acceleration) > trajectory.acceleration_limits_rad_s2 + 1e-9
            ):
                raise ValueError("timed trajectory exceeds acceleration limits")

            if collision_checker is not None:
                if not collision_checker.is_collision_free(state):
                    raise ValueError("timed trajectory is in collision")
                min_clearance = min(
                    min_clearance,
                    float(collision_checker.minimum_distance(state)),
                )
                if previous_state is not None:
                    if not is_path_collision_free(
                        collision_checker,
                        previous_state,
                        state,
                        samples=2,
                    ):
                        raise ValueError("timed trajectory segment is in collision")
            previous_state = state

    return {
        "max_velocity_rad_s": max_velocity,
        "max_acceleration_rad_s2": max_acceleration,
        "min_clearance_m": min_clearance,
    }
