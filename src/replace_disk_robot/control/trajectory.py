"""Time-parameterized trajectory executor for :class:`ArmPort`.

The executor follows a :class:`~replace_disk_robot.planning.timing.TimedTrajectory`
by sampling the actual quintic curve, sending joint-position references at the
caller's control rate, and finishing only after the measured joints have stayed
inside the goal tolerance for a configured hold time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..core import ArmPort, JointState
from ..planning.timing import TimedTrajectory


class TrajectoryExecutionError(RuntimeError):
    """Base class for trajectory executor failures."""


class TrajectoryStartError(TrajectoryExecutionError):
    """The measured start state is too far from the planned start state."""


class ExecutionState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass(frozen=True)
class ExecutionStatus:
    state: ExecutionState
    elapsed_s: float
    trajectory_duration_s: float | None
    tracking_error_rad: float
    goal_error_rad: float
    failure_reason: str | None = None
    started_at_s: float | None = None
    finished_at_s: float | None = None


class TrajectoryExecutor:
    """Execute a timed joint trajectory through an ``ArmPort``.

    Parameters
    ----------
    arm:
        Hardware or simulation adapter used for joint feedback/commands.
    start_tolerance_rad:
        Maximum per-joint deviation between measured state and the planned
        start state accepted by ``start``.
    tracking_tolerance_rad:
        Maximum per-joint deviation between measured state and the current
        reference while running.  Exceeding this threshold fails the execution.
    goal_tolerance_rad:
        Per-joint tolerance used for final arrival.
    goal_hold_s:
        Measured joints must remain inside ``goal_tolerance_rad`` for this long.
    timeout_s:
        Optional extra time after the trajectory duration before a missing
        arrival is declared a failure.
    """

    def __init__(
        self,
        arm: ArmPort,
        *,
        start_tolerance_rad: float = 0.1,
        tracking_tolerance_rad: float = 0.5,
        goal_tolerance_rad: float = 0.02,
        goal_hold_s: float = 0.2,
        timeout_s: float | None = None,
    ) -> None:
        for name, value in {
            "start_tolerance_rad": start_tolerance_rad,
            "tracking_tolerance_rad": tracking_tolerance_rad,
            "goal_tolerance_rad": goal_tolerance_rad,
            "goal_hold_s": goal_hold_s,
        }.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if timeout_s is not None and (not np.isfinite(timeout_s) or timeout_s <= 0.0):
            raise ValueError("timeout_s must be finite and positive when provided")

        self.arm = arm
        self.start_tolerance_rad = float(start_tolerance_rad)
        self.tracking_tolerance_rad = float(tracking_tolerance_rad)
        self.goal_tolerance_rad = float(goal_tolerance_rad)
        self.goal_hold_s = float(goal_hold_s)
        self.timeout_s = None if timeout_s is None else float(timeout_s)

        self._state = ExecutionState.IDLE
        self._trajectory: TimedTrajectory | None = None
        self._start_time_s: float | None = None
        self._last_update_s: float | None = None
        self._elapsed_s = 0.0
        self._finished_at_s: float | None = None
        self._tracking_error_rad = 0.0
        self._goal_error_rad = float("inf")
        self._hold_since_s: float | None = None
        self._failure_reason: str | None = None

    @property
    def state(self) -> ExecutionState:
        return self._state

    @property
    def trajectory(self) -> TimedTrajectory | None:
        return self._trajectory

    @property
    def tracking_error_rad(self) -> float:
        return self._tracking_error_rad

    @property
    def goal_error_rad(self) -> float:
        return self._goal_error_rad

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    @property
    def status(self) -> ExecutionStatus:
        return ExecutionStatus(
            state=self._state,
            elapsed_s=self._elapsed_s,
            trajectory_duration_s=(
                None if self._trajectory is None else self._trajectory.duration_s
            ),
            tracking_error_rad=self._tracking_error_rad,
            goal_error_rad=self._goal_error_rad,
            failure_reason=self._failure_reason,
            started_at_s=self._start_time_s,
            finished_at_s=self._finished_at_s,
        )

    def start(
        self,
        trajectory: TimedTrajectory,
        now_s: float,
        *,
        planned_start: JointState | None = None,
    ) -> ExecutionStatus:
        """Read feedback, verify the start state and begin execution."""
        if not np.isfinite(now_s):
            raise ValueError("now_s must be finite")
        if self._state == ExecutionState.RUNNING:
            raise TrajectoryExecutionError("executor is already running")

        actual = self.arm.read_joint_state()
        if actual.names != trajectory.joint_names:
            raise TrajectoryStartError(
                f"arm joint names {actual.names} do not match trajectory "
                f"joint names {trajectory.joint_names}"
            )
        if planned_start is not None:
            if planned_start.names != trajectory.joint_names:
                raise TrajectoryStartError(
                    "planned_start joint names do not match trajectory joint names"
                )
            expected_start = planned_start
        else:
            expected_start = JointState(
                trajectory.joint_names,
                trajectory.positions_rad[0],
            )

        start_error = float(
            np.max(np.abs(actual.position_rad - expected_start.position_rad))
        )
        if start_error > self.start_tolerance_rad:
            raise TrajectoryStartError(
                "measured start state is too far from the planned start: "
                f"max joint error={start_error:.6f} rad, "
                f"tolerance={self.start_tolerance_rad:.6f} rad"
            )

        self._trajectory = trajectory
        self._start_time_s = float(now_s)
        self._last_update_s = float(now_s)
        self._elapsed_s = 0.0
        self._finished_at_s = None
        self._tracking_error_rad = 0.0
        self._goal_error_rad = float("inf")
        self._hold_since_s = None
        self._failure_reason = None
        self._state = ExecutionState.RUNNING
        return self.status

    def update(self, now_s: float) -> ExecutionStatus:
        """Send one reference command and update tracking/completion state."""
        if not np.isfinite(now_s):
            raise ValueError("now_s must be finite")
        if self._state != ExecutionState.RUNNING or self._trajectory is None:
            return self.status

        assert self._start_time_s is not None
        elapsed = float(now_s) - self._start_time_s
        if elapsed < 0.0:
            raise TrajectoryExecutionError("now_s is earlier than execution start")
        self._elapsed_s = elapsed
        self._last_update_s = float(now_s)

        trajectory = self._trajectory
        reference_time = min(max(elapsed, 0.0), trajectory.duration_s)
        reference = trajectory.sample_joint_state(reference_time)

        try:
            actual = self.arm.read_joint_state()
            if actual.names != trajectory.joint_names:
                self._fail(
                    "arm joint names changed during execution",
                    now_s,
                )
                return self.status

            self._tracking_error_rad = float(
                np.max(np.abs(actual.position_rad - reference.position_rad))
            )
            if self._tracking_error_rad > self.tracking_tolerance_rad:
                self._fail(
                    "tracking error exceeded tolerance: "
                    f"{self._tracking_error_rad:.6f} rad > "
                    f"{self.tracking_tolerance_rad:.6f} rad",
                    now_s,
                )
                return self.status

            self.arm.command_joint_positions(reference)
        except TrajectoryExecutionError:
            raise
        except Exception as exc:
            self._fail(f"arm command/read failed: {exc}", now_s)
            return self.status

        final_position = trajectory.positions_rad[-1]
        self._goal_error_rad = float(
            np.max(np.abs(actual.position_rad - final_position))
        )

        if elapsed >= trajectory.duration_s:
            if self._goal_error_rad <= self.goal_tolerance_rad:
                if self._hold_since_s is None:
                    self._hold_since_s = float(now_s)
                if float(now_s) - self._hold_since_s >= self.goal_hold_s:
                    self._succeed(now_s)
            else:
                self._hold_since_s = None

            if (
                self._state == ExecutionState.RUNNING
                and self.timeout_s is not None
                and elapsed > trajectory.duration_s + self.timeout_s
            ):
                self._fail(
                    "execution timed out before reaching goal tolerance",
                    now_s,
                )

        return self.status

    def cancel(self, now_s: float | None = None) -> ExecutionStatus:
        """Cancel execution without issuing a new reference command."""
        if self._state == ExecutionState.RUNNING:
            self._finished_at_s = time.monotonic() if now_s is None else float(now_s)
            self._failure_reason = "canceled"
            self._state = ExecutionState.CANCELED
        return self.status

    def _succeed(self, now_s: float) -> None:
        self._finished_at_s = float(now_s)
        self._state = ExecutionState.SUCCEEDED

    def _fail(self, reason: str, now_s: float) -> None:
        self._finished_at_s = float(now_s)
        self._failure_reason = reason
        self._state = ExecutionState.FAILED
