"""P0/P1 planning pipeline: Goal -> validation/IK -> RRT -> trajectory opt.

This module intentionally stops before runtime execution.  It accepts either a
joint-space goal or a Cartesian pose goal, validates the current and target
states, runs the configured RRT planner, optionally optimizes the resulting
joint-space trajectory, and finally re-checks the optimized path.

The optimized object is always a ``Sequence[TrajectoryPoint]`` in joint space:
the Cartesian task cost is only an optional *initialization/objective* inside
``TrajectoryOptimizer``; it is not part of the pipeline's output contract.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ..core import (
    ArmPort,
    CollisionCheckerPort,
    JointState,
    KinematicsPort,
    Pose,
    TrajectoryOptimizerPort,
    TrajectoryPlannerPort,
    TrajectoryPoint,
)
from ..planning.collision import is_path_collision_free
from ..planning.timing import assign_quintic_timing, validate_timed_trajectory

if TYPE_CHECKING:
    from ..control.trajectory import ExecutionState, TrajectoryExecutor


MotionGoal: TypeAlias = JointState | Pose


class MotionPipelineError(RuntimeError):
    """Base class for planning-pipeline failures."""


class GoalValidationError(MotionPipelineError):
    """The current state or the requested goal is invalid."""


class JointLimitError(GoalValidationError):
    """A joint state violates the configured limits."""


class GoalCollisionError(GoalValidationError):
    """The start or target configuration is in collision."""


class IkError(GoalValidationError):
    """A Cartesian goal cannot be converted to an admissible joint target."""


class SearchError(MotionPipelineError):
    """RRT search failed or returned an invalid path."""


class OptimizationError(MotionPipelineError):
    """Trajectory optimization failed or returned an invalid result."""


class TrajectoryValidationError(MotionPipelineError):
    """A trajectory violates endpoint, limit, ordering, or collision checks."""


class MotionPipelineBusyError(MotionPipelineError):
    """A new goal was submitted while planning or execution was active."""


class ExecutionNotConfiguredError(MotionPipelineError):
    """Execution was requested without an arm or trajectory executor."""


@dataclass(frozen=True)
class MotionPlan:
    """Immutable result of one successful P0/P1 planning request."""

    joint_names: tuple[str, ...]
    start: JointState
    goal: JointState
    raw_trajectory: tuple[TrajectoryPoint, ...]
    trajectory: tuple[TrajectoryPoint, ...]
    optimized: bool
    planning_time_s: float
    optimization_cost_before: Mapping[str, float] | None = None
    optimization_cost_after: Mapping[str, float] | None = None

    @property
    def waypoint_count(self) -> int:
        return len(self.trajectory)


class MotionPipelineState(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass(frozen=True)
class MotionPipelineStatus:
    state: MotionPipelineState
    last_error: str | None = None
    planning_time_s: float | None = None
    trajectory_duration_s: float | None = None
    tracking_error_rad: float = 0.0
    goal_error_rad: float = float("inf")
    execution_state: Any | None = None


def _resample_path(
    path: Sequence[TrajectoryPoint],
    n: int,
) -> tuple[TrajectoryPoint, ...]:
    """Resample a path to ``n`` points evenly along joint-space arc length."""
    if len(path) >= n or len(path) < 2:
        return tuple(path)

    q = np.array([point.position_rad for point in path], dtype=float)
    times = np.array([point.time_from_start_s for point in path], dtype=float)
    segment_lengths = np.linalg.norm(np.diff(q, axis=0), axis=1)
    distances = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if distances[-1] <= 1e-12:
        return tuple(path)

    sample_distances = np.linspace(0.0, distances[-1], n)
    q_resampled = np.column_stack(
        [
            np.interp(sample_distances, distances, q[:, joint_index])
            for joint_index in range(q.shape[1])
        ]
    )
    t_resampled = np.interp(sample_distances, distances, times)
    return tuple(
        TrajectoryPoint(
            time_from_start_s=float(t_resampled[i]),
            position_rad=q_resampled[i],
        )
        for i in range(n)
    )


class MotionPipeline:
    """Plan a joint-space or Cartesian goal using existing P0/P1 primitives.

    Parameters
    ----------
    kinematics:
        Object implementing ``joint_names``, ``forward`` and ``inverse``.
    collision_checker:
        Collision checker used for start/goal/path validation.
    planner:
        RRT-like planner implementing ``plan(start, goal)``.
    optimizer:
        Optional trajectory optimizer implementing
        ``optimize(Sequence[TrajectoryPoint])``.  Required when ``optimize=True``.
    joint_limits_rad:
        Optional ``(lower, upper)`` pair.  If omitted, the pipeline uses
        ``kinematics.joint_limits_rad`` when available.
    optimize:
        If True, run the optimizer between search and final validation.
    optimization_samples:
        Optional number of waypoints to resample the searched path to before
        optimization.  The raw search path is left untouched in
        ``MotionPlan.raw_trajectory``.
    optimization_task_weight:
        Optional temporary override for ``optimizer.task_weight`` during this
        pipeline's optimize call.  P1 recommends ``0.0`` to optimize joint-space
        smoothness and collision avoidance first.
    collision_samples:
        Number of interior samples used for each segment collision check.
    endpoint_tolerance_rad:
        Maximum allowed joint distance between trajectory endpoints and the
        validated start/goal.
    arm:
        Optional ``ArmPort`` used by ``submit_goal``/``update``/``cancel``.
    executor:
        Optional pre-built ``TrajectoryExecutor``.  If omitted and ``arm`` is
        provided, the pipeline constructs one.
    velocity_limits_rad_s:
        Per-joint velocity limits used by P2 time allocation.  Defaults to
        1.0 rad/s per joint when execution is configured.
    acceleration_limits_rad_s2:
        Per-joint acceleration limits used by P2 time allocation.  Defaults to
        2.0 rad/s^2 per joint when execution is configured.
    min_segment_time_s:
        Minimum duration of each quintic segment.
    start_tolerance_rad:
        Start tracking tolerance passed to the executor.
    tracking_tolerance_rad:
        Maximum allowed tracking error passed to the executor.
    goal_tolerance_rad:
        Final goal tolerance passed to the executor.
    goal_hold_s:
        Required hold time inside the goal tolerance.
    timeout_s:
        Optional extra timeout after the trajectory duration.
    """

    def __init__(
        self,
        kinematics: KinematicsPort,
        collision_checker: CollisionCheckerPort,
        planner: TrajectoryPlannerPort,
        optimizer: TrajectoryOptimizerPort | None = None,
        *,
        joint_limits_rad: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
        optimize: bool = True,
        optimization_samples: int | None = None,
        optimization_task_weight: float | None = None,
        collision_samples: int = 40,
        endpoint_tolerance_rad: float = 1e-6,
        arm: ArmPort | None = None,
        executor: TrajectoryExecutor | None = None,
        velocity_limits_rad_s: NDArray[np.float64] | None = None,
        acceleration_limits_rad_s2: NDArray[np.float64] | None = None,
        min_segment_time_s: float = 0.05,
        start_tolerance_rad: float = 0.1,
        tracking_tolerance_rad: float = 0.5,
        goal_tolerance_rad: float = 0.02,
        goal_hold_s: float = 0.2,
        timeout_s: float | None = None,
    ) -> None:
        if collision_samples < 1:
            raise ValueError("collision_samples must be at least 1")
        if optimization_samples is not None and optimization_samples < 2:
            raise ValueError("optimization_samples must be at least 2")
        if endpoint_tolerance_rad <= 0.0:
            raise ValueError("endpoint_tolerance_rad must be positive")
        if optimize and optimizer is None:
            raise ValueError("optimizer is required when optimize=True")
        if optimization_task_weight is not None and optimization_task_weight < 0.0:
            raise ValueError("optimization_task_weight must be non-negative")
        if min_segment_time_s <= 0.0:
            raise ValueError("min_segment_time_s must be positive")

        self.kinematics = kinematics
        self.collision_checker = collision_checker
        self.planner = planner
        self.optimizer = optimizer
        self.optimize = bool(optimize)
        self.optimization_samples = optimization_samples
        self.optimization_task_weight = optimization_task_weight
        self.collision_samples = int(collision_samples)
        self.endpoint_tolerance_rad = float(endpoint_tolerance_rad)
        self.min_segment_time_s = float(min_segment_time_s)
        self._state = MotionPipelineState.IDLE
        self._last_plan: MotionPlan | None = None
        self._last_error: str | None = None
        self._planning_started_at_s: float | None = None
        self._executor: TrajectoryExecutor | None = executor
        self.arm: ArmPort | None = arm

        self.joint_names = tuple(kinematics.joint_names)
        if not self.joint_names:
            raise ValueError("kinematics.joint_names must not be empty")

        limits = joint_limits_rad
        if limits is None and hasattr(kinematics, "joint_limits_rad"):
            limits = kinematics.joint_limits_rad  # type: ignore[assignment]
        if limits is not None:
            lower, upper = limits
            lower_array = np.asarray(lower, dtype=float)
            upper_array = np.asarray(upper, dtype=float)
            if lower_array.shape != (len(self.joint_names),):
                raise ValueError(
                    "lower joint limits must have shape "
                    f"({len(self.joint_names)},), got {lower_array.shape}"
                )
            if upper_array.shape != (len(self.joint_names),):
                raise ValueError(
                    "upper joint limits must have shape "
                    f"({len(self.joint_names)},), got {upper_array.shape}"
                )
            if not np.all(np.isfinite(lower_array)) or not np.all(
                np.isfinite(upper_array)
            ):
                raise ValueError("joint limits must be finite")
            if np.any(lower_array > upper_array):
                raise ValueError("lower joint limits must not exceed upper limits")
            self.joint_lower_limits_rad: NDArray[np.float64] | None = (
                lower_array.copy()
            )
            self.joint_upper_limits_rad: NDArray[np.float64] | None = (
                upper_array.copy()
            )
        else:
            self.joint_lower_limits_rad = None
            self.joint_upper_limits_rad = None

        # Fill optimizer bounds when the optimizer exposes the same bound API.
        if self.optimizer is not None and self.joint_lower_limits_rad is not None:
            lower_attr = getattr(self.optimizer, "joint_lower_limits_rad", None)
            upper_attr = getattr(self.optimizer, "joint_upper_limits_rad", None)
            if lower_attr is None and upper_attr is None:
                try:
                    setattr(
                        self.optimizer,
                        "joint_lower_limits_rad",
                        self.joint_lower_limits_rad.copy(),
                    )
                    setattr(
                        self.optimizer,
                        "joint_upper_limits_rad",
                        self.joint_upper_limits_rad.copy(),
                    )
                except Exception:
                    pass

        execution_configured = self._executor is not None or self.arm is not None
        if velocity_limits_rad_s is None:
            self.velocity_limits_rad_s = (
                np.full(len(self.joint_names), 1.0) if execution_configured else None
            )
        else:
            self.velocity_limits_rad_s = self._coerce_limit_vector(
                velocity_limits_rad_s,
                "velocity_limits_rad_s",
            )
        if acceleration_limits_rad_s2 is None:
            self.acceleration_limits_rad_s2 = (
                np.full(len(self.joint_names), 2.0) if execution_configured else None
            )
        else:
            self.acceleration_limits_rad_s2 = self._coerce_limit_vector(
                acceleration_limits_rad_s2,
                "acceleration_limits_rad_s2",
            )

        if self._executor is None and self.arm is not None:
            from ..control.trajectory import TrajectoryExecutor

            self._executor = TrajectoryExecutor(
                self.arm,
                start_tolerance_rad=start_tolerance_rad,
                tracking_tolerance_rad=tracking_tolerance_rad,
                goal_tolerance_rad=goal_tolerance_rad,
                goal_hold_s=goal_hold_s,
                timeout_s=timeout_s,
            )
        elif self._executor is not None and self.arm is None:
            self.arm = getattr(self._executor, "arm", None)
        elif self._executor is not None:
            self.arm = self.arm or getattr(self._executor, "arm", None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plan(self, current: JointState, goal: MotionGoal) -> MotionPlan:
        """Plan from measured ``current`` joints to ``goal``.

        ``goal`` may be a :class:`JointState` or a :class:`Pose` expressed in
        the kinematics base frame.
        """
        started_at = time.perf_counter()
        start = self._validate_start(current)
        goal_joints = self._resolve_goal(goal, start)

        raw_trajectory = self._search(start, goal_joints)
        self._validate_trajectory(
            raw_trajectory,
            start,
            goal_joints,
            context="raw search trajectory",
        )

        optimized = False
        cost_before: Mapping[str, float] | None = None
        cost_after: Mapping[str, float] | None = None
        trajectory = raw_trajectory

        if self.optimize:
            assert self.optimizer is not None
            optimization_input = (
                _resample_path(raw_trajectory, self.optimization_samples)
                if self.optimization_samples is not None
                else tuple(raw_trajectory)
            )
            self._validate_trajectory(
                optimization_input,
                start,
                goal_joints,
                context="optimization input trajectory",
            )

            old_task_weight = getattr(self.optimizer, "task_weight", None)
            try:
                if (
                    self.optimization_task_weight is not None
                    and old_task_weight is not None
                ):
                    self.optimizer.task_weight = self.optimization_task_weight  # type: ignore[attr-defined]
                optimized_sequence = self.optimizer.optimize(optimization_input)
            except Exception as exc:
                raise OptimizationError(
                    f"trajectory optimization failed: {exc}"
                ) from exc
            finally:
                if (
                    self.optimization_task_weight is not None
                    and old_task_weight is not None
                ):
                    self.optimizer.task_weight = old_task_weight  # type: ignore[attr-defined]

            trajectory = tuple(optimized_sequence)
            optimized = True
            self._validate_trajectory(
                trajectory,
                start,
                goal_joints,
                context="optimized trajectory",
            )

            evaluate = getattr(self.optimizer, "evaluate", None)
            if callable(evaluate):
                try:
                    cost_before = dict(evaluate(optimization_input))
                    cost_after = dict(evaluate(trajectory))
                except Exception:
                    cost_before = None
                    cost_after = None

        return MotionPlan(
            joint_names=self.joint_names,
            start=start,
            goal=goal_joints,
            raw_trajectory=tuple(raw_trajectory),
            trajectory=tuple(trajectory),
            optimized=optimized,
            planning_time_s=time.perf_counter() - started_at,
            optimization_cost_before=cost_before,
            optimization_cost_after=cost_after,
        )

    def plan_goal(self, goal: MotionGoal, current: JointState) -> MotionPlan:
        """Convenience wrapper with Goal-first argument order."""
        return self.plan(current, goal)

    def submit_goal(
        self,
        goal: MotionGoal,
        *,
        now_s: float | None = None,
        current: JointState | None = None,
    ) -> MotionPlan:
        """Plan and start executing a goal through the configured executor."""
        if self._executor is None:
            raise ExecutionNotConfiguredError(
                "submit_goal requires an arm or a TrajectoryExecutor"
            )
        if self._state in (MotionPipelineState.PLANNING, MotionPipelineState.EXECUTING):
            raise MotionPipelineBusyError(
                f"pipeline is busy in state {self._state.value!r}"
            )
        if now_s is None:
            now_s = time.monotonic()
        if current is None:
            if self.arm is None:
                raise ExecutionNotConfiguredError(
                    "no arm is available to read the current joint state"
                )
            current = self.arm.read_joint_state()

        self._state = MotionPipelineState.PLANNING
        self._last_error = None
        self._planning_started_at_s = time.perf_counter()
        try:
            plan = self.plan(current, goal)
            if (
                self.velocity_limits_rad_s is None
                or self.acceleration_limits_rad_s2 is None
            ):
                raise ExecutionNotConfiguredError(
                    "velocity and acceleration limits are required for execution"
                )
            timed_trajectory = assign_quintic_timing(
                plan.trajectory,
                self.joint_names,
                velocity_limits_rad_s=self.velocity_limits_rad_s,
                acceleration_limits_rad_s2=self.acceleration_limits_rad_s2,
                min_segment_time_s=self.min_segment_time_s,
            )
            validate_timed_trajectory(
                timed_trajectory,
                collision_checker=self.collision_checker,
                joint_lower_limits_rad=self.joint_lower_limits_rad,
                joint_upper_limits_rad=self.joint_upper_limits_rad,
                samples_per_segment=20,
            )
            self._executor.start(
                timed_trajectory,
                float(now_s),
                planned_start=plan.start,
            )
        except Exception as exc:
            self._state = MotionPipelineState.FAILED
            self._last_error = str(exc)
            raise

        self._last_plan = plan
        self._state = MotionPipelineState.EXECUTING
        return plan

    def update(self, now_s: float) -> MotionPipelineStatus:
        """Advance execution state; call once per control cycle."""
        if self._executor is None:
            raise ExecutionNotConfiguredError("update requires a TrajectoryExecutor")
        if self._state != MotionPipelineState.EXECUTING:
            return self.status

        execution_status = self._executor.update(float(now_s))
        if execution_status.state.value == "running":
            self._state = MotionPipelineState.EXECUTING
        elif execution_status.state.value == "succeeded":
            self._state = MotionPipelineState.SUCCEEDED
        elif execution_status.state.value == "canceled":
            self._state = MotionPipelineState.CANCELED
        else:
            self._state = MotionPipelineState.FAILED
            self._last_error = execution_status.failure_reason
        return self.status

    def cancel(self, now_s: float | None = None) -> MotionPipelineStatus:
        """Cancel any active execution."""
        if self._executor is None:
            raise ExecutionNotConfiguredError("cancel requires a TrajectoryExecutor")
        if self._state == MotionPipelineState.EXECUTING:
            self._executor.cancel(now_s)
            self._state = MotionPipelineState.CANCELED
            self._last_error = self._executor.failure_reason
        return self.status

    @property
    def state(self) -> MotionPipelineState:
        return self._state

    @property
    def last_plan(self) -> MotionPlan | None:
        return self._last_plan

    @property
    def status(self) -> MotionPipelineStatus:
        execution_status = (
            None if self._executor is None else self._executor.status
        )
        duration = (
            None
            if self._last_plan is None or self._executor is None
            else getattr(self._executor.trajectory, "duration_s", None)
        )
        planning_time = None
        if self._last_plan is not None:
            planning_time = self._last_plan.planning_time_s
        return MotionPipelineStatus(
            state=self._state,
            last_error=self._last_error,
            planning_time_s=planning_time,
            trajectory_duration_s=duration,
            tracking_error_rad=(
                0.0 if execution_status is None else execution_status.tracking_error_rad
            ),
            goal_error_rad=(
                float("inf")
                if execution_status is None
                else execution_status.goal_error_rad
            ),
            execution_state=(
                None if execution_status is None else execution_status.state
            ),
        )

    def _coerce_limit_vector(
        self,
        value: NDArray[np.float64],
        name: str,
    ) -> NDArray[np.float64]:
        array = np.asarray(value, dtype=float)
        n_joints = len(self.joint_names)
        if array.ndim == 0:
            array = np.full(n_joints, float(array))
        if array.shape != (n_joints,):
            raise ValueError(f"{name} must have shape ({n_joints},) or be a scalar")
        if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
            raise ValueError(f"{name} must contain finite positive values")
        return array.copy()

    # ------------------------------------------------------------------
    # Goal and state validation
    # ------------------------------------------------------------------
    def _validate_start(self, current: JointState) -> JointState:
        if current.names != self.joint_names:
            raise GoalValidationError(
                f"current joint names must be {self.joint_names}, got {current.names}"
            )
        start = JointState(self.joint_names, current.position_rad.copy())
        self._check_joint_limits(start, context="start state")
        if not self.collision_checker.is_collision_free(start):
            raise GoalCollisionError("start configuration is in collision")
        return start

    def _resolve_goal(self, goal: MotionGoal, seed: JointState) -> JointState:
        if isinstance(goal, JointState):
            if goal.names != self.joint_names:
                raise GoalValidationError(
                    f"goal joint names must be {self.joint_names}, got {goal.names}"
                )
            target = JointState(self.joint_names, goal.position_rad.copy())
        elif isinstance(goal, Pose):
            base_frame = self._base_frame(seed)
            if goal.frame_id != base_frame:
                raise GoalValidationError(
                    f"goal pose must be expressed in frame {base_frame!r}, "
                    f"got {goal.frame_id!r}"
                )
            try:
                target = self.kinematics.inverse(goal, seed)
            except Exception as exc:
                raise IkError(f"IK failed for Cartesian goal: {exc}") from exc
            if target.names != self.joint_names:
                raise GoalValidationError(
                    "IK returned joint names "
                    f"{target.names}, expected {self.joint_names}"
                )
        else:
            raise TypeError(
                "goal must be a JointState or Pose, "
                f"got {type(goal).__name__}"
            )

        self._check_joint_limits(target, context="goal state")
        if not self.collision_checker.is_collision_free(target):
            raise GoalCollisionError("goal configuration is in collision")
        return target

    def _base_frame(self, seed: JointState) -> str:
        base_frame = getattr(self.kinematics, "base_frame", None)
        if isinstance(base_frame, str) and base_frame:
            return base_frame
        return self.kinematics.forward(seed).frame_id

    def _check_joint_limits(self, state: JointState, *, context: str) -> None:
        if self.joint_lower_limits_rad is None:
            return
        if np.any(state.position_rad < self.joint_lower_limits_rad) or np.any(
            state.position_rad > self.joint_upper_limits_rad
        ):
            lower = self.joint_lower_limits_rad
            upper = self.joint_upper_limits_rad
            raise JointLimitError(
                f"{context} violates joint limits: "
                f"q={state.position_rad}, lower={lower}, upper={upper}"
            )

    # ------------------------------------------------------------------
    # Search, optimization, and trajectory validation
    # ------------------------------------------------------------------
    def _search(
        self,
        start: JointState,
        goal: JointState,
    ) -> tuple[TrajectoryPoint, ...]:
        try:
            path = tuple(self.planner.plan(start, goal))
        except Exception as exc:
            raise SearchError(f"RRT search failed: {exc}") from exc
        if len(path) < 2:
            raise SearchError("RRT search returned fewer than two waypoints")
        return path

    def _validate_trajectory(
        self,
        trajectory: Sequence[TrajectoryPoint],
        start: JointState,
        goal: JointState,
        *,
        context: str,
    ) -> None:
        if len(trajectory) < 2:
            raise TrajectoryValidationError(
                f"{context} must contain at least two waypoints"
            )

        times = np.array(
            [point.time_from_start_s for point in trajectory],
            dtype=float,
        )
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0.0):
            raise TrajectoryValidationError(
                f"{context} must have finite, non-decreasing time stamps"
            )

        states: list[JointState] = []
        for index, point in enumerate(trajectory):
            state = JointState(self.joint_names, point.position_rad)
            self._check_joint_limits(
                state,
                context=f"{context} waypoint {index}",
            )
            states.append(state)

        if not np.allclose(
            states[0].position_rad,
            start.position_rad,
            atol=self.endpoint_tolerance_rad,
            rtol=0.0,
        ):
            raise TrajectoryValidationError(
                f"{context} does not start at the validated start state"
            )
        if not np.allclose(
            states[-1].position_rad,
            goal.position_rad,
            atol=self.endpoint_tolerance_rad,
            rtol=0.0,
        ):
            raise TrajectoryValidationError(
                f"{context} does not end at the validated goal state"
            )

        for index, (segment_start, segment_goal) in enumerate(
            zip(states[:-1], states[1:])
        ):
            if not is_path_collision_free(
                self.collision_checker,
                segment_start,
                segment_goal,
                samples=self.collision_samples,
            ):
                raise TrajectoryValidationError(
                    f"{context} segment {index} is in collision"
                )
