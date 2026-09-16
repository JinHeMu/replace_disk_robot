import numpy as np
import pytest

from replace_disk_robot.control.trajectory import (
    ExecutionState,
    TrajectoryExecutor,
    TrajectoryStartError,
)
from replace_disk_robot.core import JointState, TrajectoryPoint
from replace_disk_robot.kinematics.planar2 import Planar2LinkKinematics
from replace_disk_robot.planning.search import RRTConnectPlanner
from replace_disk_robot.planning.timing import (
    assign_quintic_timing,
    validate_timed_trajectory,
)
from replace_disk_robot.task import (
    MotionPipeline,
    MotionPipelineBusyError,
    MotionPipelineState,
)


class _AlwaysFreeChecker:
    def is_collision_free(self, joints: JointState) -> bool:
        return True

    def minimum_distance(self, joints: JointState) -> float:
        return 1.0


class _ImmediateArm:
    def __init__(self, names: tuple[str, ...], position_rad) -> None:
        self.names = tuple(names)
        self.position_rad = np.asarray(position_rad, dtype=float).copy()

    def read_joint_state(self) -> JointState:
        return JointState(self.names, self.position_rad.copy())

    def command_joint_positions(self, target: JointState) -> None:
        if target.names != self.names:
            raise ValueError("unexpected joint names")
        self.position_rad = np.asarray(target.position_rad, dtype=float).copy()


class _StuckArm(_ImmediateArm):
    def command_joint_positions(self, target: JointState) -> None:
        return


def _one_joint_trajectory():
    return assign_quintic_timing(
        [
            TrajectoryPoint(0.0, [0.0]),
            TrajectoryPoint(0.1, [1.0]),
        ],
        ("q",),
        velocity_limits_rad_s=10.0,
        acceleration_limits_rad_s2=10.0,
        min_segment_time_s=0.1,
    )


def test_quintic_timing_respects_configured_limits():
    trajectory = _one_joint_trajectory()

    metrics = validate_timed_trajectory(
        trajectory,
        samples_per_segment=30,
    )

    assert metrics["max_velocity_rad_s"] <= 10.0 + 1e-9
    assert metrics["max_acceleration_rad_s2"] <= 10.0 + 1e-9
    assert trajectory.sample(0.0)[0] == pytest.approx(0.0)
    assert trajectory.sample(trajectory.duration_s)[0] == pytest.approx(1.0)


def test_executor_finishes_only_after_tolerance_and_hold_time():
    trajectory = _one_joint_trajectory()
    arm = _ImmediateArm(("q",), [0.0])
    executor = TrajectoryExecutor(
        arm,
        start_tolerance_rad=0.01,
        tracking_tolerance_rad=0.2,
        goal_tolerance_rad=0.01,
        goal_hold_s=0.1,
        timeout_s=1.0,
    )

    executor.start(trajectory, now_s=0.0, planned_start=JointState(("q",), [0.0]))
    for now_s in np.arange(0.01, trajectory.duration_s + 0.2, 0.01):
        executor.update(float(now_s))

    status = executor.status
    assert status.state == ExecutionState.SUCCEEDED
    assert status.goal_error_rad <= 0.01


def test_executor_reports_tracking_error_and_cancel():
    stuck_arm = _StuckArm(("q",), [0.0])
    executor = TrajectoryExecutor(
        stuck_arm,
        tracking_tolerance_rad=0.2,
        goal_tolerance_rad=0.01,
        goal_hold_s=0.1,
    )
    executor.start(_one_joint_trajectory(), now_s=0.0)
    executor.update(2.0)
    assert executor.state == ExecutionState.FAILED
    assert executor.failure_reason is not None
    assert "tracking error" in executor.failure_reason

    immediate_arm = _ImmediateArm(("q",), [0.0])
    cancel_executor = TrajectoryExecutor(immediate_arm)
    cancel_executor.start(_one_joint_trajectory(), now_s=0.0)
    cancel_executor.update(0.01)
    cancel_executor.cancel(now_s=0.02)
    assert cancel_executor.state == ExecutionState.CANCELED


def test_executor_rejects_start_state_far_from_plan():
    trajectory = _one_joint_trajectory()
    arm = _ImmediateArm(("q",), [0.5])
    executor = TrajectoryExecutor(arm, start_tolerance_rad=0.01)

    with pytest.raises(TrajectoryStartError, match="measured start state"):
        executor.start(
            trajectory,
            now_s=0.0,
            planned_start=JointState(("q",), [0.0]),
        )


def test_motion_pipeline_submit_goal_updates_execution_state():
    kinematics = Planar2LinkKinematics(link1_m=1.0, link2_m=1.0)
    checker = _AlwaysFreeChecker()
    planner = RRTConnectPlanner(
        checker,
        step_size_rad=0.2,
        max_iterations=200,
        max_connection_steps=200,
        random_seed=0,
    )
    arm = _ImmediateArm(kinematics.joint_names, [0.0, 0.0])
    pipeline = MotionPipeline(
        kinematics,
        checker,
        planner,
        optimizer=None,
        optimize=False,
        joint_limits_rad=(
            np.array([-3.0, -3.0]),
            np.array([3.0, 3.0]),
        ),
        arm=arm,
        velocity_limits_rad_s=np.array([1.0, 1.0]),
        acceleration_limits_rad_s2=np.array([2.0, 2.0]),
        min_segment_time_s=0.1,
        start_tolerance_rad=0.01,
        tracking_tolerance_rad=1.0,
        goal_tolerance_rad=0.01,
        goal_hold_s=0.1,
        timeout_s=2.0,
    )
    goal = JointState(kinematics.joint_names, [0.4, 0.3])

    plan = pipeline.submit_goal(goal, now_s=0.0)
    assert pipeline.state == MotionPipelineState.EXECUTING
    assert plan.trajectory[-1].position_rad[0] == pytest.approx(0.4)
    with pytest.raises(MotionPipelineBusyError):
        pipeline.submit_goal(goal, now_s=0.0)

    for now_s in np.arange(0.02, 5.0, 0.02):
        pipeline.update(float(now_s))
        if pipeline.state == MotionPipelineState.SUCCEEDED:
            break

    assert pipeline.state == MotionPipelineState.SUCCEEDED
    np.testing.assert_allclose(
        arm.read_joint_state().position_rad,
        goal.position_rad,
        atol=0.01,
    )
