import numpy as np
import pytest

from replace_disk_robot.core import JointState, Pose, TrajectoryPoint
from replace_disk_robot.kinematics.planar2 import Planar2LinkKinematics
from replace_disk_robot.planning.optimize import TrajectoryOptimizer
from replace_disk_robot.planning.planar2_collision import (
    CircleObstacle,
    Planar2LinkCollisionChecker,
)
from replace_disk_robot.planning.search import RRTConnectPlanner
from replace_disk_robot.task import (
    GoalValidationError,
    JointLimitError,
    MotionPipeline,
)


def _make_pipeline():
    kinematics = Planar2LinkKinematics(link1_m=1.0, link2_m=1.0)
    checker = Planar2LinkCollisionChecker(
        kinematics,
        obstacles=[CircleObstacle(x=1.5, y=0.0, radius=0.2)],
        link_radius_m=0.02,
    )
    planner = RRTConnectPlanner(
        checker,
        sample_bounds=[(-3.2, 3.2), (-3.2, 3.2)],
        step_size_rad=0.2,
        max_iterations=5000,
        max_connection_steps=300,
        random_seed=0,
        path_pruning=False,
    )
    optimizer = TrajectoryOptimizer(
        kinematics,
        collision_checker=checker,
        task_weight=0.0,
        velocity_weight=10.0,
        acceleration_weight=2.0,
        dt_s=0.1,
        method="gradient",
        max_iterations_gradient=40,
    )
    pipeline = MotionPipeline(
        kinematics,
        checker,
        planner,
        optimizer,
        joint_limits_rad=(
            np.array([-3.2, -3.2]),
            np.array([3.2, 3.2]),
        ),
        optimization_samples=16,
        optimization_task_weight=0.0,
        collision_samples=20,
    )
    return kinematics, checker, pipeline


def test_motion_pipeline_plans_joint_goal_and_rechecks_optimized_path():
    kinematics, checker, pipeline = _make_pipeline()
    start = JointState(kinematics.joint_names, [-1.0, 0.0])
    goal = JointState(kinematics.joint_names, [2.0, 0.0])

    result = pipeline.plan(start, goal)

    assert result.optimized is True
    assert result.joint_names == kinematics.joint_names
    assert result.waypoint_count >= 2
    np.testing.assert_allclose(result.trajectory[0].position_rad, start.position_rad)
    np.testing.assert_allclose(result.trajectory[-1].position_rad, goal.position_rad)
    assert result.optimization_cost_after is not None
    assert (
        result.optimization_cost_after["total"]
        <= result.optimization_cost_before["total"]
    )
    for point in result.trajectory:
        state = JointState(kinematics.joint_names, point.position_rad)
        assert checker.is_collision_free(state)


def test_motion_pipeline_accepts_cartesian_pose_goal():
    kinematics, _, pipeline = _make_pipeline()
    start = JointState(kinematics.joint_names, [-1.0, 0.0])
    goal_joints = JointState(kinematics.joint_names, [2.0, 0.0])
    goal_pose = kinematics.forward(goal_joints)

    result = pipeline.plan(start, goal_pose)

    np.testing.assert_allclose(
        result.trajectory[-1].position_rad,
        goal_joints.position_rad,
        atol=1e-5,
    )


def test_motion_pipeline_rejects_out_of_limit_goal():
    kinematics, _, pipeline = _make_pipeline()
    start = JointState(kinematics.joint_names, [-1.0, 0.0])
    goal = JointState(kinematics.joint_names, [4.0, 0.0])

    with pytest.raises(JointLimitError, match="goal state violates joint limits"):
        pipeline.plan(start, goal)


def test_motion_pipeline_rejects_wrong_pose_frame():
    kinematics, _, pipeline = _make_pipeline()
    start = JointState(kinematics.joint_names, [-1.0, 0.0])
    wrong_frame_pose = Pose(
        frame_id="not_base",
        position_m=np.array([1.0, 0.0, 0.0]),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    with pytest.raises(GoalValidationError, match="goal pose must be expressed"):
        pipeline.plan(start, wrong_frame_pose)


def test_optimizer_joint_bounds_keep_interior_points_inside_limits():
    kinematics = Planar2LinkKinematics(link1_m=1.0, link2_m=1.0)
    initial = [
        TrajectoryPoint(i * 0.1, q)
        for i, q in enumerate(
            np.array(
                [
                    [0.0, 0.0],
                    [0.2, -0.3],
                    [0.4, 0.3],
                    [0.6, -0.2],
                    [0.8, 0.0],
                ]
            )
        )
    ]
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    optimizer = TrajectoryOptimizer(
        kinematics,
        task_weight=0.0,
        velocity_weight=10.0,
        acceleration_weight=2.0,
        method="gradient",
        max_iterations_gradient=40,
        joint_lower_limits_rad=lower,
        joint_upper_limits_rad=upper,
    )

    optimized = optimizer.optimize(initial)

    assert len(optimized) == len(initial)
    np.testing.assert_allclose(optimized[0].position_rad, initial[0].position_rad)
    np.testing.assert_allclose(optimized[-1].position_rad, initial[-1].position_rad)
    for point in optimized:
        assert np.all(point.position_rad >= lower)
        assert np.all(point.position_rad <= upper)


def test_optimizer_without_kinematics_does_not_require_cartesian_target():
    initial = [
        TrajectoryPoint(i * 0.1, q)
        for i, q in enumerate(np.array([[0.0, 0.0], [0.2, 0.1], [0.4, 0.0]]))
    ]
    optimizer = TrajectoryOptimizer(
        kinematics=None,
        task_weight=1.0,
        velocity_weight=10.0,
        acceleration_weight=2.0,
        method="gradient",
        max_iterations_gradient=20,
    )

    optimized = optimizer.optimize(initial)
    costs = optimizer.evaluate(optimized)

    assert len(optimized) == len(initial)
    assert costs["task"] == 0.0
    np.testing.assert_allclose(optimized[0].position_rad, initial[0].position_rad)
    np.testing.assert_allclose(optimized[-1].position_rad, initial[-1].position_rad)
