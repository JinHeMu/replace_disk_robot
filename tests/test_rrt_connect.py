import numpy as np
import pytest

from replace_disk_robot.core import JointState
from replace_disk_robot.kinematics.planar2 import Planar2LinkKinematics
from replace_disk_robot.planning.planar2_collision import (
    CircleObstacle,
    Planar2LinkCollisionChecker,
)
from replace_disk_robot.planning.search import RRTConnectPlanner


def _make_planner():
    kinematics = Planar2LinkKinematics(link1_m=1.0, link2_m=1.0)
    checker = Planar2LinkCollisionChecker(
        kinematics,
        obstacles=[CircleObstacle(x=1.5, y=0.0, radius=0.2)],
        link_radius_m=0.02,
    )
    start = JointState(kinematics.joint_names, [-1.0, 0.0])
    goal = JointState(kinematics.joint_names, [2.0, 0.0])
    planner = RRTConnectPlanner(
        checker,
        sample_bounds=[(-3.2, 3.2), (-3.2, 3.2)],
        step_size_rad=0.2,
        max_iterations=4000,
        max_connection_steps=200,
        random_seed=0,
    )
    return kinematics, checker, start, goal, planner


def test_rrt_connect_finds_collision_free_path():
    kinematics, checker, start, goal, planner = _make_planner()

    path = planner.plan(start, goal)

    assert len(path) >= 2
    assert path[0].position_rad.shape == (2,)
    assert path[-1].position_rad.shape == (2,)

    np.testing.assert_allclose(path[0].position_rad, start.position_rad)
    np.testing.assert_allclose(path[-1].position_rad, goal.position_rad)

    for point in path:
        state = JointState(kinematics.joint_names, point.position_rad)
        assert checker.is_collision_free(state)

    times = [p.time_from_start_s for p in path]
    assert times == sorted(times)


def test_rrt_connect_rejects_invalid_endpoints():
    kinematics = Planar2LinkKinematics(link1_m=1.0, link2_m=1.0)
    checker = Planar2LinkCollisionChecker(
        kinematics,
        obstacles=[CircleObstacle(x=1.5, y=0.0, radius=0.2)],
        link_radius_m=0.02,
    )
    start = JointState(kinematics.joint_names, [0.0, 0.0])  # in collision
    goal = JointState(kinematics.joint_names, [2.0, 0.0])
    planner = RRTConnectPlanner(checker, random_seed=1)

    with pytest.raises(RuntimeError, match="start configuration is in collision"):
        planner.plan(start, goal)


class _AlwaysFreeChecker:
    def is_collision_free(self, joints: JointState) -> bool:
        return True

    def minimum_distance(self, joints: JointState) -> float:
        return 1.0


def test_rrt_path_pruning_collapses_free_detour_to_straight_line():
    names = ("q",)
    start = JointState(names, [0.0])
    goal = JointState(names, [1.0])

    raw_planner = RRTConnectPlanner(
        _AlwaysFreeChecker(),
        step_size_rad=0.2,
        max_iterations=100,
        random_seed=0,
        path_pruning=False,
    )
    pruned_planner = RRTConnectPlanner(
        _AlwaysFreeChecker(),
        step_size_rad=0.2,
        max_iterations=100,
        random_seed=0,
        path_pruning=True,
    )

    raw_path = raw_planner.plan(start, goal)
    pruned_path = pruned_planner.plan(start, goal)

    assert len(raw_path) > 2
    assert len(pruned_path) == 2
    np.testing.assert_allclose(pruned_path[0].position_rad, start.position_rad)
    np.testing.assert_allclose(pruned_path[-1].position_rad, goal.position_rad)
