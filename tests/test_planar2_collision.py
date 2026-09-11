import numpy as np
import pytest

from replace_disk_robot.core import JointState
from replace_disk_robot.kinematics.planar2 import Planar2LinkKinematics
from replace_disk_robot.planning.planar2_collision import (
    CircleObstacle,
    Planar2LinkCollisionChecker,
)


def _lin_horizontal_arm() -> tuple[Planar2LinkKinematics, Planar2LinkCollisionChecker]:
    kinematics = Planar2LinkKinematics(link1_m=1.0, link2_m=0.8)
    checker = Planar2LinkCollisionChecker(
        kinematics,
        obstacles=[
            CircleObstacle(x=1.2, y=0.0, radius=0.1),
            CircleObstacle(x=0.4, y=0.6, radius=0.2),
        ],
        link_radius_m=0.02,
    )
    return kinematics, checker


def test_no_obstacle_is_free():
    kinematics = Planar2LinkKinematics(link1_m=1.0, link2_m=0.8)
    checker = Planar2LinkCollisionChecker(kinematics)
    joints = JointState(kinematics.joint_names, [0.0, 0.0])
    assert checker.is_collision_free(joints)
    assert checker.minimum_distance(joints) == float("inf")


def test_horizontal_arm_collides_with_obstacle():
    kinematics, checker = _lin_horizontal_arm()
    joints = JointState(kinematics.joint_names, [0.0, 0.0])
    assert not checker.is_collision_free(joints)
    assert checker.minimum_distance(joints) < 0.0


def test_collision_free_pose_detected():
    kinematics, checker = _lin_horizontal_arm()
    # A bent configuration that keeps both links away from the obstacles.
    joints = JointState(kinematics.joint_names, [np.pi / 2, -np.pi / 2])
    assert checker.is_collision_free(joints)


def test_obstacle_can_be_added_at_runtime():
    kinematics = Planar2LinkKinematics(link1_m=1.0, link2_m=0.8)
    checker = Planar2LinkCollisionChecker(kinematics)
    before = checker.minimum_distance(JointState(kinematics.joint_names, [0.0, 0.0]))
    checker.add_circle_obstacle(0.5, 0.0, 0.05)
    after = checker.minimum_distance(JointState(kinematics.joint_names, [0.0, 0.0]))
    assert after < before
