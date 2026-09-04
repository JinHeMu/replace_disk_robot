import numpy as np

from hard_disk_robot.core.types import JointState
from hard_disk_robot.kinematics.planar2 import Planar2LinkKinematics


def test_basic_properties():
    kin = Planar2LinkKinematics(link1_m=1.0, link2_m=0.8)

    assert kin.joint_names == ("joint1", "joint2")
    assert kin.dof == 2


def test_forward_zero_configuration():
    kin = Planar2LinkKinematics(link1_m=1.0, link2_m=0.8)

    joints = JointState(
        names=kin.joint_names,
        position_rad=np.array([0.0, 0.0]),
    )

    pose = kin.forward(joints)

    np.testing.assert_allclose(
        pose.position_m,
        np.array([1.8, 0.0, 0.0]),
    )


def test_jacobian_shape():
    kin = Planar2LinkKinematics(link1_m=1.0, link2_m=0.8)

    joints = JointState(
        names=kin.joint_names,
        position_rad=np.array([0.0, 0.0]),
    )

    J = kin.jacobian(joints)

    assert J.shape == (6, 2)