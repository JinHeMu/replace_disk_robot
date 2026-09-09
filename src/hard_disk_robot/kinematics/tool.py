"""Add a fixed TCP to an existing KinematicsPort without editing its robot model."""
import numpy as np

from ..core.ports import KinematicsPort
from ..core.types import JointState, Pose
from ..core.rotation import rotation_matrix, multiply, conjugate


class FixedToolKinematics:
    def __init__(self, parent: KinematicsPort, parent_frame: str, tool_in_parent: Pose):
        if tool_in_parent.frame_id != parent_frame:
            raise ValueError('Tool transform must be expressed in the parent end-effector frame')
        self.parent = parent
        self.offset = tool_in_parent

    @property
    def joint_names(self):
        return self.parent.joint_names

    @property
    def dof(self):
        return self.parent.dof

    def forward(self, joints: JointState) -> Pose:
        pose = self.parent.forward(joints)
        return Pose(pose.frame_id,
                    pose.position_m + rotation_matrix(pose.quaternion_wxyz) @ self.offset.position_m,
                    multiply(pose.quaternion_wxyz, self.offset.quaternion_wxyz))

    def jacobian(self, joints: JointState):
        pose = self.parent.forward(joints)
        r = rotation_matrix(pose.quaternion_wxyz) @ self.offset.position_m
        jac = self.parent.jacobian(joints).copy()
        jac[:3] += np.cross(jac[3:].T, r).T
        return jac

    def inverse(self, target: Pose, seed: JointState) -> JointState:
        q = multiply(target.quaternion_wxyz, conjugate(self.offset.quaternion_wxyz))
        position = target.position_m - rotation_matrix(q) @ self.offset.position_m
        return self.parent.inverse(Pose(target.frame_id, position, q), seed)
