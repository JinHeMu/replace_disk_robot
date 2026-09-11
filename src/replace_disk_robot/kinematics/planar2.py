import numpy as np
from numpy.typing import NDArray

from replace_disk_robot.core.types import JointState, Pose


class Planar2LinkKinematics:
    """2-DOF planar arm moving in the XY plane."""

    def __init__(self, link1_m: float, link2_m: float) -> None:
        if link1_m <= 0 or link2_m <= 0:
            raise ValueError("link lengths must be positive")

        self.link1_m = float(link1_m)
        self.link2_m = float(link2_m)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return ("joint1", "joint2")

    @property
    def dof(self) -> int:
        return 2

    def forward(self, joints: JointState) -> Pose:
        self._check_joints(joints)

        q1, q2 = joints.position_rad
        l1, l2 = self.link1_m, self.link2_m

        x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
        y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
        theta = q1 + q2

        quaternion = np.array([
            np.cos(theta / 2.0),
            0.0,
            0.0,
            np.sin(theta / 2.0),
        ])

        return Pose(
            frame_id="base",
            position_m=np.array([x, y, 0.0]),
            quaternion_wxyz=quaternion,
        )

    def jacobian(self, joints: JointState) -> NDArray[np.float64]:
        self._check_joints(joints)

        q1, q2 = joints.position_rad
        l1, l2 = self.link1_m, self.link2_m

        return np.array([
            [-l1 * np.sin(q1) - l2 * np.sin(q1 + q2),
             -l2 * np.sin(q1 + q2)],

            [ l1 * np.cos(q1) + l2 * np.cos(q1 + q2),
              l2 * np.cos(q1 + q2)],

            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ])

    def inverse(self, target: Pose, seed: JointState) -> JointState:
        self._check_joints(seed)

        x, y = target.position_m[:2]
        l1, l2 = self.link1_m, self.link2_m

        cos_q2 = (x**2 + y**2 - l1**2 - l2**2) / (2.0 * l1 * l2)

        if abs(cos_q2) > 1.0:
            raise RuntimeError("IK target is outside the reachable workspace")

        cos_q2 = np.clip(cos_q2, -1.0, 1.0)

        q2_a = np.arccos(cos_q2)
        q2_b = -q2_a

        q_a = np.array([self._solve_q1(x, y, q2_a), q2_a])
        q_b = np.array([self._solve_q1(x, y, q2_b), q2_b])

        if np.linalg.norm(q_a - seed.position_rad) <= np.linalg.norm(q_b - seed.position_rad):
            solution = q_a
        else:
            solution = q_b

        return JointState(names=self.joint_names, position_rad=solution)

    def _solve_q1(self, x: float, y: float, q2: float) -> float:
        l1, l2 = self.link1_m, self.link2_m

        return float(
            np.arctan2(y, x)
            - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        )

    def _check_joints(self, joints: JointState) -> None:
        if joints.names != self.joint_names:
            raise ValueError(
                f"expected joint names {self.joint_names}, got {joints.names}"
            )