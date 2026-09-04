import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hard_disk_robot.core.types import JointState
from hard_disk_robot.kinematics.ur5e import PROJECT_UR5E_URDF, UR5eKinematics


def _rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _pinocchio_is_safe_to_import() -> bool:
    spec = importlib.util.find_spec("pinocchio")
    if spec is None:
        return False
    origin = str(spec.origin or "")
    numpy_major = int(np.__version__.split(".", maxsplit=1)[0])
    return not (numpy_major >= 2 and origin.startswith("/opt/ros/humble/"))


@unittest.skipUnless(
    _pinocchio_is_safe_to_import(),
    "a Pinocchio build compatible with the active NumPy is required",
)
class UR5eKinematicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kinematics = UR5eKinematics(PROJECT_UR5E_URDF)

    def joints(self, position_rad: np.ndarray) -> JointState:
        return JointState(self.kinematics.joint_names, position_rad)

    def test_model_is_reduced_to_the_six_arm_joints(self) -> None:
        self.assertEqual(self.kinematics.dof, 6)
        self.assertEqual(self.kinematics.joint_names, (
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ))
        lower, upper = self.kinematics.joint_limits_rad
        self.assertEqual(lower.shape, (6,))
        self.assertEqual(upper.shape, (6,))
        self.assertTrue(np.all(lower < upper))

    def test_forward_zero_configuration(self) -> None:
        pose = self.kinematics.forward(self.joints(np.zeros(6)))
        self.assertEqual(pose.frame_id, "world")
        np.testing.assert_allclose(
            pose.position_m,
            np.array([-0.817, -0.134, 0.063]),
            atol=1e-9,
        )
        self.assertAlmostEqual(float(np.linalg.norm(pose.quaternion_wxyz)), 1.0)

    def test_jacobian_matches_central_position_difference(self) -> None:
        q = np.array([0.2, -0.8, 1.0, -0.4, 0.7, -0.2])
        jacobian = self.kinematics.jacobian(self.joints(q))
        numeric_linear = np.empty((3, 6))
        numeric_angular = np.empty((3, 6))
        epsilon = 1e-7
        for index in range(6):
            step = np.zeros(6)
            step[index] = epsilon
            plus = self.kinematics.forward(self.joints(q + step))
            minus = self.kinematics.forward(self.joints(q - step))
            numeric_linear[:, index] = (
                plus.position_m - minus.position_m
            ) / (2.0 * epsilon)

            rotation_delta = (
                _rotation_matrix(plus.quaternion_wxyz)
                @ _rotation_matrix(minus.quaternion_wxyz).T
            )
            skew = 0.5 * (rotation_delta - rotation_delta.T)
            numeric_angular[:, index] = np.array([
                skew[2, 1],
                skew[0, 2],
                skew[1, 0],
            ]) / (2.0 * epsilon)

        self.assertEqual(jacobian.shape, (6, 6))
        np.testing.assert_allclose(jacobian[:3], numeric_linear, atol=1e-7)
        np.testing.assert_allclose(jacobian[3:], numeric_angular, atol=1e-7)

    def test_inverse_recovers_a_forward_pose_from_nearby_seed(self) -> None:
        expected_q = np.array([0.2, -1.0, 1.1, -0.7, 0.8, 0.3])
        target = self.kinematics.forward(self.joints(expected_q))
        seed_offset = np.array([-0.05, 0.1, -0.1, 0.1, -0.05, -0.1])
        seed = self.joints(expected_q + seed_offset)

        solution = self.kinematics.inverse(target, seed)
        actual = self.kinematics.forward(solution)

        np.testing.assert_allclose(actual.position_m, target.position_m, atol=1e-5)
        self.assertGreaterEqual(
            abs(float(np.dot(actual.quaternion_wxyz, target.quaternion_wxyz))),
            1.0 - 1e-9,
        )

    def test_wrong_joint_order_is_rejected(self) -> None:
        wrong = JointState(tuple(reversed(self.kinematics.joint_names)), np.zeros(6))
        with self.assertRaises(ValueError):
            self.kinematics.forward(wrong)


if __name__ == "__main__":
    unittest.main()
