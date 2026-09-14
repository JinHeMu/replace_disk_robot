import importlib.util
import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from keyboard_servo import JAKA_ACTUATORS, ServoDemo
from replace_disk_robot.adapters.mujoco import load_model, reset_keyframe
from replace_disk_robot.adapters.mujoco.interfaces import MujocoRobotAdapter
from replace_disk_robot.core import JointState
from replace_disk_robot.core.rotation import rotation_matrix
from replace_disk_robot.kinematics.jaka import (
    JAKA_JOINT_NAMES,
    PROJECT_JAKA_URDF,
    JakaKinematics,
)


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
class JakaKinematicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kinematics = JakaKinematics(PROJECT_JAKA_URDF)

    def joints(self, position_rad) -> JointState:
        return JointState(self.kinematics.joint_names, position_rad)

    def test_model_contains_only_the_six_arm_dofs(self) -> None:
        self.assertEqual(self.kinematics.dof, 6)
        self.assertEqual(self.kinematics.joint_names, JAKA_JOINT_NAMES)
        self.assertEqual(self.kinematics.base_frame, "jaka_base_link")
        self.assertEqual(self.kinematics.end_effector_frame, "tool0")
        self.assertEqual(self.kinematics._model.nq, 6)
        self.assertEqual(self.kinematics._model.nv, 6)
        lower, upper = self.kinematics.joint_limits_rad
        np.testing.assert_allclose(lower, [-6.28, -1.48, -3.05, -1.48, -6.28, -6.28])
        np.testing.assert_allclose(upper, [6.28, 4.62, 3.05, 4.62, 6.28, 6.28])

    def test_forward_zero_configuration(self) -> None:
        pose = self.kinematics.forward(self.joints(np.zeros(6)))
        self.assertEqual(pose.frame_id, "jaka_base_link")
        np.testing.assert_allclose(
            pose.position_m,
            [0.7985, -0.29449958, 0.0066361],
            atol=1e-8,
        )
        self.assertAlmostEqual(float(np.linalg.norm(pose.quaternion_wxyz)), 1.0)

    def test_jacobian_matches_central_difference(self) -> None:
        q = np.array([-0.4, 1.4, -1.2, 1.2, 1.0, 0.3])
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
                rotation_matrix(plus.quaternion_wxyz)
                @ rotation_matrix(minus.quaternion_wxyz).T
            )
            skew = 0.5 * (rotation_delta - rotation_delta.T)
            numeric_angular[:, index] = np.array([
                skew[2, 1], skew[0, 2], skew[1, 0],
            ]) / (2.0 * epsilon)

        self.assertEqual(jacobian.shape, (6, 6))
        np.testing.assert_allclose(jacobian[:3], numeric_linear, atol=1e-7)
        np.testing.assert_allclose(jacobian[3:], numeric_angular, atol=1e-7)

    def test_inverse_recovers_forward_pose_from_nearby_seed(self) -> None:
        expected_q = np.array([-0.5, 1.5, -1.4, 1.4, 1.3, 0.25])
        target = self.kinematics.forward(self.joints(expected_q))
        seed = self.joints(expected_q + [0.05, -0.05, 0.05, -0.05, 0.05, 0.05])
        solution = self.kinematics.inverse(target, seed)
        actual = self.kinematics.forward(solution)
        np.testing.assert_allclose(actual.position_m, target.position_m, atol=1e-5)
        self.assertGreaterEqual(
            abs(float(np.dot(actual.quaternion_wxyz, target.quaternion_wxyz))),
            1.0 - 1e-9,
        )

    def test_low_keyframe_fk_matches_mujoco_virtual_tool0(self) -> None:
        model, data = load_model("jaka")
        reset_keyframe(model, data, "low")
        adapter = MujocoRobotAdapter(
            model,
            data,
            joint_names=JAKA_JOINT_NAMES,
            actuator_names=JAKA_ACTUATORS,
            gripper_actuator_name=None,
        )
        pose = self.kinematics.forward(adapter.read_joint_state())

        base = data.body("jaka_base_link")
        tool = data.body("tool0_and_camera_link")
        world_rotation_base = base.xmat.reshape(3, 3)
        world_rotation_tool = tool.xmat.reshape(3, 3)
        expected_position = world_rotation_base.T @ (
            tool.xpos
            + world_rotation_tool @ np.array([0.0, 0.0, 0.27])
            - base.xpos
        )
        expected_rotation = world_rotation_base.T @ world_rotation_tool

        np.testing.assert_allclose(pose.position_m, expected_position, atol=1e-8)
        np.testing.assert_allclose(
            rotation_matrix(pose.quaternion_wxyz), expected_rotation, atol=2e-4
        )

        actuator_ids = np.array([model.actuator(name).id for name in JAKA_ACTUATORS])
        np.testing.assert_allclose(data.ctrl[actuator_ids], adapter.arm_position())

    def test_up_keyframe_matches_recorded_jaka_startup_pose(self) -> None:
        expected = np.array(
            [0.0551, 1.5901, -0.0254, 1.5356, 3.2008, 0.7079],
            dtype=float,
        )
        model, data = load_model("jaka")
        reset_keyframe(model, data, "up")
        adapter = MujocoRobotAdapter(
            model,
            data,
            joint_names=JAKA_JOINT_NAMES,
            actuator_names=JAKA_ACTUATORS,
            gripper_actuator_name=None,
        )
        np.testing.assert_allclose(adapter.arm_position(), expected, atol=1e-12)

        actuator_ids = np.array([model.actuator(name).id for name in JAKA_ACTUATORS])
        np.testing.assert_allclose(data.ctrl[actuator_ids], expected, atol=1e-12)

    def test_keyboard_servo_selects_jaka_tool_frame_without_base_dofs(self) -> None:
        app = ServoDemo(model_name="jaka")
        self.assertEqual(app.robot.read_joint_state().names, JAKA_JOINT_NAMES)
        self.assertEqual(app.command_frame_mode, "tool")
        self.assertEqual(app.command_frame, "tool0")
        self.assertEqual(app.base_frame, "jaka_base_link")
        self.assertEqual(app.ft.read_wrench().frame_id, "tcp_fts_site")
        self.assertEqual(app.keyframe_name, "low")

        initial = app.kinematics.forward(app.robot.read_joint_state())
        tool_x_in_base = rotation_matrix(initial.quaternion_wxyz) @ np.array([1.0, 0.0, 0.0])
        app.keys.press("r")
        for _ in range(25):
            app.tick()
        end = app.kinematics.forward(app.robot.read_joint_state())
        displacement = end.position_m - initial.position_m
        self.assertGreater(float(displacement @ tool_x_in_base), 0.0005)
        self.assertIsNone(app.servo.fault)

    def test_keyboard_servo_can_still_use_jaka_base_frame(self) -> None:
        app = ServoDemo(model_name="jaka", command_frame="base")
        self.assertEqual(app.command_frame_mode, "base")
        self.assertEqual(app.command_frame, "jaka_base_link")
        self.assertEqual(app.keys.base_frame, "jaka_base_link")

    def test_wrong_joint_order_is_rejected(self) -> None:
        wrong = JointState(tuple(reversed(JAKA_JOINT_NAMES)), np.zeros(6))
        with self.assertRaises(ValueError):
            self.kinematics.forward(wrong)


if __name__ == "__main__":
    unittest.main()
