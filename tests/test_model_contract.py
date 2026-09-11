import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replace_disk_robot.adapters.mujoco import (
    MujocoRobotAdapter,
    MujocoWristFTAdapter,
    load_model,
    reset_home,
)
from replace_disk_robot.core import JointState


class ModelContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.data = load_model()

    def setUp(self) -> None:
        reset_home(self.model, self.data)
        self.data.xfrc_applied[:] = 0

    def test_dimensions_and_named_interfaces(self) -> None:
        self.assertEqual(self.model.nu, 7)
        self.assertEqual(self.model.nsensor, 2)
        self.assertEqual(self.model.nsensordata, 6)
        robot = MujocoRobotAdapter(self.model, self.data)
        sensor = MujocoWristFTAdapter(self.model, self.data)
        self.assertEqual(len(robot.read_joint_state().names), 6)
        self.assertEqual(sensor.read_wrench().frame_id, "wrist_ft_site")

    def test_model_steps_without_nan(self) -> None:
        for _ in range(1000):
            mujoco.mj_step(self.model, self.data)
        self.assertTrue(np.all(np.isfinite(self.data.qpos)))
        self.assertTrue(np.all(np.isfinite(self.data.sensordata)))

    def test_named_joint_and_physical_gripper_commands(self) -> None:
        robot = MujocoRobotAdapter(self.model, self.data)
        names = tuple(reversed(robot.read_joint_state().names))
        positions = np.arange(6, dtype=float) * 0.01
        robot.command_joint_positions(JointState(names, positions))
        expected = np.array([positions[names.index(name)] for name in reversed(names)])
        actuator_ids = [self.model.actuator(name).id for name in (
            "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"
        )]
        self.assertTrue(np.allclose(self.data.ctrl[actuator_ids], expected))

        gripper_id = self.model.actuator("fingers_actuator").id
        robot.command_gripper_opening(0.085)
        self.assertAlmostEqual(float(self.data.ctrl[gripper_id]), 0.0)
        robot.command_gripper_opening(0.0)
        self.assertAlmostEqual(float(self.data.ctrl[gripper_id]), 255.0)

    def test_ft_sensor_responds_to_external_load(self) -> None:
        reset_home(self.model, self.data)
        robot = MujocoRobotAdapter(self.model, self.data)
        robot.command_arm(robot.arm_position())
        for _ in range(800):
            mujoco.mj_step(self.model, self.data)
        sensor = MujocoWristFTAdapter(self.model, self.data)
        sensor.tare()
        self.data.xfrc_applied[self.model.body("g_base").id, :3] = [12, 0, 0]
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
        self.assertGreater(np.linalg.norm(sensor.wrench()[:3]), 10.0)


if __name__ == "__main__":
    unittest.main()
