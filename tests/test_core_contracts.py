import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replace_disk_robot.core import JointState, Pose, TrajectoryPoint, Wrench
from replace_disk_robot.task import TaskPrimitive, TaskState


class CoreContractsTest(unittest.TestCase):
    def test_core_does_not_import_runtime_backends(self) -> None:
        package_dir = Path(__file__).resolve().parents[1] / "src" / "replace_disk_robot"
        root_source = (package_dir / "__init__.py").read_text()
        self.assertNotIn("adapters.mujoco", root_source)
        core_dir = package_dir / "core"
        source = "\n".join(path.read_text() for path in core_dir.glob("*.py"))
        self.assertNotIn("import mujoco", source)
        self.assertNotIn("import rclpy", source)
        self.assertNotIn("import moveit", source)

    def test_pose_normalizes_quaternion_and_keeps_frame(self) -> None:
        pose = Pose("world", [1, 2, 3], [2, 0, 0, 0])
        self.assertEqual(pose.frame_id, "world")
        self.assertTrue(np.allclose(pose.quaternion_wxyz, [1, 0, 0, 0]))

    def test_wrench_order_is_force_then_torque(self) -> None:
        wrench = Wrench("wrist_ft_site", [1, 2, 3], [4, 5, 6])
        self.assertTrue(np.array_equal(wrench.as_vector(), [1, 2, 3, 4, 5, 6]))

    def test_invalid_contract_data_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Pose("", [0, 0, 0], [1, 0, 0, 0])
        with self.assertRaises(ValueError):
            JointState(("joint",), [np.nan])
        with self.assertRaises(ValueError):
            TrajectoryPoint(-0.1, [0])

    def test_task_vocabulary_is_stable(self) -> None:
        self.assertEqual(TaskPrimitive.INSERT.value, "insert")
        self.assertEqual(TaskState.PRESS_LATCH.value, "press_latch")
        self.assertEqual(TaskState.FAILED.value, "failed")


if __name__ == "__main__":
    unittest.main()
