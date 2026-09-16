import unittest

import numpy as np

from replace_disk_robot.adapters.mujoco import (
    MujocoCollisionChecker,
    MujocoRobotAdapter,
    load_model,
    reset_home,
)
from replace_disk_robot.core import JointState


class MujocoCollisionCheckerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.data = load_model()
        reset_home(cls.model, cls.data)

    def setUp(self) -> None:
        reset_home(self.model, self.data)

    def test_home_is_collision_free(self) -> None:
        checker = MujocoCollisionChecker(self.model, self.data)
        robot = MujocoRobotAdapter(self.model, self.data)
        state = robot.read_joint_state()
        self.assertTrue(checker.is_collision_free(state))
        self.assertGreater(checker.minimum_distance(state), 0.0)

    def test_arm_extension_collides_with_environment(self) -> None:
        checker = MujocoCollisionChecker(self.model, self.data)
        robot = MujocoRobotAdapter(self.model, self.data)
        names = robot.read_joint_state().names
        state = JointState(names, [0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
        self.assertFalse(checker.is_collision_free(state))
        self.assertLess(checker.minimum_distance(state), 0.0)

    def test_held_drive_socket_collision_is_not_ignored(self) -> None:
        from replace_disk_robot.adapters.mujoco.insertion_validation import solve_center
        checker = MujocoCollisionChecker(self.model, self.data)
        robot = MujocoRobotAdapter(self.model, self.data)
        target = self.data.site("drive_center").xpos.copy() + [.015,.001,0]
        q = solve_center(self.model, robot.arm_position(), target)
        self.assertFalse(checker.is_collision_free(JointState(robot.read_joint_state().names, q)))


    def test_safety_margin_is_part_of_collision_free_decision(self) -> None:
        robot = MujocoRobotAdapter(self.model, self.data)
        state = robot.read_joint_state()
        base_checker = MujocoCollisionChecker(self.model, self.data)
        clearance = base_checker.minimum_distance(state)
        margin_checker = MujocoCollisionChecker(
            self.model,
            self.data,
            safety_margin_m=clearance + 1e-3,
        )
        self.assertTrue(base_checker.is_collision_free(state))
        self.assertFalse(margin_checker.is_collision_free(state))


    def test_separate_planning_data_does_not_mutate_execution_state(self) -> None:
        robot = MujocoRobotAdapter(self.model, self.data)
        checker = MujocoCollisionChecker(self.model, self.data)
        before = self.data.qpos.copy()
        state = JointState(robot.read_joint_state().names, [0.5, 0.5, 0.5, 0.0, 0.0, 0.0])

        checker.is_collision_free(state)
        checker.minimum_distance(state)

        np.testing.assert_allclose(self.data.qpos, before)


if __name__ == "__main__":
    unittest.main()
