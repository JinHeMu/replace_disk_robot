import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replace_disk_robot.control import AdmittanceConfig, AdmittanceController
from replace_disk_robot.core import AdmittanceControllerPort, AdmittanceState, Pose, Wrench


FRAME = "world"


def nominal(position=(0.0, 0.0, 0.0)) -> Pose:
    return Pose(FRAME, position, [1.0, 0.0, 0.0, 0.0])


def wrench(force=(0.0, 0.0, 0.0), torque=(0.0, 0.0, 0.0)) -> Wrench:
    return Wrench(FRAME, force, torque)


def config(**overrides) -> AdmittanceConfig:
    values = dict(
        frame_id=FRAME,
        mass=1.0,
        damping=10.0,
        stiffness=100.0,
        max_offset=1.0,
        max_velocity=10.0,
        max_dt_s=0.01,
    )
    values.update(overrides)
    return AdmittanceConfig(**values)


class AdmittanceControllerTest(unittest.TestCase):
    def test_config_broadcasts_scalars_and_rejects_invalid_values(self) -> None:
        cfg = config()
        self.assertEqual(cfg.mass.shape, (6,))
        self.assertEqual(cfg.max_offset.shape, (6,))
        self.assertTrue(np.all(cfg.mass == 1.0))

        with self.assertRaises(ValueError):
            config(mass=0.0)
        with self.assertRaises(ValueError):
            config(damping=np.zeros(6))
        with self.assertRaises(ValueError):
            config(stiffness=-1.0)
        with self.assertRaises(ValueError):
            config(max_offset=0.0)
        with self.assertRaises(ValueError):
            config(max_velocity=[1.0] * 5)
        with self.assertRaises(ValueError):
            config(max_dt_s=0.0)
        with self.assertRaises(ValueError):
            config(frame_id="")

    def test_core_exports_admittance_contract(self) -> None:
        state = AdmittanceState.zero(FRAME)
        np.testing.assert_array_equal(state.offset, np.zeros(6))
        np.testing.assert_array_equal(state.velocity, np.zeros(6))
        self.assertEqual(state.frame_id, FRAME)
        self.assertTrue(hasattr(AdmittanceControllerPort, "reset"))
        self.assertTrue(hasattr(AdmittanceControllerPort, "update"))
        self.assertTrue(hasattr(AdmittanceControllerPort, "state"))

    def test_zero_wrench_keeps_nominal_pose(self) -> None:
        controller = AdmittanceController(config())
        target = nominal((0.1, 0.2, 0.3))

        corrected = controller.update(target, wrench(), dt_s=0.01)

        np.testing.assert_allclose(corrected.position_m, target.position_m)
        np.testing.assert_allclose(corrected.quaternion_wxyz, target.quaternion_wxyz)
        self.assertEqual(corrected.frame_id, FRAME)

    def test_constant_force_reaches_stiffness_offset(self) -> None:
        controller = AdmittanceController(
            config(mass=1.0, damping=10.0, stiffness=100.0, max_offset=1.0)
        )
        target = nominal()

        for _ in range(2000):
            corrected = controller.update(target, wrench(force=(10.0, 0.0, 0.0)), dt_s=0.01)

        self.assertAlmostEqual(controller.state().offset[0], 0.1, places=3)
        np.testing.assert_allclose(controller.state().offset[1:], np.zeros(5), atol=1e-12)
        self.assertAlmostEqual(corrected.position_m[0], 0.1, places=3)

    def test_rotation_offset_uses_parent_frame_rotation_vector(self) -> None:
        controller = AdmittanceController(
            config(
                mass=1.0,
                damping=1.0,
                stiffness=0.0,
                max_offset=np.ones(6),
                max_velocity=np.ones(6),
            )
        )
        target = nominal()

        corrected = controller.update(target, wrench(torque=(0.0, 0.0, 1.0)), dt_s=0.01)

        self.assertGreater(controller.state().offset[5], 0.0)
        self.assertGreater(corrected.quaternion_wxyz[3], 0.0)
        np.testing.assert_allclose(corrected.quaternion_wxyz[1:3], np.zeros(2), atol=1e-12)

    def test_reset_clears_state(self) -> None:
        controller = AdmittanceController(config())
        target = nominal()
        for _ in range(100):
            controller.update(target, wrench(force=(10.0, 0.0, 0.0)), dt_s=0.01)
        self.assertGreater(abs(controller.state().offset[0]), 0.0)

        controller.reset(target)

        np.testing.assert_array_equal(controller.state().offset, np.zeros(6))
        np.testing.assert_array_equal(controller.state().velocity, np.zeros(6))
        corrected = controller.update(target, wrench(), dt_s=0.01)
        np.testing.assert_allclose(corrected.position_m, target.position_m)

    def test_frame_mismatch_is_rejected(self) -> None:
        controller = AdmittanceController(config())
        target = nominal()
        other_pose = Pose("base", [0, 0, 0], [1, 0, 0, 0])
        other_wrench = Wrench("tool", [0, 0, 0], [0, 0, 0])

        with self.assertRaises(ValueError):
            controller.reset(other_pose)
        with self.assertRaises(ValueError):
            controller.update(other_pose, wrench(), dt_s=0.01)
        with self.assertRaises(ValueError):
            controller.update(target, other_wrench, dt_s=0.01)

    def test_invalid_dt_is_rejected(self) -> None:
        controller = AdmittanceController(config(max_dt_s=0.01))
        target = nominal()

        for dt_s in (0.0, -0.01, float("nan"), float("inf"), 0.011):
            with self.assertRaises(ValueError):
                controller.update(target, wrench(), dt_s=dt_s)

    def test_offset_limit_stops_outward_motion(self) -> None:
        cfg = config(
            mass=1.0,
            damping=1.0,
            stiffness=0.0,
            max_offset=np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1]),
            max_velocity=np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0]),
        )
        controller = AdmittanceController(cfg)

        for _ in range(1000):
            controller.update(nominal(), wrench(force=(1000.0, 0.0, 0.0)), dt_s=0.01)

        state = controller.state()
        self.assertLessEqual(abs(state.offset[0]), cfg.max_offset[0])
        self.assertAlmostEqual(state.offset[0], cfg.max_offset[0])
        self.assertLessEqual(abs(state.velocity[0]), cfg.max_velocity[0])

    def test_state_returns_copies_in_controller_frame(self) -> None:
        controller = AdmittanceController(config())
        state = controller.state()
        self.assertEqual(state.frame_id, FRAME)
        state.offset[0] = 123.0

        self.assertEqual(controller.state().offset[0], 0.0)


if __name__ == "__main__":
    unittest.main()
