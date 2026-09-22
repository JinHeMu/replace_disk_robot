"""Contracts for keyboard nominal-pose generation plus admittance."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replace_disk_robot.control import (
    AdmittanceConfig,
    AdmittanceController,
    KeyboardAdmittanceController,
    MotionReferenceConfig,
)
from replace_disk_robot.core import CartesianJog, Pose, Wrench
from replace_disk_robot.core.rotation import rotation_matrix


def pose(position=(0.0, 0.0, 0.0), quaternion=(1.0, 0.0, 0.0, 0.0)) -> Pose:
    return Pose("world", position, quaternion)


def admittance(**overrides) -> AdmittanceController:
    values = dict(
        frame_id="world",
        mass=1.0,
        damping=10.0,
        stiffness=100.0,
        max_velocity=0.5,
        max_dt_s=0.05,
    )
    values.update(overrides)
    return AdmittanceController(AdmittanceConfig(**values))


def test_keyboard_integration_in_base_and_intrinsic_tool_axes() -> None:
    motion = KeyboardAdmittanceController(
        admittance(stiffness=0.0),
        pose(),
        MotionReferenceConfig(),
    )
    measured = pose()
    linear_jog = CartesianJog("world", [0.1, 0.0, 0.0], [0.0, 0.0, 0.0])
    for _ in range(10):
        corrected = motion.update(
            linear_jog, Wrench("world", [0.0] * 3, [0.0] * 3), measured, 0.01
        )
    np.testing.assert_allclose(motion.nominal_pose.position_m, [0.01, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(corrected.position_m, motion.nominal_pose.position_m)

    angular_jog = CartesianJog("world", [0.0] * 3, [0.0, 0.0, 1.0])
    for _ in range(10):
        motion.update(
            angular_jog, Wrench("world", [0.0] * 3, [0.0] * 3), measured, 0.01
        )
    expected_rotation = rotation_matrix(
        [np.cos(0.05), 0.0, 0.0, np.sin(0.05)]
    )
    np.testing.assert_allclose(
        rotation_matrix(motion.nominal_pose.quaternion_wxyz),
        expected_rotation,
        atol=1e-12,
    )


def test_nominal_pose_is_not_lead_limited() -> None:
    motion = KeyboardAdmittanceController(
        admittance(stiffness=0.0),
        pose(),
        MotionReferenceConfig(),
    )
    measured = pose()
    jog = CartesianJog("world", [1.0, 0.0, 0.0], [0.0] * 3)
    for _ in range(100):
        motion.update(jog, Wrench("world", [0.0] * 3, [0.0] * 3), measured, 0.01)

    np.testing.assert_allclose(motion.nominal_pose.position_m, [1.0, 0.0, 0.0], atol=1e-12)


def test_admittance_axes_mask_blocks_rotation_compliance() -> None:
    motion = KeyboardAdmittanceController(
        admittance(),
        pose(),
        MotionReferenceConfig(
            enabled_axes=(True, True, True, False, False, False),
        ),
    )
    measured = pose()

    corrected = motion.update(
        CartesianJog("world", [0.0] * 3, [0.0] * 3),
        Wrench("world", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]),
        measured,
        0.01,
    )

    np.testing.assert_allclose(corrected.position_m, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(corrected.quaternion_wxyz, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(motion.state().offset, np.zeros(6), atol=1e-12)


def test_external_force_moves_corrected_pose_and_state_is_exposed() -> None:
    controller = admittance()
    motion = KeyboardAdmittanceController(
        controller,
        pose(),
        MotionReferenceConfig(enabled_axes=(True, True, True, False, False, False)),
    )
    measured = pose()
    for _ in range(100):
        corrected = motion.update(
            CartesianJog("world", [0.0] * 3, [0.0] * 3),
            Wrench("world", [5.0, 0.0, 0.0], [0.0] * 3),
            measured,
            0.01,
        )

    self_state = motion.state()
    assert self_state.frame_id == "world"
    assert self_state.offset[0] > 0.0
    assert corrected.position_m[0] > 0.0
    np.testing.assert_allclose(motion.corrected_pose.position_m[0], corrected.position_m[0])


def test_reset_and_validation_keep_frames_explicit() -> None:
    motion = KeyboardAdmittanceController(admittance(), pose())
    with pytest.raises(ValueError):
        motion.update(
            CartesianJog("tool", [0.0] * 3, [0.0] * 3),
            Wrench("world", [0.0] * 3, [0.0] * 3),
            pose(),
            0.01,
        )
    with pytest.raises(ValueError):
        motion.update(
            CartesianJog("world", [0.0] * 3, [0.0] * 3),
            Wrench("tool", [0.0] * 3, [0.0] * 3),
            pose(),
            0.01,
        )
    with pytest.raises(ValueError):
        motion.update(
            CartesianJog("world", [0.0] * 3, [0.0] * 3),
            Wrench("world", [0.0] * 3, [0.0] * 3),
            Pose("base", [0.0] * 3, [1.0, 0.0, 0.0, 0.0]),
            0.01,
        )
    with pytest.raises(ValueError):
        motion.update(
            CartesianJog("world", [0.0] * 3, [0.0] * 3),
            Wrench("world", [0.0] * 3, [0.0] * 3),
            pose(),
            0.5,
        )

    motion.update(
        CartesianJog("world", [0.1, 0.0, 0.0], [0.0] * 3),
        Wrench("world", [1.0, 0.0, 0.0], [0.0] * 3),
        pose(),
        0.01,
    )
    assert motion.nominal_pose.position_m[0] > 0.0
    motion.reset(pose([0.3, 0.0, 0.0]))
    np.testing.assert_allclose(motion.nominal_pose.position_m, [0.3, 0.0, 0.0])
    np.testing.assert_allclose(motion.nominal_pose.position_m, motion.corrected_pose.position_m)
    np.testing.assert_allclose(motion.state().offset, np.zeros(6))

    with pytest.raises(ValueError):
        motion.set_enabled_axes((True, False))


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        MotionReferenceConfig(enabled_axes=(True,) * 5)
    with pytest.raises(ValueError):
        MotionReferenceConfig(max_dt_s=0.0)
