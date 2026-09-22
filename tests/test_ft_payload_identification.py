"""Offline tests for the six-axis F/T payload gravity identifier."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tool"))

from identify_ft_payload import StaticPose, identify_payload  # noqa: E402


def _synthetic_poses(count: int = 18) -> tuple[list[StaticPose], dict[str, np.ndarray | float]]:
    rng = np.random.default_rng(7)
    expected = {
        "mass": 2.4,
        "gravity": np.array([0.0, 0.0, -2.4 * 9.80665]),
        "force_bias": np.array([0.21, -0.12, 0.08]),
        "com": np.array([0.035, -0.018, 0.11]),
        "torque_bias": np.array([0.012, -0.009, 0.006]),
    }
    poses = []
    for capture_id in range(count):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1.0
        gravity_sensor = rotation.T @ expected["gravity"]
        force = gravity_sensor + expected["force_bias"]
        torque = np.cross(expected["com"], gravity_sensor) + expected["torque_bias"]
        poses.append(StaticPose(capture_id, 125, rotation, force, torque))
    return poses, expected


def test_identify_payload_recovers_mass_com_and_bias() -> None:
    poses, expected = _synthetic_poses()

    result = identify_payload(poses)

    assert result["mass_kg"] == pytest.approx(expected["mass"], abs=1e-10)
    np.testing.assert_allclose(result["center_of_mass_sensor_m"], expected["com"], atol=1e-10)
    np.testing.assert_allclose(result["force_bias_sensor_n"], expected["force_bias"], atol=1e-10)
    np.testing.assert_allclose(result["torque_bias_sensor_nm"], expected["torque_bias"], atol=1e-10)
    assert result["fit"]["force_rms_n"] < 1e-10
    assert result["fit"]["torque_rms_nm"] < 1e-10


def test_identify_payload_rejects_unexcited_or_too_small_dataset() -> None:
    poses, _expected = _synthetic_poses(count=5)
    with pytest.raises(ValueError, match="at least 6"):
        identify_payload(poses)

    poses, _expected = _synthetic_poses(count=6)
    fixed_rotation = poses[0].rotation_base_sensor
    unexcited = [
        StaticPose(
            pose.capture_id,
            pose.samples,
            fixed_rotation,
            pose.force_sensor_n,
            pose.torque_sensor_nm,
        )
        for pose in poses
    ]
    with pytest.raises(ValueError, match="rank deficient"):
        identify_payload(unexcited)
