"""Offline tests for the six-axis F/T payload gravity identifier."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tool"))

from identify_ft_payload import StaticPose, build_warnings, identify_payload  # noqa: E402


def _synthetic_poses(
    count: int = 18,
    *,
    gravity: np.ndarray | None = None,
) -> tuple[list[StaticPose], dict[str, np.ndarray | float]]:
    rng = np.random.default_rng(7)
    expected = {
        "mass": 2.4,
        "gravity": np.array([0.0, 0.0, -2.4 * 9.80665]) if gravity is None else np.asarray(gravity, dtype=float),
        "force_bias": np.array([0.21, -0.12, 0.08]),
        "com": np.array([0.035, -0.018, 0.11]),
        "torque_bias": np.array([0.012, -0.009, 0.006]),
    }
    expected["mass"] = float(np.linalg.norm(expected["gravity"])) / 9.80665
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


def test_tilted_gravity_vector_is_fitted_not_assumed_vertical() -> None:
    """A tilted chassis must be recovered, not forced towards base -Z."""

    mass = 1.7
    tilt_deg, azimuth_deg = 16.0, 140.0
    direction = np.array([
        np.sin(np.deg2rad(tilt_deg)) * np.cos(np.deg2rad(azimuth_deg)),
        np.sin(np.deg2rad(tilt_deg)) * np.sin(np.deg2rad(azimuth_deg)),
        -np.cos(np.deg2rad(tilt_deg)),
    ])
    poses, expected = _synthetic_poses(gravity=mass * 9.80665 * direction)

    result = identify_payload(poses, gravity_mode="free")

    assert result["mass_kg"] == pytest.approx(mass, abs=1e-10)
    assert result["gravity_tilt_deg"] == pytest.approx(tilt_deg, abs=1e-9)
    assert result["gravity_tilt_azimuth_deg"] == pytest.approx(azimuth_deg, abs=1e-9)
    np.testing.assert_allclose(
        result["signed_gravity_force_base_n"], expected["gravity"], atol=1e-9
    )
    np.testing.assert_allclose(result["gravity_down_base_unit"], direction, atol=1e-12)
    # The legacy key must keep reporting the same tilt.
    assert result["vertical_alignment_error_deg"] == pytest.approx(tilt_deg, abs=1e-9)


def test_gravity_tilt_reporting_is_sign_convention_agnostic() -> None:
    """A flipped sensor sign convention must not flip the reported tilt."""

    direction = np.array([0.0, np.sin(np.deg2rad(20.0)), -np.cos(np.deg2rad(20.0))])
    poses, _expected = _synthetic_poses(gravity=-1.7 * 9.80665 * direction)

    result = identify_payload(poses, gravity_mode="free")

    assert result["gravity_tilt_deg"] == pytest.approx(20.0, abs=1e-9)
    assert result["gravity_tilt_azimuth_deg"] == pytest.approx(90.0, abs=1e-9)


def test_build_warnings_accepts_a_tilted_chassis_below_the_limit() -> None:
    direction = np.array([
        np.sin(np.deg2rad(25.0)), 0.0, -np.cos(np.deg2rad(25.0))
    ])
    poses, _expected = _synthetic_poses(gravity=1.7 * 9.80665 * direction)
    result = identify_payload(poses, gravity_mode="free")

    assert build_warnings(result, max_tilt_deg=30.0, max_com_m=0.5) == []
    warnings = build_warnings(result, max_tilt_deg=10.0, max_com_m=0.5)
    assert len(warnings) == 1 and "25.0 deg away from base -Z" in warnings[0]


@pytest.mark.parametrize('sign', [1., -1.])
def test_bounded_gravity_recovers_valid_small_tilt_and_both_sensor_signs(sign):
    direction = np.array([np.sin(np.deg2rad(1.8)), 0., -np.cos(np.deg2rad(1.8))])
    poses, expected = _synthetic_poses(gravity=sign*2.4*9.80665*direction)
    result = identify_payload(poses)
    np.testing.assert_allclose(result['signed_gravity_force_base_n'], expected['gravity'], atol=1e-10)
    assert result['gravity_tilt_deg'] == pytest.approx(1.8)
    assert result['gravity_direction_model'] == 'bounded'
    assert result['gravity_tilt_limit_deg'] == 4.0


def test_bad_free_tilt_is_constrained_in_fit_and_remains_visible_in_diagnostics():
    direction = np.array([np.sin(np.deg2rad(52)), 0., -np.cos(np.deg2rad(52))])
    poses, expected = _synthetic_poses(gravity=2.4*9.80665*direction)
    result = identify_payload(poses, gravity_tilt_limit_deg=2)
    assert result['gravity_tilt_deg'] <= 2 + 1e-9
    assert result['unconstrained_gravity_tilt_deg'] == pytest.approx(52)
    assert result['fit']['force_rms_n'] > .1
    assert result['fit']['unconstrained_force_rms_n'] < 1e-10
    assert any('model assumption' in warning for warning in build_warnings(result, max_tilt_deg=30, max_com_m=.5))
    # Bias must be refitted: clipping only h and keeping the old bias is wrong.
    h = np.asarray(result['signed_gravity_force_base_n'])
    assert np.linalg.norm(np.asarray(result['force_bias_sensor_n'])-expected['force_bias']) > .01
    assert np.linalg.norm(h[:2]) <= abs(h[2])*np.tan(np.deg2rad(2)) + 1e-10


def test_zero_tilt_limit_refits_signed_vertical_load_and_bias():
    direction = np.array([.1, .1, -1.]);direction/=np.linalg.norm(direction)
    poses, _ = _synthetic_poses(gravity=2.4*9.80665*direction)
    result = identify_payload(poses, gravity_tilt_limit_deg=0)
    assert result['signed_gravity_force_base_n'][:2] == [0., 0.]
    assert result['gravity_tilt_deg'] == 0
    assert result['fit']['force_rms_n'] > 0


@pytest.mark.parametrize('limit', [-1, 90, float('nan')])
def test_invalid_tilt_bounds_rejected(limit):
    poses, _ = _synthetic_poses()
    with pytest.raises(ValueError, match='gravity_tilt_limit_deg'):
        identify_payload(poses, gravity_tilt_limit_deg=limit)


def test_stale_tool0_repair_restores_sensor_orientation_and_lever():
    from identify_ft_payload import repair_stale_tool0_extrinsics
    poses, _ = _synthetic_poses()
    flip = np.diag([-1., -1., 1.])
    transforms = {'rotation_sensor_to_tool': np.eye(3), 'tool_to_sensor_m': np.array([0., -.05, -.4])}
    corrupted = [StaticPose(p.capture_id,p.samples,p.rotation_base_sensor@flip,p.force_sensor_n,p.torque_sensor_nm) for p in poses]
    repaired, corrected = repair_stale_tool0_extrinsics(corrupted,transforms)
    for before,after in zip(poses,repaired):
        np.testing.assert_allclose(before.rotation_base_sensor,after.rotation_base_sensor)
    np.testing.assert_allclose(corrected['rotation_sensor_to_tool'],flip)
    np.testing.assert_allclose(corrected['tool_to_sensor_m'],[0., .05, -.4])
    assert identify_payload(repaired)['mass_kg'] == pytest.approx(2.4)
    with pytest.raises(ValueError, match='requires'):
        repair_stale_tool0_extrinsics(poses,{})


def test_repaired_fit_plot_uses_matching_sensor_orientation():
    from plot_ft_gravity_data import _compensate
    poses, expected = _synthetic_poses()
    rotations = np.array([p.rotation_base_sensor for p in poses])
    flip = np.diag([-1., -1., 1.])
    data = {'rotation_base_sensor': rotations@flip,
            'rotation_sensor_to_tool_recorded': np.broadcast_to(np.eye(3),rotations.shape),
            'raw_force_n': np.array([p.force_sensor_n for p in poses]),
            'raw_torque_nm': np.array([p.torque_sensor_nm for p in poses])}
    fit = {'gravity_force_base_n': expected['gravity'], 'center_of_mass_sensor_m': expected['com'],
           'force_bias_sensor_n': expected['force_bias'], 'torque_bias_sensor_nm': expected['torque_bias'],
           'stale_tool0_extrinsics_repaired': True}
    f,t = _compensate(data,fit)
    np.testing.assert_allclose(f,0,atol=1e-12)
    np.testing.assert_allclose(t,0,atol=1e-12)
    data.pop('rotation_sensor_to_tool_recorded')
    with pytest.raises(ValueError,match='original CSV'):
        _compensate(data,fit)


def test_constrained_solver_matches_independent_boundary_grid():
    from identify_ft_payload import _bounded_force_solution
    poses,_ = _synthetic_poses(gravity=np.array([8.,-3.,-5.]))
    a=np.vstack([np.hstack((p.rotation_base_sensor.T,np.eye(3))) for p in poses])
    y=np.array([p.force_sensor_n for p in poses]).ravel()
    solution=_bounded_force_solution(a,y,2)
    score=np.sum((a@solution-y)**2)
    candidates=[]
    for azimuth in np.linspace(0,2*np.pi,1500,endpoint=False):
        unit=np.array([np.sin(np.deg2rad(2))*np.cos(azimuth),np.sin(np.deg2rad(2))*np.sin(azimuth),-np.cos(np.deg2rad(2))])
        design=np.column_stack((a[:,:3]@unit,a[:,3:]))
        fitted=np.linalg.lstsq(design,y,rcond=None)[0]
        candidates.append(np.sum((design@fitted-y)**2))
    assert score <= min(candidates)+1e-8
    assert min(candidates)-score < 1e-3
