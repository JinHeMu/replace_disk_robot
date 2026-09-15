from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replace_disk_robot.core.types import Wrench

from rl.config import RLConfig
from rl.features import FEATURE_NAMES, FeatureSnapshot, build_observation, observation_schema
from rl.policy import LinearPolicy, ScriptedPolicy, ZeroPolicy
from rl.residual import ACTION_NAMES, ResidualLimiter, compose_taskspace_command
from rl.reward import RewardInputs, compute_reward


def _snapshot() -> FeatureSnapshot:
    return FeatureSnapshot(
        depth_m=0.02,
        lateral_offset_m=np.array([0.0002, -0.0001]),
        pitch_yaw_rad=np.array([0.001, -0.002]),
        tcp_velocity_m_s=np.array([0.002, 0.0001, -0.0001]),
        tcp_angular_velocity_rad_s=np.array([0.0, 0.001]),
        wrench=Wrench("socket_entry", np.array([-4.0, 0.5, 0.2]), np.array([0.1, 0.2, 0.0])),
        wrench_rate=Wrench("socket_entry", np.array([-1.0, 0.1, 0.0]), np.array([0.0, 0.05, 0.0])),
        baseline_command=np.array([0.002, 0.0, 0.0, 0.0, 0.0]),
        last_residual=np.zeros(5),
    )


def test_observation_schema_is_fixed_and_finite() -> None:
    cfg = RLConfig()
    observation = build_observation(_snapshot(), cfg.observation)
    assert observation.shape == (32,)
    assert len(FEATURE_NAMES) == 32
    assert np.all(np.isfinite(observation))
    assert np.max(np.abs(observation)) <= cfg.observation.clip_abs + 1e-12
    schema = observation_schema()
    assert schema["version"] == cfg.observation.schema_version
    assert schema["names"] == list(FEATURE_NAMES)


def test_residual_limiter_is_bounded_and_rate_limits() -> None:
    limiter = ResidualLimiter()
    limited, info = limiter.limit(np.ones(5, dtype=float), 0.05, now_s=0.0)
    assert limited.shape == (5,)
    assert np.all(np.abs(limited) <= np.asarray(limiter.config.action_limits) + 1e-15)
    # A second opposite command must be rate limited relative to the first.
    opposite, opposite_info = limiter.limit(-np.ones(5, dtype=float), 0.05, now_s=0.05)
    assert opposite_info.rate_clipped
    assert np.all(np.abs(opposite - limited) <= np.asarray(limiter.config.max_action_delta) + 1e-15)


def test_compose_task_command_applies_envelope() -> None:
    baseline = np.zeros(5)
    residual = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    command = compose_taskspace_command(
        baseline,
        residual,
        max_forward_m_s=0.005,
        max_lateral_m_s=0.004,
        max_angular_rad_s=0.01,
    )
    assert command[0] == pytest.approx(0.005)
    assert np.all(np.abs(command[1:3]) <= 0.004 + 1e-15)
    assert np.all(np.abs(command[3:5]) <= 0.01 + 1e-15)


def test_policy_action_shapes() -> None:
    observation = np.zeros(32)
    for policy in (ZeroPolicy(), ScriptedPolicy(), LinearPolicy(np.zeros((5, 32)), np.zeros(5))):
        action = policy.act(observation)
        assert action.shape == (5,)
        assert np.all(np.abs(action) <= 1.0 + 1e-12)


def test_linear_policy_save_load_roundtrip(tmp_path: Path) -> None:
    policy = LinearPolicy(np.arange(160, dtype=float).reshape(5, 32) * 1e-4, np.zeros(5))
    path = tmp_path / "policy.npz"
    policy.save(path)
    loaded = LinearPolicy.load(path)
    observation = np.linspace(-1.0, 1.0, 32)
    assert np.allclose(policy.act(observation), loaded.act(observation))


def test_reward_penalizes_axial_overload_and_rewards_success() -> None:
    base = dict(
        previous_depth_m=0.0,
        depth_m=0.001,
        lateral_error_m=0.0,
        angular_error_rad=0.0,
        normalized_action=np.zeros(5),
        previous_normalized_action=np.zeros(5),
        dt_s=0.05,
        success=False,
        fault=False,
        axial_soft_limit_n=8.5,
        expected_depth_step_m=0.000125,
    )
    normal = compute_reward(
        RewardInputs(
            wrench=Wrench("socket_entry", np.array([-7.0, 0.0, 0.0]), np.zeros(3)),
            **base,
        )
    )
    overload = compute_reward(
        RewardInputs(
            wrench=Wrench("socket_entry", np.array([-10.0, 0.0, 0.0]), np.zeros(3)),
            **base,
        )
    )
    success = compute_reward(
        RewardInputs(
            wrench=Wrench("socket_entry", np.array([-7.0, 0.0, 0.0]), np.zeros(3)),
            **{**base, "success": True},
        )
    )
    assert overload.total < normal.total
    assert success.total > normal.total
    assert set(ACTION_NAMES) == {"delta_vx", "delta_vy", "delta_vz", "delta_wy", "delta_wz"}
