from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl import ResidualInsertionEnv, ZeroPolicy, run_episode


def test_environment_observation_and_zero_policy_baseline() -> None:
    env = ResidualInsertionEnv(seed=0)
    observation, info = env.reset(seed=0)
    assert observation.shape == (32,)
    assert np.all(np.isfinite(observation))
    assert info["depth_m"] < 0.0

    result = run_episode(env, ZeroPolicy(), seed=0, policy_name="zero")
    assert result.success
    assert not result.fault
    assert result.final_depth_m >= env.config.insertion.success_depth_m
    assert result.peak_force_n < env.guard.config.hard_force_n
    assert result.residual_timeout_steps == 0
    assert result.residual_clip_steps == 0


def test_same_seed_reset_is_deterministic() -> None:
    env_a = ResidualInsertionEnv(seed=3)
    env_b = ResidualInsertionEnv(seed=3)
    first, _ = env_a.reset(seed=3)
    second, _ = env_b.reset(seed=3)
    if not np.allclose(first, second):
        raise AssertionError("same seed must produce the same reset observation")
