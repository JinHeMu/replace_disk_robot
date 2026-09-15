"""Episode runner and evaluation helpers for residual insertion."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .env import ResidualInsertionEnv
from .policy import ResidualPolicy, ZeroPolicy


@dataclass
class EpisodeResult:
    policy_name: str
    seed: int | None
    total_reward: float
    steps: int
    success: bool
    truncated: bool
    fault: bool
    fault_reason: str
    guard_reason: str
    final_depth_m: float
    max_depth_m: float
    episode_time_s: float
    peak_force_n: float
    peak_torque_nm: float
    mean_force_n: float
    residual_clip_steps: int
    residual_timeout_steps: int
    wall_time_s: float
    reward_components: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self, *, include_history: bool = False) -> dict[str, Any]:
        values = {
            "policy": self.policy_name,
            "seed": self.seed,
            "total_reward": self.total_reward,
            "steps": self.steps,
            "success": self.success,
            "truncated": self.truncated,
            "fault": self.fault,
            "fault_reason": self.fault_reason,
            "guard_reason": self.guard_reason,
            "final_depth_m": self.final_depth_m,
            "max_depth_m": self.max_depth_m,
            "episode_time_s": self.episode_time_s,
            "peak_force_n": self.peak_force_n,
            "peak_torque_nm": self.peak_torque_nm,
            "mean_force_n": self.mean_force_n,
            "residual_clip_steps": self.residual_clip_steps,
            "residual_timeout_steps": self.residual_timeout_steps,
            "wall_time_s": self.wall_time_s,
            "reward_components": dict(self.reward_components),
        }
        if include_history:
            values["history"] = list(self.history)
        return values


def _safe_action(
    policy: ResidualPolicy,
    observation: NDArray[np.float64],
    expected_size: int,
) -> NDArray[np.float64]:
    try:
        action = np.asarray(policy.act(observation), dtype=float)
    except Exception:
        return np.zeros(expected_size, dtype=float)
    if action.shape != (expected_size,) or not np.all(np.isfinite(action)):
        return np.zeros(expected_size, dtype=float)
    return np.clip(action, -1.0, 1.0)


def run_episode(
    env: ResidualInsertionEnv,
    policy: ResidualPolicy | None = None,
    *,
    seed: int | None = None,
    policy_name: str | None = None,
    record_history: bool = False,
) -> EpisodeResult:
    """Run one episode and return compact task/learning metrics."""

    active_policy = policy or ZeroPolicy()
    name = policy_name or type(active_policy).__name__
    observation, _ = env.reset(seed=seed)
    active_policy.reset()
    total_reward = 0.0
    steps = 0
    history: list[dict[str, Any]] = []
    wall_start = perf_counter()
    terminated = truncated = False
    last_info: dict[str, Any] = {}
    while not (terminated or truncated):
        action = _safe_action(active_policy, observation, env.action_size)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        last_info = info.as_dict()
        steps += 1
        if record_history:
            history.append(
                {
                    "step": steps,
                    "reward": float(reward),
                    "depth_m": float(last_info["depth_m"]),
                    "lateral_error_m": float(last_info["lateral_error_m"]),
                    "angular_error_rad": float(last_info["angular_error_rad"]),
                    "peak_force_n": float(last_info["peak_force_n"]),
                    "baseline_command": list(last_info["baseline_command"]),
                    "residual_command": list(last_info["residual_command"]),
                    "guard_reason": last_info["guard_reason"],
                }
            )
        if steps > 100_000:
            raise RuntimeError("aborting episode: step limit exceeded")
    wall_time = perf_counter() - wall_start
    return EpisodeResult(
        policy_name=name,
        seed=seed,
        total_reward=float(total_reward),
        steps=steps,
        success=bool(last_info.get("success", False)),
        truncated=bool(last_info.get("truncated", False)),
        fault=bool(last_info.get("fault", False)),
        fault_reason=str(last_info.get("fault_reason", "")),
        guard_reason=str(last_info.get("guard_reason", "")),
        final_depth_m=float(last_info.get("depth_m", 0.0)),
        max_depth_m=float(last_info.get("max_depth_m", 0.0)),
        episode_time_s=float(last_info.get("episode_time_s", 0.0)),
        peak_force_n=float(last_info.get("peak_force_n", 0.0)),
        peak_torque_nm=float(last_info.get("peak_torque_nm", 0.0)),
        mean_force_n=float(last_info.get("mean_force_n", 0.0)),
        residual_clip_steps=int(last_info.get("residual_clip_steps", 0)),
        residual_timeout_steps=int(last_info.get("residual_timeout_steps", 0)),
        wall_time_s=float(wall_time),
        reward_components=dict(last_info.get("reward_components", {})),
        history=history,
    )


def evaluate_policy(
    env: ResidualInsertionEnv,
    policy: ResidualPolicy | None = None,
    *,
    seeds: list[int],
    policy_name: str | None = None,
    record_history: bool = False,
) -> dict[str, Any]:
    """Evaluate one policy on a fixed seed matrix."""

    results = [
        run_episode(
            env,
            policy,
            seed=seed,
            policy_name=policy_name,
            record_history=record_history,
        )
        for seed in seeds
    ]
    success_rate = float(np.mean([r.success for r in results])) if results else 0.0
    fault_counts: dict[str, int] = {}
    for result in results:
        key = str(result.fault_reason or "none")
        fault_counts[key] = fault_counts.get(key, 0) + 1
    return {
        "policy": policy_name or (type(policy).__name__ if policy else "ZeroPolicy"),
        "episodes": len(results),
        "success_rate": success_rate,
        "mean_final_depth_m": float(np.mean([r.final_depth_m for r in results]))
        if results
        else 0.0,
        "mean_episode_time_s": float(np.mean([r.episode_time_s for r in results]))
        if results
        else 0.0,
        "mean_peak_force_n": float(np.mean([r.peak_force_n for r in results]))
        if results
        else 0.0,
        "mean_peak_torque_nm": float(np.mean([r.peak_torque_nm for r in results]))
        if results
        else 0.0,
        "faults": fault_counts,
        "results": [r.as_dict(include_history=record_history) for r in results],
    }


__all__ = ["EpisodeResult", "evaluate_policy", "run_episode"]
