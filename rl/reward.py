"""Pure reward function for residual insertion.

Reward terms are physically scaled but dimensionless.  Hard safety thresholds
never appear here as optimized targets; the soft axial limit is the upper bound
of the desired operating band and remains below the guard threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from replace_disk_robot.core.types import Wrench

from .config import RewardConfig


@dataclass(frozen=True)
class RewardInputs:
    previous_depth_m: float
    depth_m: float
    lateral_error_m: float
    angular_error_rad: float
    wrench: Wrench
    normalized_action: NDArray[np.float64]
    previous_normalized_action: NDArray[np.float64]
    dt_s: float
    success: bool
    fault: bool
    axial_soft_limit_n: float
    expected_depth_step_m: float


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    progress: float
    lateral_force: float
    lateral_moment: float
    axial_force_soft: float
    action: float
    action_rate: float
    time: float
    success: float
    fault: float

    def as_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "progress": self.progress,
            "lateral_force": self.lateral_force,
            "lateral_moment": self.lateral_moment,
            "axial_force_soft": self.axial_force_soft,
            "action": self.action,
            "action_rate": self.action_rate,
            "time": self.time,
            "success": self.success,
            "fault": self.fault,
        }


def compute_reward(
    inputs: RewardInputs,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Return the current reward and its named components.

    Sign convention: ``wrench`` is the external load on the tool in the task
    frame.  Positive X is insertion; environmental resistance is therefore
    approximately negative X.
    """

    cfg = config or RewardConfig()
    action = np.asarray(inputs.normalized_action, dtype=float)
    previous_action = np.asarray(inputs.previous_normalized_action, dtype=float)
    if action.shape != (5,) or previous_action.shape != (5,):
        raise ValueError("actions must have shape (5,)")
    if not np.isfinite(inputs.dt_s) or inputs.dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    if inputs.expected_depth_step_m <= 0:
        raise ValueError("expected_depth_step_m must be positive")

    force = np.asarray(inputs.wrench.force_n, dtype=float)
    torque = np.asarray(inputs.wrench.torque_nm, dtype=float)
    if force.shape != (3,) or torque.shape != (3,):
        raise ValueError("wrench must contain 3+3 values")

    progress = (float(inputs.depth_m) - float(inputs.previous_depth_m)) / inputs.expected_depth_step_m
    progress = float(np.clip(progress, -1.0, 2.0))

    # Lateral force and moment are secondary costs; do not reward low force in
    # isolation because that would encourage the policy to stop advancing.
    lateral_force = -cfg.lateral_force_weight * float(np.linalg.norm(force[1:3]) / 5.0) ** 2
    lateral_moment = -cfg.lateral_moment_weight * float(np.linalg.norm(torque[1:3]) / 0.5) ** 2
    resistance = max(0.0, -float(force[0]))
    axial_excess = max(0.0, resistance - float(inputs.axial_soft_limit_n))
    axial_force_soft = -cfg.axial_force_soft_weight * (axial_excess / 2.0) ** 2

    action_penalty = -cfg.action_weight * float(np.mean(np.square(np.clip(action, -1.0, 1.0))))
    action_rate_penalty = -cfg.action_rate_weight * float(
        np.mean(np.square(np.clip(action - previous_action, -2.0, 2.0) / 2.0))
    )
    time_penalty = -cfg.time_weight * float(inputs.dt_s)
    success_bonus = cfg.success_bonus if inputs.success else 0.0
    fault_penalty = -cfg.fault_penalty if inputs.fault else 0.0

    total = float(
        cfg.progress_weight * progress
        + lateral_force
        + lateral_moment
        + axial_force_soft
        + action_penalty
        + action_rate_penalty
        + time_penalty
        + success_bonus
        + fault_penalty
    )
    return RewardBreakdown(
        total=total,
        progress=cfg.progress_weight * progress,
        lateral_force=lateral_force,
        lateral_moment=lateral_moment,
        axial_force_soft=axial_force_soft,
        action=action_penalty,
        action_rate=action_rate_penalty,
        time=time_penalty,
        success=success_bonus,
        fault=fault_penalty,
    )


def reward_schema(config: RewardConfig | None = None) -> dict[str, Any]:
    cfg = config or RewardConfig()
    return {
        "weights": cfg.__dict__,
        "soft_axial_limit_source": "InsertionConfig.axial_soft_limit_n",
        "hard_limits_source": "GuardConfig",
    }


__all__ = ["RewardBreakdown", "RewardInputs", "compute_reward", "reward_schema"]
