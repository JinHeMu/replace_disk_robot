"""Final-command guard for the simulation residual-RL demo.

This is intentionally a small, deterministic gate inspired by the repository's
``ForceLimitGuard``.  It does not depend on the reward, the policy, or training
configuration.  On any hard violation it latches a fault and holds the measured
joint position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class GuardConfig:
    hard_force_n: float = 10.0
    hard_torque_nm: float = 1.0
    max_depth_m: float = 0.075
    min_depth_m: float = -0.025
    max_lateral_error_m: float = 0.010
    max_angular_error_rad: float = np.deg2rad(8.0)
    max_tracking_error_rad: float = 10.0

    def __post_init__(self) -> None:
        if self.hard_force_n <= 0 or self.hard_torque_nm <= 0:
            raise ValueError("hard safety limits must be positive")
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError("max_depth_m must be greater than min_depth_m")


@dataclass(frozen=True)
class GuardResult:
    joint_target: NDArray[np.float64]
    fault: bool
    reason: str
    force_norm_n: float
    torque_norm_nm: float


class InsertionGuard:
    """Latching hard-limit and task-envelope supervisor."""

    def __init__(self, config: GuardConfig | None = None) -> None:
        self.config = config or GuardConfig()
        self.reset()

    def reset(self) -> None:
        self._fault = False
        self._reason = "ok"

    @property
    def fault(self) -> bool:
        return self._fault

    @property
    def reason(self) -> str:
        return self._reason

    def update(
        self,
        wrench_task: ArrayLike,
        current_q: ArrayLike,
        requested_q: ArrayLike,
        *,
        depth_m: float,
        lateral_error_m: float,
        angular_error_rad: float,
        tracking_error_rad: float | None = None,
    ) -> GuardResult:
        """Return a safe joint target and latch on the first hard violation."""

        wrench = np.asarray(wrench_task, dtype=float)
        current = np.asarray(current_q, dtype=float)
        requested = np.asarray(requested_q, dtype=float)
        if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
            return self._trip(current, "invalid_wrench", np.nan, np.nan)
        if current.shape != requested.shape or current.ndim != 1:
            return self._trip(current, "invalid_joint_target", np.nan, np.nan)
        if not all(np.isfinite(v) for v in (depth_m, lateral_error_m, angular_error_rad)):
            return self._trip(current, "invalid_task_state", np.nan, np.nan)

        force_norm = float(np.linalg.norm(wrench[:3]))
        torque_norm = float(np.linalg.norm(wrench[3:]))
        if self._fault:
            return GuardResult(
                current.copy(), True, self._reason, force_norm, torque_norm
            )
        if force_norm > self.config.hard_force_n:
            return self._trip(current, "hard_force", force_norm, torque_norm)
        if torque_norm > self.config.hard_torque_nm:
            return self._trip(current, "hard_torque", force_norm, torque_norm)
        if depth_m > self.config.max_depth_m:
            return self._trip(current, "workspace_depth", force_norm, torque_norm)
        if depth_m < self.config.min_depth_m:
            return self._trip(current, "reverse_out_of_bounds", force_norm, torque_norm)
        if abs(lateral_error_m) > self.config.max_lateral_error_m:
            return self._trip(current, "lateral_workspace", force_norm, torque_norm)
        if abs(angular_error_rad) > self.config.max_angular_error_rad:
            return self._trip(current, "angular_workspace", force_norm, torque_norm)
        if (
            tracking_error_rad is not None
            and np.isfinite(tracking_error_rad)
            and tracking_error_rad > self.config.max_tracking_error_rad
        ):
            return self._trip(current, "tracking_error", force_norm, torque_norm)
        return GuardResult(requested.copy(), False, "ok", force_norm, torque_norm)

    def _trip(
        self,
        current: NDArray[np.float64],
        reason: str,
        force_norm: float,
        torque_norm: float,
    ) -> GuardResult:
        self._fault = True
        self._reason = reason
        return GuardResult(current.copy(), True, reason, force_norm, torque_norm)


__all__ = ["GuardConfig", "GuardResult", "InsertionGuard"]
