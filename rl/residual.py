"""Action schema, scaling, rate limiting and fusion for the 5-D residual.

The policy emits a normalized action ``a in [-1, 1]^5``.  The limiter owns the
transition from normalized output to a bounded physical Cartesian velocity
residual.  The final command envelope is still applied by the runner/guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import ResidualConfig


ACTION_NAMES: tuple[str, ...] = (
    "delta_vx",
    "delta_vy",
    "delta_vz",
    "delta_wy",
    "delta_wz",
)


def _five(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if array.shape != (5,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape (5,)")
    return array.copy()


@dataclass(frozen=True)
class ResidualLimitInfo:
    """Diagnostics recorded for every policy action."""

    raw_physical: NDArray[np.float64]
    limited_physical: NDArray[np.float64]
    amplitude_clipped: bool
    rate_clipped: bool
    cumulative_clipped: bool
    timeout: bool = False

    @property
    def any_clipped(self) -> bool:
        return bool(
            self.amplitude_clipped
            or self.rate_clipped
            or self.cumulative_clipped
            or self.timeout
        )


class ResidualLimiter:
    """Stateful deterministic limiter for 5-D task-space residual velocity."""

    def __init__(self, config: ResidualConfig | None = None) -> None:
        self.config = config or ResidualConfig()
        self._limits = np.asarray(self.config.action_limits, dtype=float)
        self._max_delta = np.asarray(self.config.max_action_delta, dtype=float)
        self.reset()

    def reset(self) -> None:
        self._last_physical = np.zeros(5, dtype=float)
        self._cumulative = np.zeros(5, dtype=float)
        self._last_time_s: float | None = None
        self._last_timeout = False

    @property
    def last_physical(self) -> NDArray[np.float64]:
        return self._last_physical.copy()

    def zero(
        self,
        now_s: float | None = None,
        *,
        timed_out: bool = False,
    ) -> tuple[NDArray[np.float64], ResidualLimitInfo]:
        """Return a zero residual while preserving limiter history."""

        zeros = np.zeros(5, dtype=float)
        self._last_physical = zeros
        if now_s is not None:
            self._last_time_s = float(now_s)
        self._last_timeout = bool(timed_out)
        return zeros, ResidualLimitInfo(
            raw_physical=zeros.copy(),
            limited_physical=zeros.copy(),
            amplitude_clipped=False,
            rate_clipped=False,
            cumulative_clipped=False,
            timeout=bool(timed_out),
        )

    def limit(
        self,
        normalized_action: ArrayLike,
        dt_s: float,
        now_s: float | None = None,
    ) -> tuple[NDArray[np.float64], ResidualLimitInfo]:
        """Scale, rate-limit and cumulative-limit one normalized action."""

        action = np.asarray(normalized_action, dtype=float)
        if action.shape != (5,) or not np.all(np.isfinite(action)):
            raise ValueError("normalized_action must be a finite vector with shape (5,)")
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be finite and positive")

        amplitude = np.clip(action, -1.0, 1.0)
        amplitude_clipped = bool(np.any(amplitude != action))
        raw = amplitude * self._limits

        delta = np.clip(raw - self._last_physical, -self._max_delta, self._max_delta)
        rate_clipped = bool(np.any(delta != raw - self._last_physical))
        candidate = self._last_physical + delta

        # Bound the time-integral of lateral and angular residual commands.
        # The axial residual is intentionally not included: progress belongs to
        # the baseline and must not be suppressed by a lateral wind-up limit.
        cumulative_candidate = self._cumulative + candidate * dt_s
        cumulative_clipped = False
        lateral_norm = float(np.linalg.norm(cumulative_candidate[[1, 2]]))
        angle_norm = float(np.linalg.norm(cumulative_candidate[[3, 4]]))
        if lateral_norm > self.config.max_cumulative_lateral_m:
            scale = self.config.max_cumulative_lateral_m / max(lateral_norm, 1e-15)
            candidate[[1, 2]] *= scale
            cumulative_clipped = True
        if angle_norm > self.config.max_cumulative_angular_rad:
            scale = self.config.max_cumulative_angular_rad / max(angle_norm, 1e-15)
            candidate[[3, 4]] *= scale
            cumulative_clipped = True
        if cumulative_clipped:
            cumulative_candidate = self._cumulative + candidate * dt_s

        self._last_physical = candidate.copy()
        self._cumulative = cumulative_candidate.copy()
        if now_s is not None:
            self._last_time_s = float(now_s)
        self._last_timeout = False
        return candidate.copy(), ResidualLimitInfo(
            raw_physical=raw,
            limited_physical=candidate.copy(),
            amplitude_clipped=amplitude_clipped,
            rate_clipped=rate_clipped,
            cumulative_clipped=cumulative_clipped,
        )

    def action_age_s(self, now_s: float) -> float:
        if self._last_time_s is None:
            return float("inf")
        return max(0.0, float(now_s) - self._last_time_s)

    def check_timeout(self, now_s: float) -> bool:
        """Return True when the last policy action is older than the timeout."""

        if self._last_time_s is None:
            return False
        timeout = self.action_age_s(now_s) > self.config.policy_timeout_s
        if timeout:
            self.zero(now_s, timed_out=True)
        return timeout

    def info(self) -> dict[str, Any]:
        return {
            "action_names": list(ACTION_NAMES),
            "last_physical": self._last_physical.tolist(),
            "cumulative": self._cumulative.tolist(),
        }


def denormalize_action(
    normalized_action: ArrayLike,
    config: ResidualConfig | None = None,
) -> NDArray[np.float64]:
    """Scale a normalized action without any stateful limiting."""

    cfg = config or ResidualConfig()
    action = _five(normalized_action, "normalized_action")
    limits = np.asarray(cfg.action_limits, dtype=float)
    return np.clip(action, -1.0, 1.0) * limits


def compose_taskspace_command(
    baseline_command: ArrayLike,
    residual_command: ArrayLike,
    *,
    max_forward_m_s: float,
    max_lateral_m_s: float,
    max_angular_rad_s: float,
) -> NDArray[np.float64]:
    """Fuse a baseline command and bounded residual, then apply task limits."""

    baseline = _five(baseline_command, "baseline_command")
    residual = _five(residual_command, "residual_command")
    command = baseline + residual
    command[0] = np.clip(command[0], -max_forward_m_s, max_forward_m_s)
    command[1:3] = np.clip(command[1:3], -max_lateral_m_s, max_lateral_m_s)
    command[3:5] = np.clip(command[3:5], -max_angular_rad_s, max_angular_rad_s)
    return command


__all__ = [
    "ACTION_NAMES",
    "ResidualLimitInfo",
    "ResidualLimiter",
    "compose_taskspace_command",
    "denormalize_action",
]
