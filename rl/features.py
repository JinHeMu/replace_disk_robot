"""Fixed-order, normalized actor observations for residual insertion.

The actor must never see MuJoCo contact-object identifiers, ground-truth
randomization parameters, or raw arrays whose units are implicit.  This module
turns a named physical snapshot into the 32-dimensional schema defined by the
design document.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from replace_disk_robot.core.types import Wrench

from .config import ObservationConfig


OBSERVATION_SCHEMA_VERSION = "residual-insertion-v1"

FEATURE_NAMES: tuple[str, ...] = (
    "insertion_depth",
    "lateral_y",
    "lateral_z",
    "pitch",
    "yaw",
    "tcp_vx",
    "tcp_vy",
    "tcp_vz",
    "tcp_wy",
    "tcp_wz",
    "wrench_fx",
    "wrench_fy",
    "wrench_fz",
    "wrench_tx",
    "wrench_ty",
    "wrench_tz",
    "wrench_rate_fx",
    "wrench_rate_fy",
    "wrench_rate_fz",
    "wrench_rate_tx",
    "wrench_rate_ty",
    "wrench_rate_tz",
    "baseline_vx",
    "baseline_vy",
    "baseline_vz",
    "baseline_wy",
    "baseline_wz",
    "last_residual_vx",
    "last_residual_vy",
    "last_residual_vz",
    "last_residual_wy",
    "last_residual_wz",
)


def _finite_vector(value: ArrayLike, size: int, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    return array.copy()


@dataclass(frozen=True)
class FeatureSnapshot:
    """Named physical quantities in the ``socket_entry`` task frame."""

    depth_m: float
    lateral_offset_m: NDArray[np.float64]  # [y, z]
    pitch_yaw_rad: NDArray[np.float64]  # [pitch, yaw]
    tcp_velocity_m_s: NDArray[np.float64]  # [vx, vy, vz]
    tcp_angular_velocity_rad_s: NDArray[np.float64]  # [wy, wz]
    wrench: Wrench
    wrench_rate: Wrench
    baseline_command: NDArray[np.float64]  # [vx, vy, vz, wy, wz]
    last_residual: NDArray[np.float64]  # [vx, vy, vz, wy, wz]

    def __post_init__(self) -> None:
        if not np.isfinite(self.depth_m):
            raise ValueError("depth_m must be finite")
        object.__setattr__(
            self,
            "lateral_offset_m",
            _finite_vector(self.lateral_offset_m, 2, "lateral_offset_m"),
        )
        object.__setattr__(
            self,
            "pitch_yaw_rad",
            _finite_vector(self.pitch_yaw_rad, 2, "pitch_yaw_rad"),
        )
        object.__setattr__(
            self,
            "tcp_velocity_m_s",
            _finite_vector(self.tcp_velocity_m_s, 3, "tcp_velocity_m_s"),
        )
        object.__setattr__(
            self,
            "tcp_angular_velocity_rad_s",
            _finite_vector(
                self.tcp_angular_velocity_rad_s, 2, "tcp_angular_velocity_rad_s"
            ),
        )
        object.__setattr__(
            self,
            "baseline_command",
            _finite_vector(self.baseline_command, 5, "baseline_command"),
        )
        object.__setattr__(
            self,
            "last_residual",
            _finite_vector(self.last_residual, 5, "last_residual"),
        )


def observation_schema() -> dict[str, object]:
    """Return a serializable schema for manifests and deployment checks."""

    return {
        "version": OBSERVATION_SCHEMA_VERSION,
        "size": len(FEATURE_NAMES),
        "names": list(FEATURE_NAMES),
    }


def build_observation(
    snapshot: FeatureSnapshot,
    config: ObservationConfig | None = None,
) -> NDArray[np.float64]:
    """Return the fixed 32-vector, clipped to ``config.clip_abs``.

    The function is deterministic, uses no running statistics, and never
    mutates the input snapshot.
    """

    cfg = config or ObservationConfig()
    if snapshot.wrench.frame_id != snapshot.wrench_rate.frame_id:
        raise ValueError("wrench and wrench_rate must use the same frame")
    wrench = snapshot.wrench.as_vector()
    wrench_rate = snapshot.wrench_rate.as_vector()
    command_scale = np.asarray(cfg.command_scale, dtype=float)
    limit = float(cfg.clip_abs)
    normalized = np.concatenate(
        (
            np.array([snapshot.depth_m / cfg.depth_scale_m]),
            snapshot.lateral_offset_m / cfg.alignment_scale_m,
            snapshot.pitch_yaw_rad / cfg.angle_scale_rad,
            snapshot.tcp_velocity_m_s / cfg.linear_velocity_scale_m_s,
            snapshot.tcp_angular_velocity_rad_s / cfg.angular_velocity_scale_rad_s,
            wrench[:3] / cfg.force_scale_n,
            wrench[3:] / cfg.torque_scale_nm,
            wrench_rate[:3] / cfg.force_scale_n,
            wrench_rate[3:] / cfg.torque_scale_nm,
            snapshot.baseline_command / command_scale,
            snapshot.last_residual / command_scale,
        )
    )
    if normalized.shape != (len(FEATURE_NAMES),):
        raise RuntimeError(
            f"internal observation size mismatch: {normalized.shape} != "
            f"({len(FEATURE_NAMES)},)"
        )
    return np.clip(normalized, -limit, limit).astype(np.float64)


def feature_indices(names: Iterable[str]) -> tuple[int, ...]:
    """Return feature indices for the requested names, preserving order."""

    index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    missing = [name for name in names if name not in index]
    if missing:
        raise KeyError(f"unknown feature names: {missing}")
    return tuple(index[name] for name in names)


__all__ = [
    "FEATURE_NAMES",
    "OBSERVATION_SCHEMA_VERSION",
    "FeatureSnapshot",
    "build_observation",
    "feature_indices",
    "observation_schema",
]
