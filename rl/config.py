"""Strict configuration objects for the residual-RL insertion demo.

The module deliberately has no dependency on PyTorch, Gym or MuJoCo.  The
built-in dataclass defaults are a complete, runnable configuration.  Optional
YAML files under ``rl/configs`` document the same values and can be loaded when
PyYAML is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import numpy as np

T = TypeVar("T")


def _rad(degrees: float) -> float:
    return float(np.deg2rad(degrees))


def _tuple(value: Any) -> tuple:
    return tuple(value)


@dataclass(frozen=True)
class InsertionConfig:
    """Nominal insertion controller and task-space safety envelope."""

    control_hz: float = 100.0
    rl_hz: float = 20.0
    settle_time_s: float = 0.5
    max_episode_time_s: float = 35.0

    forward_speed_m_s: float = 0.0025
    max_forward_speed_m_s: float = 0.0040
    max_retract_speed_m_s: float = 0.0010
    target_axial_force_n: float = 7.5
    axial_force_gain_m_s_n: float = 0.0010
    axial_soft_limit_n: float = 8.5

    lateral_force_gain_m_s_n: float = 0.0010
    lateral_speed_limit_m_s: float = 0.0035
    angular_torque_gain_rad_s_nm: float = 0.020
    angular_speed_limit_rad_s: float = _rad(0.5)

    success_depth_m: float = 0.050
    stable_steps: int = 5
    max_lateral_error_m: float = 0.0015
    max_pitch_yaw_error_rad: float = _rad(1.0)

    # Task-space command envelope after baseline + residual fusion.
    max_command_forward_m_s: float = 0.005
    max_command_lateral_m_s: float = 0.004
    max_command_angular_rad_s: float = _rad(0.75)

    def __post_init__(self) -> None:
        if self.control_hz <= 0 or self.rl_hz <= 0:
            raise ValueError("control_hz and rl_hz must be positive")
        if self.control_hz % self.rl_hz != 0:
            raise ValueError("control_hz must be an integer multiple of rl_hz")
        if self.settle_time_s < 0 or self.max_episode_time_s <= 0:
            raise ValueError("settle_time_s must be non-negative and max_episode_time_s positive")
        if self.success_depth_m <= 0 or self.stable_steps <= 0:
            raise ValueError("success_depth_m and stable_steps must be positive")


@dataclass(frozen=True)
class ObservationConfig:
    """Physical scales for the fixed 32-dimensional actor observation."""

    schema_version: str = "residual-insertion-v1"
    clip_abs: float = 3.0
    depth_scale_m: float = 0.060
    alignment_scale_m: float = 0.0015
    angle_scale_rad: float = _rad(1.0)
    linear_velocity_scale_m_s: float = 0.005
    angular_velocity_scale_rad_s: float = _rad(0.75)
    force_scale_n: float = 10.0
    torque_scale_nm: float = 1.0
    wrench_rate_time_s: float = 0.05
    command_scale: tuple[float, float, float, float, float] = (
        0.005,
        0.004,
        0.004,
        _rad(0.75),
        _rad(0.75),
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("depth_scale_m", self.depth_scale_m),
            ("alignment_scale_m", self.alignment_scale_m),
            ("angle_scale_rad", self.angle_scale_rad),
            ("linear_velocity_scale_m_s", self.linear_velocity_scale_m_s),
            ("angular_velocity_scale_rad_s", self.angular_velocity_scale_rad_s),
            ("force_scale_n", self.force_scale_n),
            ("torque_scale_nm", self.torque_scale_nm),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        scales = _tuple(self.command_scale)
        if len(scales) != 5 or any(not np.isfinite(s) or s <= 0 for s in scales):
            raise ValueError("command_scale must contain five positive values")
        object.__setattr__(self, "command_scale", scales)


@dataclass(frozen=True)
class ResidualConfig:
    """Bounds for the five-dimensional residual action."""

    action_limits: tuple[float, float, float, float, float] = (
        0.0005,
        0.0003,
        0.0003,
        _rad(0.1),
        _rad(0.1),
    )
    # Per-control-step change bounds on the physical residual.
    max_action_delta: tuple[float, float, float, float, float] = (
        0.0005,
        0.0003,
        0.0003,
        _rad(0.1),
        _rad(0.1),
    )
    max_cumulative_lateral_m: float = 0.012
    max_cumulative_angular_rad: float = _rad(5.0)
    policy_timeout_s: float = 0.10

    def __post_init__(self) -> None:
        limits = _tuple(self.action_limits)
        deltas = _tuple(self.max_action_delta)
        if len(limits) != 5 or any(not np.isfinite(v) or v <= 0 for v in limits):
            raise ValueError("action_limits must contain five positive values")
        if len(deltas) != 5 or any(not np.isfinite(v) or v <= 0 for v in deltas):
            raise ValueError("max_action_delta must contain five positive values")
        if self.max_cumulative_lateral_m <= 0 or self.max_cumulative_angular_rad <= 0:
            raise ValueError("cumulative residual limits must be positive")
        object.__setattr__(self, "action_limits", limits)
        object.__setattr__(self, "max_action_delta", deltas)


@dataclass(frozen=True)
class RewardConfig:
    """Dimensionless reward weights; hard safety thresholds live elsewhere."""

    progress_weight: float = 1.0
    lateral_force_weight: float = 0.20
    lateral_moment_weight: float = 0.10
    axial_force_soft_weight: float = 0.30
    action_weight: float = 0.05
    action_rate_weight: float = 0.05
    time_weight: float = 0.02
    success_bonus: float = 10.0
    fault_penalty: float = 10.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class RandomizationConfig:
    """C0/C1/C2 randomization ranges.

    The demo defaults to C0 (off).  Ranges are intentionally small and are not
    allowed to steal ownership of the hard safety limits.
    """

    enabled: bool = False
    lateral_offset_m: float = 0.0005
    angle_rad: float = _rad(0.4)
    wrench_noise_n: float = 0.05
    friction_scale: float = 0.03


@dataclass(frozen=True)
class SACConfig:
    """Optional training hyper-parameters for the Torch SAC entry point."""

    hidden_sizes: tuple[int, int] = (256, 256)
    learning_rate: float = 3e-4
    buffer_size: int = 200_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    target_entropy: float = -5.0
    seed: int = 0


@dataclass(frozen=True)
class RLConfig:
    """Complete immutable configuration bundle."""

    insertion: InsertionConfig = field(default_factory=InsertionConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    sac: SACConfig = field(default_factory=SACConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _from_mapping(cls: type[T], values: dict[str, Any]) -> T:
    """Recursively construct a frozen dataclass from a mapping."""

    if not isinstance(values, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}")
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {unknown}")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in values:
            continue
        value = values[item.name]
        field_type = hints.get(item.name)
        if is_dataclass(field_type):
            value = _from_mapping(field_type, value)
        kwargs[item.name] = value
    return cls(**kwargs)


def load_rl_config(path: str | Path | None = None) -> RLConfig:
    """Load an optional YAML bundle, otherwise return built-in defaults.

    PyYAML is intentionally optional so the runtime demo can be executed with a
    minimal NumPy/MuJoCo environment.
    """

    if path is None:
        return RLConfig()
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"RL config does not exist: {config_path}")
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "PyYAML is required to load external RL config files; install "
            "`pip install PyYAML` or use the built-in defaults."
        ) from exc

    if config_path.is_dir():
        values: dict[str, Any] = {}
        files = sorted(config_path.glob("*.yaml")) + sorted(config_path.glob("*.yml"))
        if not files:
            raise ValueError(f"RL config directory contains no YAML files: {config_path}")
        for config_file in files:
            with config_file.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
            if loaded is None:
                continue
            if not isinstance(loaded, dict):
                raise ValueError(f"RL config section must be a mapping: {config_file}")
            duplicate = set(values).intersection(loaded)
            if duplicate:
                raise ValueError(f"duplicate RL config keys in {config_file}: {sorted(duplicate)}")
            values.update(loaded)
    else:
        with config_path.open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"RL config root must be a mapping: {config_path}")
    return _from_mapping(RLConfig, values)


__all__ = [
    "InsertionConfig",
    "ObservationConfig",
    "RandomizationConfig",
    "ResidualConfig",
    "RewardConfig",
    "RLConfig",
    "SACConfig",
    "load_rl_config",
]
