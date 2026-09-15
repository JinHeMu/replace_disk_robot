"""Runtime residual policies and a Torch-free deployment contract.

The deployment path only sees NumPy observations and returns NumPy actions.
Training frameworks are optional and must not be imported by the environment,
MuJoCo adapter, contact layer or safety layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .features import FEATURE_NAMES
from .residual import ACTION_NAMES


@runtime_checkable
class ResidualPolicy(Protocol):
    """Minimal policy interface safe for real-time deployment."""

    @property
    def observation_size(self) -> int: ...

    @property
    def action_size(self) -> int: ...

    def reset(self) -> None: ...

    def act(self, observation: NDArray[np.float64]) -> NDArray[np.float64]: ...


class ZeroPolicy:
    """Always return a zero residual; used to validate the classical baseline."""

    observation_size = len(FEATURE_NAMES)
    action_size = len(ACTION_NAMES)

    def reset(self) -> None:
        return None

    def act(self, observation: ArrayLike) -> NDArray[np.float64]:
        observation = np.asarray(observation, dtype=float)
        if observation.shape != (self.observation_size,):
            raise ValueError(
                f"observation must have shape ({self.observation_size},), "
                f"got {observation.shape}"
            )
        return np.zeros(self.action_size, dtype=float)


class ConstantPolicy:
    """Return a bounded constant action; useful for actuator-path smoke tests."""

    observation_size = len(FEATURE_NAMES)
    action_size = len(ACTION_NAMES)

    def __init__(self, action: ArrayLike) -> None:
        value = np.asarray(action, dtype=float)
        if value.shape != (self.action_size,) or not np.all(np.isfinite(value)):
            raise ValueError(f"action must be a finite vector with shape ({self.action_size},)")
        self._action = np.clip(value, -1.0, 1.0)

    def reset(self) -> None:
        return None

    def act(self, observation: ArrayLike) -> NDArray[np.float64]:
        observation = np.asarray(observation, dtype=float)
        if observation.shape != (self.observation_size,):
            raise ValueError(
                f"observation must have shape ({self.observation_size},), "
                f"got {observation.shape}"
            )
        return self._action.copy()


class ScriptedPolicy:
    """Small hand-coded residual for demonstration and ablation.

    The residual is based on the normalized force and lateral-error features.
    It is intentionally weak: the deterministic baseline remains the main
    controller, and the residual is only intended to show the fusion path.
    """

    observation_size = len(FEATURE_NAMES)
    action_size = len(ACTION_NAMES)

    def __init__(
        self,
        force_damping_gain: float = 0.08,
        alignment_gain: float = 0.05,
    ) -> None:
        self.force_damping_gain = float(force_damping_gain)
        self.alignment_gain = float(alignment_gain)

    def reset(self) -> None:
        return None

    def act(self, observation: ArrayLike) -> NDArray[np.float64]:
        observation = np.asarray(observation, dtype=float)
        if observation.shape != (self.observation_size,):
            raise ValueError(
                f"observation must have shape ({self.observation_size},), "
                f"got {observation.shape}"
            )
        index = {name: i for i, name in enumerate(FEATURE_NAMES)}
        fy = observation[index["wrench_fy"]]
        fz = observation[index["wrench_fz"]]
        z = observation[index["lateral_z"]]
        y = observation[index["lateral_y"]]
        # External load pushes the tool; move with it slightly to relieve contact.
        action = np.zeros(self.action_size, dtype=float)
        action[1] = np.clip(self.force_damping_gain * fy - self.alignment_gain * y, -1.0, 1.0)
        action[2] = np.clip(self.force_damping_gain * fz - self.alignment_gain * z, -1.0, 1.0)
        return action


@dataclass(frozen=True)
class LinearPolicy:
    """A simple NumPy linear actor with tanh output.

    ``weights`` maps the 32-dimensional observation to the 5-dimensional
    action.  It exists so the demo can be trained by derivative-free search
    without importing a deep-learning framework; it is not a replacement for
    the SAC training entry point.
    """

    weights: NDArray[np.float64]
    bias: NDArray[np.float64]
    observation_size: int = len(FEATURE_NAMES)
    action_size: int = len(ACTION_NAMES)

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=float)
        bias = np.asarray(self.bias, dtype=float)
        expected = (self.action_size, self.observation_size)
        if weights.shape != expected or not np.all(np.isfinite(weights)):
            raise ValueError(f"weights must have shape {expected}")
        if bias.shape != (self.action_size,) or not np.all(np.isfinite(bias)):
            raise ValueError(f"bias must have shape ({self.action_size},)")
        object.__setattr__(self, "weights", weights.copy())
        object.__setattr__(self, "bias", bias.copy())

    def reset(self) -> None:
        return None

    def act(self, observation: ArrayLike) -> NDArray[np.float64]:
        observation = np.asarray(observation, dtype=float)
        if observation.shape != (self.observation_size,):
            raise ValueError(
                f"observation must have shape ({self.observation_size},), "
                f"got {observation.shape}"
            )
        return np.tanh(self.weights @ observation + self.bias)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            weights=self.weights,
            bias=self.bias,
            observation_names=np.asarray(FEATURE_NAMES, dtype=object),
            action_names=np.asarray(ACTION_NAMES, dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LinearPolicy":
        with np.load(Path(path), allow_pickle=True) as data:
            weights = data["weights"]
            bias = data["bias"]
            observation_names = tuple(str(x) for x in data["observation_names"].tolist())
            action_names = tuple(str(x) for x in data["action_names"].tolist())
        if observation_names != FEATURE_NAMES:
            raise ValueError("saved policy observation schema does not match runtime schema")
        if action_names != ACTION_NAMES:
            raise ValueError("saved policy action schema does not match runtime schema")
        return cls(weights, bias)


class TorchPolicy:
    """Load an exported Torch actor without making Torch a runtime dependency.

    The exported TorchScript module must accept a single float32 tensor with
    shape ``(N, 32)`` and return a single float32 tensor with shape ``(N, 5)``.
    """

    observation_size = len(FEATURE_NAMES)
    action_size = len(ACTION_NAMES)

    def __init__(self, module_path: str | Path) -> None:
        try:
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Torch is required to load a TorchScript residual policy."
            ) from exc
        self._torch = torch
        self._module = torch.jit.load(str(module_path), map_location="cpu")
        self._module.eval()
        self._module_path = Path(module_path)

    def reset(self) -> None:
        return None

    def act(self, observation: ArrayLike) -> NDArray[np.float64]:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.observation_size,):
            raise ValueError(
                f"observation must have shape ({self.observation_size},), "
                f"got {observation.shape}"
            )
        with self._torch.no_grad():
            tensor = self._torch.from_numpy(observation).unsqueeze(0)
            action = self._module(tensor).squeeze(0).numpy().astype(np.float64)
        if action.shape != (self.action_size,) or not np.all(np.isfinite(action)):
            raise ValueError("Torch policy returned an invalid action")
        return np.clip(action, -1.0, 1.0)


__all__ = [
    "ConstantPolicy",
    "LinearPolicy",
    "ResidualPolicy",
    "ScriptedPolicy",
    "TorchPolicy",
    "ZeroPolicy",
]
