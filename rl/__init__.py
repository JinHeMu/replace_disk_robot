"""Residual RL insertion demo package.

The package is intentionally thin and can be imported directly from the
repository root.  Runtime modules depend on the existing ``replace_disk_robot``
package, but never on a training framework.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from .config import RLConfig, load_rl_config  # noqa: E402
from .env import ResidualInsertionEnv  # noqa: E402
from .policy import (  # noqa: E402
    ConstantPolicy,
    LinearPolicy,
    ResidualPolicy,
    ScriptedPolicy,
    TorchPolicy,
    ZeroPolicy,
)
from .runner import EpisodeResult, evaluate_policy, run_episode  # noqa: E402

__all__ = [
    "ConstantPolicy",
    "EpisodeResult",
    "LinearPolicy",
    "RLConfig",
    "ResidualInsertionEnv",
    "ResidualPolicy",
    "ScriptedPolicy",
    "TorchPolicy",
    "ZeroPolicy",
    "evaluate_policy",
    "load_rl_config",
    "run_episode",
]
