"""Shared helpers for the standalone JAKA example tools.

The canonical implementation lives in ``replace_disk_robot.adapters.jaka``.
This module only adds the repository ``src`` directory to ``sys.path`` and
provides small CLI/session/scheduling helpers shared by the example scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from replace_disk_robot.adapters.jaka import (  # noqa: E402
    ABS,
    CONTINUE,
    COORD_BASE,
    COORD_JOINT,
    COORD_TOOL,
    DEFAULT_SENSOR_TO_TOOL_ROTATION,
    DEFAULT_TOOL_ARM_M,
    INCR,
    JAKA_JOINT_NAMES,
    STOP,
    EdgState,
    JakaEdgClient,
    JakaError,
    JakaRobotAdapter,
    JakaWristFTAdapter,
    fmt_array,
    is_arm_port,
    is_force_torque_port,
    load_jkrc,
)

__all__ = [
    "ABS",
    "CONTINUE",
    "COORD_BASE",
    "COORD_JOINT",
    "COORD_TOOL",
    "DEFAULT_SENSOR_TO_TOOL_ROTATION",
    "DEFAULT_TOOL_ARM_M",
    "EdgState",
    "INCR",
    "JAKA_JOINT_NAMES",
    "JakaEdgClient",
    "JakaError",
    "JakaRobotAdapter",
    "JakaWristFTAdapter",
    "RateLoop",
    "STOP",
    "add_network_args",
    "edg_session",
    "fmt_array",
    "is_arm_port",
    "is_force_torque_port",
    "load_jkrc",
    "make_client",
    "parse_bool",
    "wait_for_edg_state",
]


from replace_disk_robot.adapters.jaka.session import (  # noqa: E402
    RateLoop, add_network_args, edg_session, make_client, parse_bool, wait_for_edg_state,
)
