"""Keyboard-specific CLI descriptions and default parameter file."""
from pathlib import Path
from ..common.config import (
    PROJECT_ROOT, _build_parser as build_common_parser, _load_config,
    _validate_config_args, _validate_admittance_args, parse_config,
)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("jaka_keyboard_servo_config.yaml")
CLI_DESCRIPTION = """Keyboard Cartesian servo for a real JAKA ZU5.

This is the hardware counterpart of ``examples/keyboard_servo.py``.  It does
not power on, enable, or disable the robot by itself; run ``jaka_start.py``
first and ``jaka_stop.py`` afterwards.

The initial Cartesian target is always the measured joint state at startup, not
a MuJoCo keyframe.  R/F are always the insertion pair and move along the current
``tool0`` +Z/-Z (blue) axis.  In the default base mode, W/S use
``jaka_base_link`` +/-Z and A/D use -X/+X (platform left/right); rotations use
base axes.  With ``--command-frame tool`` those other keys use the current
``tool0`` frame instead; R/F stay on tool0 Z.

The 125 Hz loop reads EDG state, applies the same damped Cartesian servo and
force gate used by the simulation, and sends joint targets through
``JakaRobotAdapter.command_joint_positions()``.  By default it loads the
payload-identification JSON and removes the identified sensor bias, payload
gravity and center-of-mass moment before the force gate.  A non-blocking
six-axis force/torque plot is shown by default.  ``--tare-compensated`` takes
the current gravity-compensated wrench as an additional zero offset.  Use
``--dry-run`` first: it reads EDG/F/T and keyboard state but never enables
servo or sends commands."""

def _build_parser():
    parser = build_common_parser(DEFAULT_CONFIG_PATH, description=CLI_DESCRIPTION)
    descriptions = {
        "linear_speed": "keyboard linear speed in m/s (default: 0.01)",
        "angular_speed_deg": "keyboard angular speed in deg/s (default: 5)",
        "command_frame": "base: W/S base Z, A/D base X; tool: tool0 axes; R/F always tool0 Z",
        "headless": "run without GLFW and command zero velocity (requires --seconds)",
        "dry_run": "read EDG/F/T and keyboard; never enable servo or send motion",
    }
    for action in parser._actions:
        if action.dest in descriptions:
            action.help = descriptions[action.dest]
    return parser


def _parse_args(argv=None):
    return parse_config(_build_parser(), argv, allow_missing_default=True)
