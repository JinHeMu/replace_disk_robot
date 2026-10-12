"""Xbox-specific CLI options and default parameter file."""
from pathlib import Path
import numpy as np
from replace_disk_robot.control import JoyControl
from ..common import config

DEFAULT_CONFIG_PATH = Path(__file__).with_name("jaka_joy_config.yaml")

def parse_args(argv=None):
    parser = config._build_parser(DEFAULT_CONFIG_PATH, description=__doc__)
    descriptions = {
        "headless": "poll the controller without a viewer; requires --seconds",
        "dry_run": "read feedback and controller; never enable servo or send motion",
        "command_frame": "base=JAKA base X/Z and tool Z insertion; tool=three independent TCP axes",
        "linear_speed": "maximum joystick translation speed, m/s",
        "angular_speed_deg": "maximum joystick rotation speed, deg/s",
    }
    for action in parser._actions:
        if action.dest in descriptions:
            action.help = descriptions[action.dest]
    parser.add_argument("--backend", choices=("sim", "real"), default="sim")
    parser.add_argument("--controller-index", type=int, default=0)
    parser.add_argument("--joy-deadzone", type=float, default=.15)
    parser.add_argument("--joy-precision-scale", type=float, default=.25)
    parser.add_argument("--joy-timeout-s", type=float, default=.15)
    parser.add_argument("--keyframe", default="low")
    parser.add_argument("--probe", action="store_true", help="print controller input; never open a robot")
    parser.add_argument("--validate", action="store_true", help="synthetic full-loop MuJoCo checks")
    parser.add_argument("--output", type=Path, help="validation report JSON")
    args = config.parse_config(parser, argv)
    JoyControl(args.linear_speed, np.deg2rad(args.angular_speed_deg),
               deadzone=args.joy_deadzone, precision_scale=args.joy_precision_scale,
               timeout_s=args.joy_timeout_s)
    if not np.isfinite([args.rate_hz, args.seconds]).all() or args.rate_hz <= 0 or args.seconds < 0:
        parser.error("rate must be positive and duration non-negative, both finite")
    if args.backend not in ("sim", "real"):
        parser.error("backend must be sim or real")
    if args.validate and args.backend != "sim":
        parser.error("--validate requires --backend sim")
    if args.headless and args.seconds <= 0 and not args.validate:
        parser.error("--headless requires --seconds > 0")
    if args.backend == "real" and not args.probe:
        config._validate_admittance_args(args)
    return args
