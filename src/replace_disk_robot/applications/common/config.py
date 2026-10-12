"""Shared JAKA CLI options, configuration loading and validation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from replace_disk_robot.adapters.jaka.session import add_network_args, parse_bool

PROJECT_ROOT = Path(__file__).resolve().parents[4]

def _build_parser(default_config_path: Path, *, description="JAKA Cartesian teleoperation") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path,
        help=f"YAML parameter file (default: {default_config_path})",
    )
    add_network_args(parser)
    parser.add_argument("--rate-hz", type=float, default=125.0,
                        help="EDG servo command rate; 125 Hz sends step_num=1")
    parser.add_argument("--linear-speed", type=float, default=0.01,
                        help="input linear speed in m/s (default: 0.01)")
    parser.add_argument("--angular-speed-deg", type=float, default=5.0,
                        help="input angular speed in deg/s (default: 5)")
    parser.add_argument("--command-frame", choices=("base", "tool"), default="base",
                        help=("velocity frame. base=jaka_base_link; tool=tool0; "
                              "axis mapping is defined by the application"))
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="stop after this many seconds; 0 means run until exit")
    parser.add_argument("--headless", action="store_true",
                        help="run without a control window (requires --seconds)")
    parser.add_argument("--dry-run", action="store_true",
                        help="read EDG/F/T and input; never enable servo or send motion")
    parser.add_argument("--plot-wrench-enable", type=parse_bool, default=True,
                        metavar="true|false", help="enable live force/torque curves")
    parser.add_argument("--max-force-n", type=float, default=10.0,
                        help="stop when filtered force norm exceeds this value")
    parser.add_argument("--max-torque-nm", type=float, default=2.0,
                        help="stop when filtered torque norm exceeds this value")
    parser.add_argument("--max-joint-error-rad", type=float, default=0.15,
                        help="joint tracking error limit passed to CartesianServo")
    parser.add_argument("--torque-sensor-mode", type=int, default=1,
                        help="passed to set_torque_sensor_mode before EDG; negative disables")
    parser.add_argument("--tare-samples", type=int, default=50)
    parser.add_argument("--tare-period-ms", type=float, default=10.0)
    parser.add_argument("--tare-enable", type=parse_bool, default=True,
                        metavar="true|false", help="enable startup F/T tare without gravity compensation")
    parser.add_argument(
        "--tare-compensated",
        action="store_true",
        help=("after gravity compensation, take the current gravity-compensated "
              "wrench as zero and subtract it from later readings"),
    )
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="F/T low-pass filter alpha")
    parser.add_argument("--deadband-force", type=float, default=0.0)
    parser.add_argument("--deadband-torque", type=float, default=0.0)
    parser.add_argument("--identity-transform", action="store_true",
                        help="skip F/T frame/arm compensation; useful for raw diagnostics")
    parser.add_argument(
        "--gravity-json",
        type=Path,
        default=PROJECT_ROOT / "tool" / "ft_gravity_samples_identified.json",
        help=("payload gravity identification JSON used for online compensation "
              "(default: tool/ft_gravity_samples_identified.json)"),
    )
    parser.add_argument(
        "--gravity-compensation-enable", type=parse_bool, default=True,
        metavar="true|false", help="enable payload gravity compensation",
    )
    parser.add_argument(
        "--admittance",
        action="store_true",
        help=("input nominal pose + six-axis admittance compliance. Needs the "
              "gravity compensation above, otherwise the payload's own weight would "
              "push the compliant offset away in every pose. Consider raising "
              "--max-torque-nm: the 0.4 m tool arm turns a few newtons at the tool "
              "tip into more than the default 2 N*m about tool0"),
    )
    parser.add_argument("--admittance-axes", choices=("translation", "all"),
                        default="translation",
                        help="wrench axes used by admittance (default: translation only)")
    parser.add_argument(
        "--adm-disable-axes",
        nargs="*",
        choices=("x", "y", "z", "rx", "ry", "rz"),
        default=(),
        metavar="AXIS",
        help=("disable individual admittance axes after --admittance-axes, e.g. "
              "'--adm-disable-axes z' disables vertical compliance"),
    )
    parser.add_argument("--adm-mass", type=float, nargs=6,
                        default=(2.0, 2.0, 2.0, 0.02, 0.02, 0.02),
                        help="admittance mass: kg (translation) then kg*m^2 (rotation)")
    parser.add_argument("--adm-damping", type=float, nargs=6,
                        default=(60.0, 60.0, 60.0, 1.5, 1.5, 1.5),
                        help="admittance damping: N*s/m then N*m*s/rad")
    parser.add_argument("--adm-stiffness", type=float, nargs=6,
                        default=(300.0, 300.0, 300.0, 30.0, 30.0, 30.0),
                        help="admittance stiffness: N/m then N*m/rad")
    parser.add_argument("--adm-max-velocity", type=float, nargs=6,
                        default=(0.05, 0.05, 0.05, 0.17, 0.17, 0.17),
                        help="admittance velocity limit: m/s then rad/s")
    parser.add_argument("--adm-max-offset-m", type=float, default=0.03,
                        help="maximum compliant translation offset from the nominal pose")
    parser.add_argument("--adm-max-offset-deg", type=float, default=8.0,
                        help="maximum compliant rotation offset from the nominal pose")
    parser.add_argument("--log-dir", type=Path, default=None,
                        help="new session directory for asynchronous logs; disabled by default")
    parser.add_argument("--log-queue-size", type=int, default=4096,
                        help="bounded logging queue; full queues drop records and count losses")
    return parser


def _load_config(
    path: Path,
    parser: argparse.ArgumentParser,
) -> dict[str, object]:
    """Read a YAML parameter file and return argparse-compatible defaults.

    JSON files are still accepted for compatibility.  YAML comments are
    ignored by the parser, so this is the preferred format for annotated
    user-facing configuration.
    """

    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot read config {path}: {exc}") from exc

    suffix = path.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            import yaml
            raw = yaml.safe_load(text)
        elif suffix == ".json":
            raw = json.loads(text)
        else:
            # Unknown extension: try YAML first, then JSON.
            try:
                import yaml
                raw = yaml.safe_load(text)
            except Exception:
                raw = json.loads(text)
    except ImportError as exc:
        raise SystemExit(
            "YAML config requires PyYAML. Install with: pip install pyyaml"
        ) from exc
    except Exception as exc:
        raise SystemExit(f"cannot parse config {path}: {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise SystemExit(f"config root must be a mapping: {path}")

    allowed = {
        action.dest for action in parser._actions
        if action.dest not in ("help", "config")
    }
    config: dict[str, object] = {}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        if key not in allowed:
            raise SystemExit(
                f"unknown config key {key!r} in {path}; "
                f"allowed keys: {', '.join(sorted(allowed))}"
            )
        if key in ("gravity_compensation_enable", "tare_enable", "plot_wrench_enable"):
            if not isinstance(value, bool):
                parser.error(f"config {key} must be a boolean (true or false)")
        if key == "gravity_json" and isinstance(value, str):
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
            value = candidate
        elif key == "adm_disable_axes":
            if not isinstance(value, list):
                raise SystemExit("config adm_disable_axes must be a JSON array")
            value = tuple(value)
        config[key] = value
    return config


def _validate_config_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Validate values supplied by a config file because set_defaults skips choices."""

    for name in ("gravity_compensation_enable", "tare_enable", "plot_wrench_enable"):
        if not isinstance(getattr(args, name), bool):
            parser.error(f"{name} must be a YAML/JSON boolean (true or false)")
    if args.command_frame not in ("base", "tool"):
        parser.error("command_frame must be 'base' or 'tool'")
    if args.admittance_axes not in ("translation", "all"):
        parser.error("admittance_axes must be 'translation' or 'all'")
    valid_axes = {"x", "y", "z", "rx", "ry", "rz"}
    invalid_axes = [axis for axis in args.adm_disable_axes if axis not in valid_axes]
    if invalid_axes:
        parser.error(f"invalid adm_disable_axes: {invalid_axes}")
    for name in ("adm_mass", "adm_damping", "adm_stiffness", "adm_max_velocity"):
        value = getattr(args, name)
        if len(value) != 6:
            parser.error(f"config {name} must contain six numbers")
    if (not isinstance(args.log_queue_size, int) or isinstance(args.log_queue_size, bool)
            or args.log_queue_size < 1):
        parser.error("log_queue_size must be positive")


def parse_config(parser, argv=None, *, allow_missing_default=False):
    """Load file defaults, apply explicit CLI overrides, then validate."""
    preliminary, _ = parser.parse_known_args(argv)
    path = Path(preliminary.config).expanduser().resolve()
    default = Path(parser.get_default("config")).expanduser().resolve()
    if not path.is_file() and not (allow_missing_default and path == default):
        parser.error(f"config file not found: {path}")
    parser.set_defaults(**_load_config(path, parser))
    args = parser.parse_args(argv)
    _validate_config_args(args, parser)
    return args


def _validate_admittance_args(args: argparse.Namespace) -> None:
    """Reject admittance configurations that cannot be safe on hardware.

    Called from ``main`` before any network access, so a misconfigured
    compliance run never reaches the robot.  ``--admittance`` deliberately
    requires payload gravity compensation: without it the tool's own weight
    feeds the admittance integrator and the offset drifts away in every pose.
    """

    if not getattr(args, "admittance", False):
        return
    gravity_json = getattr(args, "gravity_json", None)
    if not gravity_json or not getattr(args, "gravity_compensation_enable", True):
        raise SystemExit(
            "--admittance requires payload gravity compensation: set "
            "--gravity-compensation-enable true and pass --gravity-json "
            "<identified.json> from tool/identify_ft_payload.py"
        )
    if args.adm_max_offset_m <= 0 or args.adm_max_offset_deg <= 0:
        raise SystemExit("--adm-max-offset-m and --adm-max-offset-deg must be positive")
