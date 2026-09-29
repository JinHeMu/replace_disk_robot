#!/usr/bin/env python3
"""Keyboard Cartesian servo for a real JAKA ZU5.

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
servo or sends commands.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import glfw

from jaka_common import (
    DEFAULT_SENSOR_TO_TOOL_ROTATION,
    DEFAULT_TOOL_ARM_M,
    JakaError,
    JakaRobotAdapter,
    JakaWristFTAdapter,
    RateLoop,
    add_network_args,
    edg_session,
    fmt_array,
    make_client,
)
from replace_disk_robot.contact import (
    SensorCompensationConfig,
    SensorWrenchCompensator,
    WrenchProcessor,
    WrenchProcessorConfig,
)
from replace_disk_robot.control import (
    AdmittanceConfig,
    AdmittanceController,
    CartesianJog,
    CartesianServo,
    KeyControl,
    KeyboardAdmittanceController,
    MotionReferenceConfig,
    ServoConfig,
)
from replace_disk_robot.core import JointState, Pose, Wrench
from replace_disk_robot.core.rotation import (
    multiply as quaternion_multiply,
    quaternion_from_rotation_matrix,
    quaternion_from_rotation_vector,
    rotation_matrix,
    rotation_vector_from_matrix,
)
from replace_disk_robot.kinematics.jaka import JakaKinematics
from replace_disk_robot.safety import ForceLimitGuard


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "examples" / "jaka_driver_tool" /
    "jaka_keyboard_servo_config.yaml"
)


_KEY_TO_NAME = {
    glfw.KEY_W: "w",
    glfw.KEY_S: "s",
    glfw.KEY_A: "a",
    glfw.KEY_D: "d",
    glfw.KEY_R: "r",
    glfw.KEY_F: "f",
    glfw.KEY_Q: "q",
    glfw.KEY_E: "e",
    glfw.KEY_UP: "up",
    glfw.KEY_DOWN: "down",
    glfw.KEY_LEFT: "left",
    glfw.KEY_RIGHT: "right",
}

_RECOVERABLE_FAULTS = {
    None,
    "focus_lost",
    "stopped",
    "command_timeout",
    "force_limit",
    "tracking_error",
    "loop_timeout",
    "invalid_dt",
    "invalid_wrench",
}
_FORCE_RESUME_RATIO = 0.8


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML parameter file (default: {DEFAULT_CONFIG_PATH})",
    )
    add_network_args(parser)
    parser.add_argument("--rate-hz", type=float, default=125.0,
                        help="EDG servo command rate; 125 Hz sends step_num=1")
    parser.add_argument("--linear-speed", type=float, default=0.01,
                        help="keyboard linear speed in m/s (default: 0.01)")
    parser.add_argument("--angular-speed-deg", type=float, default=5.0,
                        help="keyboard angular speed in deg/s (default: 5)")
    parser.add_argument("--command-frame", choices=("base", "tool"), default="base",
                        help=("keyboard velocity frame. base=jaka_base_link (default) for "
                              "W/S, A/D=-X/+X and rotations; tool=those keys in current "
                              "tool0 frame; "
                              "R/F always insert/retract along current tool0 +Z/-Z"))
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="stop after this many seconds; 0 means run until Esc/window close")
    parser.add_argument("--headless", action="store_true",
                        help="run without GLFW and command zero velocity (requires --seconds)")
    parser.add_argument("--dry-run", action="store_true",
                        help="read EDG/F/T and keyboard; never enable servo or send motion")
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument("--plot-wrench", dest="plot_wrench", action="store_true",
                            help="show non-blocking live force/torque curves (default)")
    plot_group.add_argument("--no-plot", dest="plot_wrench", action="store_false",
                            help="disable the default live force/torque plot")
    parser.set_defaults(plot_wrench=True)
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
    parser.add_argument("--no-tare", action="store_true",
                        help="skip the startup F/T tare")
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
        "--no-gravity-compensation",
        action="store_true",
        help="disable payload gravity compensation and use the adapter tare only",
    )
    parser.add_argument(
        "--admittance",
        action="store_true",
        help=("keyboard nominal pose + six-axis admittance compliance. Needs the "
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


def _parse_args() -> argparse.Namespace:
    """Parse CLI options, with JSON config values as defaults.

    Command-line options explicitly supplied by the user override the JSON
    config.  This keeps the config file convenient for long admittance setups
    while still allowing one-off overrides.
    """

    parser = _build_parser()
    preliminary, _unknown = parser.parse_known_args()
    config_path = Path(preliminary.config).expanduser().resolve()
    if not config_path.is_file() and config_path != DEFAULT_CONFIG_PATH.resolve():
        raise SystemExit(f"config file not found: {config_path}")
    config = _load_config(config_path, parser)
    if config:
        parser.set_defaults(**config)
    args = parser.parse_args()
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
    if not gravity_json or getattr(args, "no_gravity_compensation", False):
        raise SystemExit(
            "--admittance requires payload gravity compensation: drop "
            "--no-gravity-compensation and pass --gravity-json "
            "<identified.json> from tool/identify_ft_payload.py"
        )
    if args.adm_max_offset_m <= 0 or args.adm_max_offset_deg <= 0:
        raise SystemExit("--adm-max-offset-m and --adm-max-offset-deg must be positive")


def _ft_transforms(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.identity_transform:
        return np.eye(3), np.zeros(3)
    return DEFAULT_SENSOR_TO_TOOL_ROTATION, DEFAULT_TOOL_ARM_M


def _base_translation_from_keys(
    pressed: set[str],
    tool_z_in_base: np.ndarray,
    speed_m_s: float,
) -> np.ndarray:
    """Build the real-robot base-mode translation command.

    The arm mount ``jaka_base_link`` is yawed -90 degrees relative to the
    mobile platform.  Platform left/right is therefore jaka-base -X/+X.
    R/F deliberately remain on the current tool +Z insertion axis.
    """

    tool_z = np.asarray(tool_z_in_base, dtype=float).copy()
    norm = float(np.linalg.norm(tool_z))
    if tool_z.shape != (3,) or not np.isfinite(tool_z).all() or norm < 1e-12:
        raise ValueError("tool_z_in_base must be a finite non-zero 3-vector")
    tool_z /= norm

    linear = np.zeros(3)
    if "w" in pressed:
        linear += np.array([0.0, 0.0, 1.0])
    if "s" in pressed:
        linear -= np.array([0.0, 0.0, 1.0])
    if "a" in pressed:
        linear -= np.array([1.0, 0.0, 0.0])
    if "d" in pressed:
        linear += np.array([1.0, 0.0, 0.0])
    if "r" in pressed:
        linear += tool_z
    if "f" in pressed:
        linear -= tool_z
    linear /= max(1.0, float(np.linalg.norm(linear)))
    return linear * float(speed_m_s)


SampleCallback = Callable[[float, object, JointState, object, "JakaKeyboardServo"], None]


class JakaKeyboardServo:
    """State machine for real JAKA keyboard Cartesian servo."""

    def __init__(
        self,
        client,
        args: argparse.Namespace,
        *,
        sample_callback: SampleCallback | None = None,
    ) -> None:
        self.client = client
        self.args = args
        self.kinematics = JakaKinematics()
        # The Jaka adapter defaults to ``joint1..joint6`` while Pinocchio and
        # the MuJoCo model use ``joint_1..joint_6``.  Use the kinematics names
        # so measured states and CartesianServo targets share one namespace.
        self.arm = JakaRobotAdapter(
            client,
            joint_names=self.kinematics.joint_names,
        )

        self.base_frame = self.kinematics.base_frame
        self.tool_frame = self.kinematics.end_effector_frame
        self.command_frame = (
            self.tool_frame if args.command_frame == "tool" else self.base_frame
        )

        rotation, tool_arm = _ft_transforms(args)
        self.ft = JakaWristFTAdapter(
            client,
            frame_id="tcp_fts_site",
            sensor_to_tool_rotation=rotation,
            tool_arm_m=tool_arm,
            filter_alpha=args.alpha,
            deadband_force_n=args.deadband_force,
            deadband_torque_nm=args.deadband_torque,
        )
        # Online gravity compensation is configured in initialize() when a
        # payload identification JSON is available.
        self.sensor_frame_id = "tcp_fts_sensor"
        self.sensor_to_tool_rotation = np.asarray(
            DEFAULT_SENSOR_TO_TOOL_ROTATION if args.identity_transform else rotation,
            dtype=float,
        ).copy()
        self.tool_to_sensor_m = np.asarray(
            np.zeros(3) if args.identity_transform else tool_arm,
            dtype=float,
        ).copy()
        self.gravity_compensator: SensorWrenchCompensator | None = None
        self.wrench_processor: WrenchProcessor | None = None
        self.payload_mass_kg: float | None = None
        self.payload_com_sensor_m: np.ndarray | None = None

        self.servo = CartesianServo(
            self.kinematics,
            self.kinematics.joint_limits_rad,
            ServoConfig(
                linear_speed_m_s=args.linear_speed,
                angular_speed_rad_s=np.deg2rad(args.angular_speed_deg),
                max_tracking_error_rad=args.max_joint_error_rad,
            ),
        )
        self.keys = KeyControl(
            args.linear_speed,
            np.deg2rad(args.angular_speed_deg),
            base_frame=self.command_frame,
        )
        self.guard = ForceLimitGuard(
            force_limit_n=args.max_force_n,
            torque_limit_nm=args.max_torque_nm,
        )
        self.sample_callback = sample_callback

        # Compliant stage between the keyboard and the pose servo.  The wrench
        # it consumes is _read_wrench(), i.e. already gravity-compensated.
        self.admittance_enabled = bool(args.admittance)
        self.motion: KeyboardAdmittanceController | None = None
        self.last_external_wrench: Wrench | None = None
        self.last_corrected_pose: Pose | None = None
        if self.admittance_enabled:
            self.admittance = AdmittanceController(AdmittanceConfig(
                frame_id=self.tool_frame,
                mass=args.adm_mass,
                damping=args.adm_damping,
                stiffness=args.adm_stiffness,
                max_velocity=args.adm_max_velocity,
                max_dt_s=max(0.01, 4.0 / args.rate_hz),
            ))
            enabled_axes = np.array(
                [True, True, True, False, False, False]
                if args.admittance_axes == "translation" else [True] * 6,
                dtype=bool,
            )
            axis_index = {"x": 0, "y": 1, "z": 2, "rx": 3, "ry": 4, "rz": 5}
            for axis in getattr(args, "adm_disable_axes", ()):
                enabled_axes[axis_index[axis]] = False
            self.motion_config = MotionReferenceConfig(
                enabled_axes=enabled_axes,
                max_dt_s=max(0.01, 4.0 / args.rate_hz),
            )

        self.last_wrench = None
        self.servo_enabled = False
        self.status = "initializing"
        self._next_print = 0.0
        self._last_now = None
        self._closed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        state0 = self.client.read_edg_state()
        q0 = JointState(self.arm.joint_names, state0.joint_position_rad)
        print(
            f"[keyboard] initial q={fmt_array(q0.position_rad, 4)} "
            f"base={self.base_frame} command_frame={self.command_frame}"
        )

        gravity_json = getattr(self.args, "gravity_json", None)
        use_gravity = bool(gravity_json) and not getattr(
            self.args, "no_gravity_compensation", False
        )
        if use_gravity:
            self._configure_gravity_compensation()
            if getattr(self.args, "tare_compensated", False):
                self._tare_compensated_wrench(state0, q0)
        else:
            if getattr(self.args, "tare_compensated", False):
                raise ValueError(
                    "--tare-compensated requires gravity compensation; "
                    "remove --no-gravity-compensation"
                )
            if not getattr(self.args, "no_tare", False):
                print(f"[keyboard] taring FT with {self.args.tare_samples} samples")
                bias = self.ft.tare(
                    samples=self.args.tare_samples,
                    period_s=self.args.tare_period_ms / 1000.0,
                )
                print(f"[keyboard] FT bias={fmt_array(bias, 4)}")

        self.last_wrench = self._read_wrench(state0, q0)
        if self.admittance_enabled:
            print(
                "[keyboard] admittance: axes="
                f"{'translation' if self.args.admittance_axes == 'translation' else 'all'}, "
                f"offset limit={self.args.adm_max_offset_m * 1000.0:g} mm / "
                f"{self.args.adm_max_offset_deg:g} deg"
            )
            self._print_compliance_seed(q0, state0)

        if self.args.dry_run:
            self.servo.reset(q0)
            self.status = "dry_run"
            print("[keyboard] dry-run: servo not enabled, no motion commands sent")
            return

        self.client.servo_move_enable(True)
        self.servo_enabled = True
        print("[keyboard] servo_move_enable(True)")

        # The robot can settle slightly when servo mode is entered.  Reset the
        # hold target from the post-enable measured state.
        state_servo = self.client.read_edg_state()
        q_servo = JointState(self.arm.joint_names, state_servo.joint_position_rad)
        self.servo.reset(q_servo)
        self.last_wrench = self._read_wrench(state_servo, q_servo)
        self._print_compliance_seed(q_servo, state_servo)
        self.status = "holding"

    def _configure_gravity_compensation(self) -> None:
        path = Path(self.args.gravity_json).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"gravity identification JSON not found: {path}")
        try:
            fit = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read gravity JSON {path}: {exc}") from exc
        if not isinstance(fit, dict):
            raise ValueError(f"gravity JSON root must be an object: {path}")

        try:
            mass = float(fit["mass_kg"])
            h_base = np.asarray(fit["signed_gravity_force_base_n"], dtype=float)
            com_sensor = np.asarray(fit["center_of_mass_sensor_m"], dtype=float)
            force_bias = np.asarray(fit["force_bias_sensor_n"], dtype=float)
            torque_bias = np.asarray(fit["torque_bias_sensor_nm"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"gravity JSON is missing required fields: {exc}") from exc

        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"gravity JSON mass_kg must be positive, got {mass!r}")
        for name, vector in (
            ("signed_gravity_force_base_n", h_base),
            ("center_of_mass_sensor_m", com_sensor),
            ("force_bias_sensor_n", force_bias),
            ("torque_bias_sensor_nm", torque_bias),
        ):
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError(f"gravity JSON {name} must be a finite 3-vector")

        response = fit.get("sensor_to_tool_rotation")
        if response is None:
            sensor_to_tool = np.asarray(DEFAULT_SENSOR_TO_TOOL_ROTATION, dtype=float)
        else:
            sensor_to_tool = np.asarray(response, dtype=float)
        if sensor_to_tool.shape != (3, 3) or not np.isfinite(sensor_to_tool).all():
            raise ValueError("gravity JSON sensor_to_tool_rotation must be finite 3x3")

        response = fit.get("tool_to_sensor_m")
        if response is None:
            tool_to_sensor = np.asarray(DEFAULT_TOOL_ARM_M, dtype=float)
        else:
            tool_to_sensor = np.asarray(response, dtype=float)
        if tool_to_sensor.shape != (3,) or not np.isfinite(tool_to_sensor).all():
            raise ValueError("gravity JSON tool_to_sensor_m must be a finite 3-vector")

        self.sensor_to_tool_rotation = sensor_to_tool
        self.tool_to_sensor_m = tool_to_sensor
        self.payload_mass_kg = mass
        self.payload_com_sensor_m = com_sensor
        self.gravity_compensator = SensorWrenchCompensator(SensorCompensationConfig(
            frame_id=self.sensor_frame_id,
            load_sign=1.0,
            gravity_frame_id=self.base_frame,
            gravity_m_s2=h_base / mass,
            payload_mass_kg=mass,
            payload_com_sensor_m=com_sensor,
        ))
        # The identification result already contains the static sensor bias.
        self.gravity_compensator.bias_sensor = np.r_[force_bias, torque_bias]
        self.wrench_processor = WrenchProcessor(WrenchProcessorConfig(
            frame_id=self.tool_frame,
            load_sign=1.0,
            filter_alpha=self.args.alpha,
            force_deadband_n=self.args.deadband_force,
            torque_deadband_nm=self.args.deadband_torque,
        ))
        print(
            f"[keyboard] gravity compensation ON: {path} "
            f"mass={mass:.6f} kg, |h|={float(np.linalg.norm(h_base)):.4f} N"
        )

    def _sensor_pose_in_base(self, measured: JointState) -> Pose:
        tool_pose = self.kinematics.forward(measured)
        base_from_tool = rotation_matrix(tool_pose.quaternion_wxyz)
        base_from_sensor = base_from_tool @ self.sensor_to_tool_rotation
        sensor_origin = tool_pose.position_m + base_from_tool @ self.tool_to_sensor_m
        return Pose(
            self.base_frame,
            sensor_origin,
            quaternion_from_rotation_matrix(base_from_sensor),
        )

    def _tare_compensated_wrench(self, state, measured: JointState) -> None:
        """Zero the current gravity-compensated sensor wrench.

        The offset is folded into ``SensorWrenchCompensator.bias_sensor`` so the
        low-pass filter and force guard see the already-tared wrench.
        """

        if self.gravity_compensator is None or self.wrench_processor is None:
            raise RuntimeError("gravity compensation is not configured")
        raw = Wrench(
            self.sensor_frame_id,
            state.torque_sensor[:3].copy(),
            state.torque_sensor[3:].copy(),
        )
        sensor_pose = self._sensor_pose_in_base(measured)
        compensated_sensor = self.gravity_compensator.compensate(
            raw,
            sensor_pose,
            payload_mass_kg=self.payload_mass_kg,
            payload_com_sensor_m=self.payload_com_sensor_m,
        )
        offset = compensated_sensor.as_vector()
        self.gravity_compensator.bias_sensor = (
            self.gravity_compensator.bias_sensor + offset
        )
        self.wrench_processor.reset()
        print(
            "[keyboard] gravity-compensated tare: "
            f"offset={fmt_array(offset, 4)}"
        )

    def _gravity_compensated_wrench(self, state, measured: JointState) -> Wrench:
        if self.gravity_compensator is None or self.wrench_processor is None:
            raise RuntimeError("gravity compensation is not configured")
        raw = Wrench(
            self.sensor_frame_id,
            state.torque_sensor[:3].copy(),
            state.torque_sensor[3:].copy(),
        )
        sensor_pose = self._sensor_pose_in_base(measured)
        compensated_sensor = self.gravity_compensator.compensate(
            raw,
            sensor_pose,
            payload_mass_kg=self.payload_mass_kg,
            payload_com_sensor_m=self.payload_com_sensor_m,
        )
        tool_pose = self.kinematics.forward(measured)
        return self.wrench_processor.update_tcp(
            compensated_sensor,
            sensor_pose,
            tool_pose,
        )

    def _read_wrench(self, state, measured: JointState) -> Wrench:
        if self.gravity_compensator is not None and self.wrench_processor is not None:
            return self._gravity_compensated_wrench(state, measured)
        return self.ft.read_wrench_from(state)

    # ------------------------------------------------------------------
    # Admittance compliance
    # ------------------------------------------------------------------
    def _print_compliance_seed(self, measured: JointState, state) -> None:
        """Print the external wrench and offset at a freshly seeded reference."""

        if not self.admittance_enabled:
            return
        self._reset_compliance(measured)
        try:
            wrench = self._read_wrench(state, measured)
        except (ValueError, RuntimeError):
            return
        self.last_wrench = wrench
        self.last_external_wrench = wrench
        print(f"[keyboard] admittance reference: {self.admittance_text()}")

    def _reset_compliance(self, measured: JointState | None) -> None:
        """Re-seed nominal/admittance state at the measured pose.

        Called on start, resume and every stop so that a fault recovery never
        replays a stale compliant offset.
        """

        if not self.admittance_enabled:
            return
        if measured is None:
            try:
                measured = self.arm.read_joint_state()
            except Exception:  # noqa: BLE001 - keep the previous reference
                return
        pose = self.kinematics.forward(measured)
        if self.motion is None:
            self.motion = KeyboardAdmittanceController(
                self.admittance, pose, self.motion_config, tcp_frame_id=self.tool_frame,
            )
        else:
            self.motion.reset(pose)
        if self.wrench_processor is not None:
            self.wrench_processor.reset()
        self.last_external_wrench = None
        self.last_corrected_pose = pose

    def _clamp_offset(self, corrected: Pose, nominal: Pose) -> Pose:
        """Limit the compliant displacement relative to the nominal pose.

        ``AdmittanceController`` intentionally has no offset clamp; on hardware
        the caller owns that limit.
        """

        delta = np.asarray(corrected.position_m, dtype=float) - np.asarray(
            nominal.position_m, dtype=float
        )
        distance = float(np.linalg.norm(delta))
        limit_m = float(self.args.adm_max_offset_m)
        position = (
            np.asarray(nominal.position_m, dtype=float) + delta * (limit_m / distance)
            if distance > limit_m else np.asarray(corrected.position_m, dtype=float)
        )

        relative = (
            rotation_matrix(corrected.quaternion_wxyz)
            @ rotation_matrix(nominal.quaternion_wxyz).T
        )
        rotation_vector = rotation_vector_from_matrix(relative)
        angle = float(np.linalg.norm(rotation_vector))
        limit_rad = float(np.deg2rad(self.args.adm_max_offset_deg))
        if angle > limit_rad:
            # Scale the *relative* rotation and apply it to the nominal pose;
            # the corrected quaternion is already nominal-composed.
            delta_quaternion = quaternion_from_rotation_vector(
                rotation_vector * (limit_rad / angle)
            )
            quaternion = quaternion_multiply(delta_quaternion, np.asarray(
                nominal.quaternion_wxyz, dtype=float
            ))
            quaternion = quaternion / np.linalg.norm(quaternion)
        else:
            quaternion = np.asarray(corrected.quaternion_wxyz, dtype=float)
        return Pose(nominal.frame_id, position, quaternion)

    def _update_compliance(self, jog, wrench, tcp_pose, dt_s) -> Pose:
        """One keyboard-nominal + admittance step; returns the clamped pose."""

        if self.motion is None:
            raise RuntimeError("admittance is enabled but the motion reference is not seeded")
        corrected = self.motion.update(jog, wrench, tcp_pose, dt_s)
        self.last_corrected_pose = self._clamp_offset(corrected, self.motion.nominal_pose)
        return self.last_corrected_pose

    def admittance_text(self) -> str:
        """Compact admittance state for the status line and window title."""

        if not self.admittance_enabled:
            return ""
        force = 0.0
        if self.last_external_wrench is not None:
            force = float(np.linalg.norm(self.last_external_wrench.force_n))
        offset = self.motion.state().offset if self.motion is not None else np.zeros(6)
        axes = self.motion.enabled_axes if self.motion is not None else np.zeros(6, dtype=bool)
        return (
            f"|Fext|={force:5.2f}N "
            f"dx={np.linalg.norm(offset[:3]) * 1000.0:4.1f}mm "
            f"dth={np.rad2deg(np.linalg.norm(offset[3:])):4.1f}deg "
            f"axes={''.join('1' if value else '0' for value in axes)}"
        )

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.keys.clear()
        if self.servo_enabled:
            try:
                self.client.servo_move_enable(False)
                print("[keyboard] servo_move_enable(False)")
            except Exception as exc:  # noqa: BLE001 - best-effort stop
                print(f"[keyboard] servo-disable warning: {exc}", file=sys.stderr)
            finally:
                self.servo_enabled = False

    # ------------------------------------------------------------------
    # Safety / command helpers
    # ------------------------------------------------------------------
    def _jog_to_servo(self, reference: JointState | Pose | None = None) -> CartesianJog:
        """Map keyboard keys to Servo's base-linear/intrinsic-TCP convention.

        * tool mode: W/S/A/D and rotations use the current tool0 frame.  R/F
          still follow the current tool0 +Z/-Z insertion axis.
        * base mode: W/S use base +/-Z; A/D use base -X/+X, which corresponds
          to platform left/right for the -90-degree arm mounting; R/F move
          along current tool0 +Z/-Z.  Base-axis rotations are converted to
          intrinsic TCP angular velocity.

        ``reference`` overrides the pose the keys are interpreted in; the
        admittance path passes the *nominal* pose so that compliant offsets do
        not feed back into the keyboard frame.
        """
        if reference is None:
            target = self.servo.target
            if target is None:
                target = self.arm.read_joint_state()
            pose = self.kinematics.forward(target)
        elif isinstance(reference, Pose):
            pose = reference
        else:
            pose = self.kinematics.forward(reference)
        base_from_tool = rotation_matrix(pose.quaternion_wxyz)

        if self.command_frame == self.tool_frame:
            command = self.keys.command(forward_axis=np.array([0.0, 0.0, 1.0]))
            return CartesianJog(
                self.base_frame,
                base_from_tool @ command.linear_m_s,
                command.angular_rad_s,
            )

        command = self.keys.command()
        linear = _base_translation_from_keys(
            self.keys.pressed,
            base_from_tool[:, 2],
            self.args.linear_speed,
        )
        return CartesianJog(
            self.base_frame,
            linear,
            base_from_tool.T @ command.angular_rad_s,
        )

    def stop(self, reason: str, measured: JointState | None = None) -> None:
        self.keys.clear()
        if measured is None:
            try:
                measured = self.arm.read_joint_state()
            except Exception:  # noqa: BLE001 - still record fault
                measured = None

        if measured is not None:
            self.servo.halt(measured, reason)
            if self.servo_enabled and not self.args.dry_run:
                try:
                    self.arm.command_joint_positions(measured)
                except Exception as exc:  # noqa: BLE001 - best-effort hold
                    print(f"[keyboard] hold-command warning: {exc}", file=sys.stderr)
        self._reset_compliance(measured)
        self.status = reason

    def resume(self) -> bool:
        """Resume a recoverable stop from fresh, verified robot feedback.

        Enter never clears a force stop while the load is still close to the
        trip threshold.  Resetting the Cartesian target to the current measured
        joints prevents replaying the target that existed before the fault.
        """

        self.keys.clear()
        fault = self.servo.fault
        if self.args.dry_run:
            print("[keyboard] resume ignored in dry-run")
            return False
        if fault not in _RECOVERABLE_FAULTS:
            print(f"[keyboard] {fault} is not recoverable with Enter; restart after inspection")
            return False

        try:
            state = self.client.read_edg_state()
            measured = JointState(self.arm.joint_names, state.joint_position_rad)
            wrench = self._read_wrench(state, measured)
        except Exception as exc:  # noqa: BLE001 - stay stopped on bad feedback
            print(f"[keyboard] resume denied: feedback unavailable: {exc}", file=sys.stderr)
            return False

        wrench_vector = wrench.as_vector()
        if not np.isfinite(wrench_vector).all():
            print("[keyboard] resume denied: wrench is not finite", file=sys.stderr)
            return False
        force = float(np.linalg.norm(wrench.force_n))
        torque = float(np.linalg.norm(wrench.torque_nm))
        force_resume_limit = _FORCE_RESUME_RATIO * self.args.max_force_n
        torque_resume_limit = _FORCE_RESUME_RATIO * self.args.max_torque_nm
        if force > force_resume_limit or torque > torque_resume_limit:
            print(
                "[keyboard] resume denied: release the load first; "
                f"|F|={force:.3f} N (need <= {force_resume_limit:.3f}), "
                f"|T|={torque:.3f} Nm (need <= {torque_resume_limit:.3f})",
                file=sys.stderr,
            )
            return False

        try:
            self.servo.reset(measured)
        except (ValueError, RuntimeError) as exc:
            print(f"[keyboard] resume denied: invalid joint state: {exc}", file=sys.stderr)
            return False
        self.last_wrench = wrench
        self.status = "holding"
        print(
            f"[keyboard] resumed from measured pose after {fault or 'hold'}; "
            f"|F|={force:.3f} N |T|={torque:.3f} Nm"
        )
        return True

    def handle_focus(self, focused: bool) -> None:
        self.keys.clear()
        if not focused and self.servo.fault is None:
            self.stop("focus_lost")
            print("[keyboard] focus lost: holding measured pose")
        elif focused and self.servo.fault == "focus_lost":
            self.resume()

    def _status_text(self) -> str:
        if self.servo.fault:
            if self.servo.fault == "force_limit":
                return "force_limit: release load, then press Enter"
            if self.servo.fault in _RECOVERABLE_FAULTS:
                return f"{self.servo.fault}: press Enter to re-anchor and resume"
            return f"{self.servo.fault}: restart script after inspection"
        return self.status

    # ------------------------------------------------------------------
    # One control tick
    # ------------------------------------------------------------------
    def tick(self, elapsed_s: float, dt_s: float) -> None:
        state = self.client.read_edg_state()
        measured = JointState(self.arm.joint_names, state.joint_position_rad)
        try:
            wrench = self._read_wrench(state, measured)
        except (ValueError, RuntimeError) as exc:
            self.stop("invalid_wrench", measured)
            print(f"[keyboard] invalid wrench: {exc}", file=sys.stderr)
            return
        self.last_wrench = wrench

        # Optional instrumentation hook used by data-collection tools.  The
        # callback receives the same EDG packet that drives this control tick,
        # so joint state, raw F/T data and robot pose remain synchronized.
        if self.sample_callback is not None:
            self.sample_callback(elapsed_s, state, measured, wrench, self)

        if not np.isfinite(dt_s) or dt_s <= 0:
            self.stop("invalid_dt", measured)
            return
        if dt_s > self.servo.config.max_dt_s:
            self.stop("loop_timeout", measured)
            return

        jog = self._jog_to_servo(
            self.motion.nominal_pose if self.motion is not None else None
        )
        if self.args.dry_run:
            # Still integrate the compliance chain so the printed offset and
            # the sign of the external force can be verified before enabling
            # Servo; no command is submitted in dry-run.
            if self.admittance_enabled and self.servo.fault is None:
                self._update_compliance(
                    jog, wrench, self.kinematics.forward(measured), dt_s
                )
                self.last_external_wrench = wrench
            self.status = "dry_run"
            self._print_line(elapsed_s, measured, measured.position_rad, wrench)
            return

        if self.admittance_enabled:
            self.last_external_wrench = wrench
            # A latched fault must not let the integrator wind up; the offset is
            # re-seeded from the measured pose on resume.
            if self.servo.fault is None:
                corrected = self._update_compliance(
                    jog, wrench, self.kinematics.forward(measured), dt_s
                )
                self.servo.submit_pose(corrected, elapsed_s)
        else:
            self.servo.submit(jog, elapsed_s)
        try:
            target = self.servo.update(measured, dt_s, elapsed_s)
        except (ValueError, RuntimeError) as exc:
            self.stop("servo_error", measured)
            print(f"[keyboard] servo error: {exc}", file=sys.stderr)
            return

        safe_q, tripped = self.guard.filter_arm_target(
            wrench.as_vector(),
            measured.position_rad,
            target.position_rad,
        )
        if tripped:
            self.stop("force_limit", measured)
            safe_q = measured.position_rad
        else:
            self.status = self.servo.status

        self.arm.command_joint_positions(
            JointState(self.arm.joint_names, safe_q)
        )
        self._print_line(elapsed_s, measured, safe_q, wrench)

    def _print_line(self, elapsed_s, measured, command_q, wrench) -> None:
        if elapsed_s < self._next_print:
            return
        self._next_print += 0.5
        force = float(np.linalg.norm(wrench.force_n))
        torque = float(np.linalg.norm(wrench.torque_nm))
        admittance = self.admittance_text()
        print(
            f"[keyboard] t={elapsed_s:6.3f}s "
            f"q={fmt_array(measured.position_rad, 4)} "
            f"cmd={fmt_array(command_q, 4)} "
            f"|F|={force:6.3f}N |T|={torque:6.3f}Nm "
            + (f"{admittance} " if admittance else "")
            + f"{self._status_text()}"
        )


# ----------------------------------------------------------------------
# GLFW keyboard front end
# ----------------------------------------------------------------------
def _on_key(window, key, scancode, action, mods) -> None:  # noqa: ARG001
    app = glfw.get_window_user_pointer(window)
    if app is None:
        return
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
        return
    if key == glfw.KEY_SPACE and action == glfw.PRESS:
        app.stop("stopped")
        return
    if key == glfw.KEY_ENTER and action == glfw.PRESS:
        app.resume()
        return
    name = _KEY_TO_NAME.get(key)
    if name is None:
        return
    if action in (glfw.PRESS, glfw.REPEAT):
        app.keys.press(name)
    elif action == glfw.RELEASE:
        app.keys.release(name)


def _on_focus(window, focused: bool) -> None:
    app = glfw.get_window_user_pointer(window)
    if app is not None:
        app.handle_focus(bool(focused))


def _window_title(app: JakaKeyboardServo) -> str:
    force = 0.0
    torque = 0.0
    if app.last_wrench is not None:
        force = float(np.linalg.norm(app.last_wrench.force_n))
        torque = float(np.linalg.norm(app.last_wrench.torque_nm))
    admittance = app.admittance_text()
    return (
        f"JAKA keyboard servo | {app.command_frame} | "
        f"F={force:.2f} N | T={torque:.2f} Nm | "
        + (f"{admittance} | " if admittance else "")
        + f"{app._status_text()}"
    )


def _run_window(app: JakaKeyboardServo) -> None:
    if not glfw.init():
        raise RuntimeError("cannot initialize GLFW display; use --headless")

    window = None
    clear_window = None
    plotter = None
    plot_error_reported = False
    try:
        # Use a normal OpenGL-capable window and clear it every frame.  A
        # GLFW_NO_API window can be invisible on some Wayland compositors
        # because no buffer is ever attached, which makes keyboard focus hard.
        try:
            from OpenGL.GL import GL_COLOR_BUFFER_BIT, glClear, glClearColor
            clear_window = (glClear, glClearColor, GL_COLOR_BUFFER_BIT)
        except Exception:
            clear_window = None

        window = glfw.create_window(620, 220, "JAKA keyboard servo", None, None)
        if window is None:
            raise RuntimeError("cannot create GLFW window; use --headless")

        glfw.make_context_current(window)
        glfw.swap_interval(0)
        glfw.set_window_user_pointer(window, app)
        glfw.set_key_callback(window, _on_key)
        glfw.set_window_focus_callback(window, _on_focus)

        # Bring the control window to the front.  Some compositors deny
        # focus stealing, so the operator may still need to click it.
        glfw.show_window(window)
        glfw.focus_window(window)
        glfw.set_window_title(window, _window_title(app))

        if app.args.plot_wrench:
            import os
            import tempfile

            os.environ.setdefault(
                "MPLCONFIGDIR",
                str(Path(tempfile.gettempdir()) / "replace_disk_robot_matplotlib"),
            )
            from replace_disk_robot.visual import ProcessTypePlotter
            plotter = ProcessTypePlotter(window_s=10.0, refresh_hz=10.0)

        print(
            "[keyboard] a small window named 'JAKA keyboard servo' has opened.\n"
            "[keyboard] Click that window, then hold keys there. "
            "Space stop, Enter resume, Esc exit."
            + (
                "\n[keyboard] admittance is on: keys move the nominal pose, the "
                "gravity-compensated external wrench moves the compliant offset."
                if app.admittance_enabled else ""
            )
        )
        loop = RateLoop(app.args.rate_hz)
        last_now = None
        for elapsed_s, now in loop:
            glfw.poll_events()
            if glfw.window_should_close(window):
                break
            dt_s = 1.0 / app.args.rate_hz if last_now is None else now - last_now
            last_now = now
            app.tick(elapsed_s, dt_s)
            if plotter is not None and app.last_wrench is not None:
                plotter.update(app.last_wrench, elapsed_s)
                if plotter.error and not plot_error_reported:
                    print(
                        f"[plot] plot disabled; keyboard control remains active:\n"
                        f"{plotter.error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    plot_error_reported = True
            glfw.set_window_title(window, _window_title(app))
            if clear_window is not None:
                glClear, glClearColor, color_bit = clear_window
                glClearColor(0.72, 0.80, 0.90, 1.0)
                glClear(color_bit)
                glfw.swap_buffers(window)
            if app.args.seconds > 0 and elapsed_s >= app.args.seconds:
                break
    finally:
        if plotter is not None:
            plotter.close()
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()


def _run_headless(app: JakaKeyboardServo) -> None:
    loop = RateLoop(app.args.rate_hz)
    last_now = None
    for elapsed_s, now in loop:
        dt_s = 1.0 / app.args.rate_hz if last_now is None else now - last_now
        last_now = now
        app.tick(elapsed_s, dt_s)
        if elapsed_s >= app.args.seconds:
            break


def main() -> None:
    args = _parse_args()
    if args.rate_hz <= 0:
        raise SystemExit("--rate-hz must be positive")
    if args.seconds < 0:
        raise SystemExit("--seconds must be non-negative")
    if args.headless and args.seconds <= 0:
        raise SystemExit("--headless requires --seconds > 0")
    _validate_admittance_args(args)

    client = make_client(args)
    app = None
    try:
        with edg_session(client, torque_sensor_mode=args.torque_sensor_mode) as client:
            app = JakaKeyboardServo(client, args)
            try:
                app.initialize()
                if args.headless:
                    _run_headless(app)
                else:
                    _run_window(app)
            finally:
                app.shutdown()
    except KeyboardInterrupt:
        print("\n[keyboard] interrupted by user")
    except (JakaError, RuntimeError, ValueError) as exc:
        print(f"[keyboard] FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        if app is not None:
            app.shutdown()


if __name__ == "__main__":
    main()
