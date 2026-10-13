#!/usr/bin/env python3
"""Collect static-pose data for six-axis F/T payload gravity identification.

This tool reuses the real JAKA Joy Cartesian servo. Move to a distinct,
collision-free tool orientation with LB held, then release LB: no
capture key is needed any more.  The collector samples the EDG packet
continuously at ``--capture-rate-hz`` (5 Hz by default), notices that the arm has
stopped, and saves one static window for that pose by itself.  Move the arm
again and the next stopped pose is recorded automatically.  The CSV contains
synchronized EDG joint state, raw sensor wrench and the sensor orientation
computed from JAKA forward kinematics.

The 125 Hz control loop is unchanged; only the CSV recording is decimated to
5 Hz, because the JAKA Servo must keep receiving joint targets at its command
rate and ``ServoConfig.max_dt_s`` rejects control ticks slower than 20 Hz.
Samples that are not part of an accepted static window are still written, with
``capture_id=-1``, so the file remains a complete, monotonically timed 5 Hz
record of the session. Use ``identify_ft_payload.py --static-poses`` to fit only
accepted static windows.

The script commands real hardware unless --dry-run is supplied.  It does not
power on or enable the robot; use the repository's jaka_start.py first.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIVER_TOOL_DIR = PROJECT_ROOT / "examples" / "jaka_driver_tool"
SRC_DIR = PROJECT_ROOT / "src"
for search_path in (DRIVER_TOOL_DIR, SRC_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from jaka_common import (  # noqa: E402
    JakaError,
    RateLoop,
    add_network_args,
    edg_session,
    make_client,
    parse_bool,
)
from replace_disk_robot.applications.jaka_joy.joy_node import JakaJoyServo  # noqa: E402
from replace_disk_robot.adapters.joy import XboxController  # noqa: E402
from replace_disk_robot.control import JoyControl  # noqa: E402
from replace_disk_robot.applications.common.wrench import (  # noqa: E402
    default_tool0_ft_transforms,
)
from replace_disk_robot.core.rotation import rotation_matrix  # noqa: E402


CSV_COLUMNS = (
    "time_s", "utc_iso", "capture_id", "capture_phase", "command_active",
    *(f"q{i}_rad" for i in range(1, 7)),
    *(f"qd{i}_rad_s" for i in range(1, 7)),
    "tcp_x_m", "tcp_y_m", "tcp_z_m", "tcp_rx_rad", "tcp_ry_rad", "tcp_rz_rad",
    "tool_qw", "tool_qx", "tool_qy", "tool_qz",
    *(f"r_base_sensor_{row}{col}" for row in range(3) for col in range(3)),
    "raw_fx_n", "raw_fy_n", "raw_fz_n", "raw_tx_nm", "raw_ty_nm", "raw_tz_nm",
    "processed_fx_n", "processed_fy_n", "processed_fz_n",
    "processed_tx_nm", "processed_ty_nm", "processed_tz_nm",
    *(f"r_sensor_tool_{row}{col}" for row in range(3) for col in range(3)),
    "tool_to_sensor_x_m", "tool_to_sensor_y_m", "tool_to_sensor_z_m",
)

# Default --min-samples of identify_ft_payload.py.  Shorter windows are still
# recorded, but that tool skips them unless its own limit is lowered.
IDENTIFY_DEFAULT_MIN_SAMPLES = 50


def _capture_input_ready(app, elapsed_s: float) -> bool:
    sample = app.joy_sample
    return bool(app.joy.valid(elapsed_s) and app.joy.neutral()
                and not {"lb", "b", "back"}.intersection(sample.buttons)
                and not app.resume_required)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_network_args(parser)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "tool" / "ft_gravity_samples.csv",
        help="output CSV path",
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing output CSV")
    parser.add_argument("--rate-hz", type=float, default=125.0,
                        help="EDG control/Servo loop rate; CSV sampling is decimated separately")
    parser.add_argument("--capture-rate-hz", type=float, default=5.0,
                        help="CSV sampling rate (default: 5 Hz, i.e. one row every 200 ms)")
    parser.add_argument("--linear-speed", type=float, default=0.005)
    parser.add_argument("--angular-speed-deg", type=float, default=1.0)
    parser.add_argument("--command-frame", choices=("base", "tool"), default="base")
    parser.add_argument("--controller-index", type=int, default=0)
    parser.add_argument("--joy-deadzone", type=float, default=0.15)
    parser.add_argument("--joy-precision-scale", type=float, default=0.25)
    parser.add_argument("--joy-timeout-s", type=float, default=0.15)
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="optional total duration; 0 runs until Back")
    parser.add_argument("--dry-run", action="store_true",
                        help="read and capture only; never enable Servo or send commands")
    parser.add_argument("--settle-seconds", type=float, default=0.75,
                        help="continuous still time required before a window starts")
    parser.add_argument("--capture-seconds", type=float, default=10.0,
                        help=("length of each accepted static window; at 5 Hz the default "
                              f"gives 51 rows, which meets the default --min-samples "
                              f"{IDENTIFY_DEFAULT_MIN_SAMPLES} of identify_ft_payload.py"))
    parser.add_argument("--start-delay-seconds", type=float, default=2.0,
                        help="ignore motion/stillness for this long after startup")
    parser.add_argument("--stationary-speed-rad-s", type=float, default=0.01,
                        help="maximum absolute joint velocity that still counts as still")
    parser.add_argument("--max-force-n", type=float, default=10.0)
    parser.add_argument("--max-torque-nm", type=float, default=2.0)
    parser.add_argument("--max-joint-error-rad", type=float, default=0.15)
    parser.add_argument("--torque-sensor-mode", type=int, default=1,
                        help="JAKA sensor mode; it must expose uncompensated EDG wrench")
    parser.add_argument("--tare-samples", type=int, default=50,
                        help="tare used only by the live safety gate, not by saved raw data")
    parser.add_argument("--tare-period-ms", type=float, default=10.0)
    parser.add_argument("--tare-enable", type=parse_bool, default=True, metavar="true|false",
                        help="enable safety-gate tare; saved raw data is never tared")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--deadband-force", type=float, default=1.0)
    parser.add_argument("--deadband-torque", type=float, default=0.2)
    parser.add_argument("--identity-transform", action="store_true",
                        help="use identity sensor-to-tool rotation (diagnostics only)")
    # Calibration must not load the old payload fit or run admittance.
    parser.set_defaults(headless=True, plot_wrench_enable=False, admittance=False,
                        gravity_json=None, gravity_compensation_enable=False)
    args = parser.parse_args(argv)
    JoyControl(args.linear_speed, np.deg2rad(args.angular_speed_deg),
               deadzone=args.joy_deadzone, precision_scale=args.joy_precision_scale,
               timeout_s=args.joy_timeout_s)
    return args


class GravityDatasetWriter:
    """Automatic still-pose detector and synchronized CSV writer.

    The state machine is evaluated on the decimated capture ticks:

    * ``idle``: the arm is moving, a window is not allowed yet, or a window was
      just saved and the arm has not moved since.  Rows get ``capture_id=-1``.
    * ``settling``: the arm has been still for less than ``--settle-seconds``.
    * ``recording``: a tentative static window.  Its rows are buffered in memory
      and only committed with a real ``capture_id`` once the window is complete,
      so a window that is interrupted by motion never reaches identification.
      The buffered rows are then written anyway with ``capture_id=-1`` to keep
      the 5 Hz record of the session complete.
    """

    def __init__(self, path: Path, args: argparse.Namespace) -> None:
        self.path = path
        self.args = args
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if args.overwrite else "x"
        self.file = self.path.open(mode, newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_COLUMNS)
        self.writer.writeheader()

        self.capture_period_s = 1.0 / args.capture_rate_hz
        # A window that lasts --capture-seconds at --capture-rate-hz contains
        # both endpoints, hence the +1.
        self.capture_rows_target = max(
            2, int(args.capture_seconds * args.capture_rate_hz) + 1
        )
        self.phase = "idle"
        self.armed = True
        self.capture_id = 0
        self.stable_since_s: float | None = None
        self.stable_seconds = 0.0
        self.recorded_rows = 0
        self.capture_rows: list[dict[str, object]] = []
        self.completed_captures = 0
        self.rejected_windows = 0
        self.written_rows = 0
        self.app: JakaJoyServo | None = None
        self._next_capture_s = 0.0
        self._interval_max_speed = 0.0
        self._interval_input_ready = True
        self._last_capture_q = None

    @property
    def status_text(self) -> str:
        if self.phase == "settling":
            return f"settling {self.stable_seconds:.1f}/{self.args.settle_seconds:g}s"
        if self.phase == "recording":
            return (f"recording #{self.capture_id} "
                    f"{self.recorded_rows}/{self.capture_rows_target} rows")
        return "armed" if self.armed else "move arm to re-arm"

    def close(self) -> None:
        if self.file.closed:
            return
        self._flush_rejected_window("collector stopped")
        self.file.flush()
        self.file.close()

    # ------------------------------------------------------------------
    # Capture state machine
    # ------------------------------------------------------------------
    def _update_phase(self, elapsed_s: float, max_joint_speed: float, app,
                      input_ready: bool) -> None:
        """Advance the state machine for one decimated capture tick.

        ``max_joint_speed`` is the largest absolute joint speed seen since the
        previous tick, so a short twitch between two 5 Hz samples still cancels
        a window.  A latched Servo fault blocks new windows, exactly like the
        old manual capture request did.
        """

        self.stable_seconds = 0.0
        stationary = (
            app.servo.fault in (None, "stopped")
            and input_ready
            and elapsed_s >= self.args.start_delay_seconds
            and max_joint_speed <= self.args.stationary_speed_rad_s
        )

        if self.phase == "settling":
            if not stationary:
                self.phase = "idle"
                self.stable_since_s = None
                return
            started_s = (
                self.stable_since_s if self.stable_since_s is not None else elapsed_s
            )
            self.stable_seconds = elapsed_s - started_s
            if self.stable_seconds >= self.args.settle_seconds:
                self.phase = "recording"
                self.recorded_rows = 0
                self.capture_rows = []
                print(
                    f"[capture] #{self.capture_id}: still for "
                    f"{self.stable_seconds:.2f}s; recording "
                    f"{self.capture_rows_target} rows at "
                    f"{self.args.capture_rate_hz:g} Hz"
                )
            return

        if self.phase == "recording":
            if not stationary:
                fault = app.servo.fault
                reason = (f"Servo fault: {fault}" if fault not in (None, "stopped")
                          else "controller not neutral/ready" if not input_ready
                          else f"arm moved ({max_joint_speed:.4f} rad/s)")
                self._flush_rejected_window(reason)
                self.phase = "idle"
                self.armed = True
            return

        # idle
        if not stationary:
            # Input changes alone must not produce duplicate static poses.
            if max_joint_speed > self.args.stationary_speed_rad_s:
                self.armed = True
            return
        if not self.armed:
            return
        self.phase = "settling"
        self.stable_since_s = elapsed_s

    def _flush_rejected_window(self, reason: str) -> None:
        """Write a tentative window out as ordinary (capture_id=-1) rows."""

        if not self.capture_rows:
            return
        rows = self.capture_rows
        self.capture_rows = []
        self.recorded_rows = 0
        for row in rows:
            row["capture_id"] = -1
            row["capture_phase"] = "rejected"
        self.writer.writerows(rows)
        self.written_rows += len(rows)
        self.rejected_windows += 1
        self.file.flush()
        print(
            f"[capture] #{self.capture_id} not saved: {reason}; its "
            f"{len(rows)} still rows stay in the CSV with capture_id=-1, and the "
            f"pose is re-armed automatically"
        )

    def _commit_window(self) -> None:
        completed_id = self.capture_id
        rows = self.capture_rows
        self._last_capture_q = np.array([float(rows[-1][f"q{i}_rad"]) for i in range(1, 7)])
        self.writer.writerows(rows)
        self.written_rows += len(rows)
        self.capture_rows = []
        self.recorded_rows = 0
        self.phase = "idle"
        # Wait for real motion before the next window, otherwise the same still
        # pose would be recorded again and again.
        self.armed = False
        self.completed_captures += 1
        self.capture_id += 1
        self.file.flush()
        print(
            f"[capture] saved #{completed_id}: {len(rows)} rows; "
            f"total={self.completed_captures}. Move the arm to a new orientation "
            f"for the next pose."
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _build_row(self, elapsed_s, state, measured, wrench, app) -> dict[str, object]:
        pose = app.kinematics.forward(measured)
        r_base_tool = rotation_matrix(pose.quaternion_wxyz)
        default_rotation, default_tool_arm = default_tool0_ft_transforms()
        r_sensor_tool = (
            np.eye(3)
            if self.args.identity_transform
            else default_rotation
        )
        r_base_sensor = r_base_tool @ r_sensor_tool
        tool_to_sensor = (
            np.zeros(3)
            if self.args.identity_transform
            else default_tool_arm
        )
        row: dict[str, object] = {
            "time_s": f"{elapsed_s:.9f}",
            "utc_iso": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "capture_id": self.capture_id if self.phase == "recording" else -1,
            "capture_phase": self.phase,
            "command_active": int(bool(app._active and not app.joy.neutral())),
        }
        row.update({f"q{i}_rad": f"{value:.12g}" for i, value in enumerate(state.joint_position_rad, 1)})
        row.update({f"qd{i}_rad_s": f"{value:.12g}" for i, value in enumerate(state.joint_velocity_rad_s, 1)})
        tcp = np.r_[state.tcp_position_m, state.tcp_rpy_rad]
        for name, value in zip(
            ("tcp_x_m", "tcp_y_m", "tcp_z_m", "tcp_rx_rad", "tcp_ry_rad", "tcp_rz_rad"),
            tcp,
            strict=True,
        ):
            row[name] = f"{value:.12g}"
        for name, value in zip(("tool_qw", "tool_qx", "tool_qy", "tool_qz"), pose.quaternion_wxyz, strict=True):
            row[name] = f"{value:.12g}"
        for index, value in enumerate(r_base_sensor.reshape(-1)):
            row[f"r_base_sensor_{index // 3}{index % 3}"] = f"{value:.12g}"
        for name, value in zip(
            ("raw_fx_n", "raw_fy_n", "raw_fz_n", "raw_tx_nm", "raw_ty_nm", "raw_tz_nm"),
            state.torque_sensor,
            strict=True,
        ):
            row[name] = f"{value:.12g}"
        for name, value in zip(
            ("processed_fx_n", "processed_fy_n", "processed_fz_n", "processed_tx_nm", "processed_ty_nm", "processed_tz_nm"),
            wrench.as_vector(),
            strict=True,
        ):
            row[name] = f"{value:.12g}"
        for index, value in enumerate(r_sensor_tool.reshape(-1)):
            row[f"r_sensor_tool_{index // 3}{index % 3}"] = f"{value:.12g}"
        for name, value in zip(
            ("tool_to_sensor_x_m", "tool_to_sensor_y_m", "tool_to_sensor_z_m"),
            tool_to_sensor,
            strict=True,
        ):
            row[name] = f"{value:.12g}"
        return row

    def __call__(self, elapsed_s, state, measured, wrench, app) -> None:
        """``JakaJoyServo`` sample hook, called once per control tick."""

        self.app = app
        self._interval_input_ready &= _capture_input_ready(app, elapsed_s)
        # Precision-mode motion can remain below the still-speed threshold.
        # Re-arm from actual displacement too, without counting reconnects as poses.
        if (self._last_capture_q is not None
                and np.max(np.abs(state.joint_position_rad - self._last_capture_q)) > 0.002):
            self.armed = True
        joint_speed = float(np.max(np.abs(state.joint_velocity_rad_s)))
        self._interval_max_speed = (
            max(self._interval_max_speed, joint_speed)
            if math.isfinite(joint_speed)
            else math.inf
        )

        if elapsed_s < self._next_capture_s:
            return
        # Keep a fixed sampling grid; a late tick must not burst-write rows.
        self._next_capture_s += self.capture_period_s
        if self._next_capture_s <= elapsed_s:
            self._next_capture_s = elapsed_s + self.capture_period_s
        interval_max_speed = self._interval_max_speed
        self._interval_max_speed = 0.0
        interval_input_ready = self._interval_input_ready
        self._interval_input_ready = True

        self._update_phase(elapsed_s, interval_max_speed, app, interval_input_ready)
        row = self._build_row(elapsed_s, state, measured, wrench, app)
        if self.phase == "recording":
            self.capture_rows.append(row)
            self.recorded_rows += 1
            if self.recorded_rows >= self.capture_rows_target:
                self._commit_window()
        else:
            self.writer.writerow(row)
            self.written_rows += 1
        self.file.flush()


def _run_joy(app: JakaJoyServo, source: XboxController, dataset: GravityDatasetWriter) -> None:
    """Poll the controller and run Servo in the same elapsed-time domain."""
    print(
        "[collector] Center inputs, release LB, then hold LB to move. "
        "Release LB and keep inputs neutral to record; move again for the next pose. "
        "RB precision; B stop; A resume; Back exit. "
        f"Each window has {dataset.capture_rows_target} rows at "
        f"{app.args.capture_rate_hz:g} Hz. Aim for >=12 diverse orientations."
    )
    start = time.monotonic()
    previous = None
    next_status = 0.0
    for _elapsed, _now in RateLoop(app.args.rate_hz):
        sample = source.poll()
        elapsed = time.monotonic() - start
        sample = replace(sample, timestamp_s=sample.timestamp_s - start)
        dt = 1.0 / app.args.rate_hz if previous is None else elapsed - previous
        previous = elapsed
        app.apply_sample(sample, elapsed)
        if app.exit_requested:
            break
        app.tick(elapsed, dt)
        if elapsed >= next_status:
            print(f"[collector] {app._status_text()} | {dataset.status_text} "
                  f"saved={dataset.completed_captures}")
            next_status = elapsed + 1.0
        if app.args.seconds > 0 and elapsed >= app.args.seconds:
            break
    print(
        f"[collector] finished: {dataset.completed_captures} captures "
        f"({dataset.rejected_windows} interrupted), {dataset.written_rows} rows, "
        f"CSV={dataset.path}"
    )


def main() -> None:
    args = _parse_args()
    if args.rate_hz <= 0 or args.seconds < 0:
        raise SystemExit("--rate-hz must be positive and --seconds non-negative")
    if args.capture_rate_hz <= 0:
        raise SystemExit("--capture-rate-hz must be positive")
    if args.capture_rate_hz > args.rate_hz:
        raise SystemExit("--capture-rate-hz must not exceed --rate-hz")
    if args.settle_seconds <= 0 or args.capture_seconds <= 0:
        raise SystemExit("--settle-seconds and --capture-seconds must be positive")
    if args.start_delay_seconds < 0:
        raise SystemExit("--start-delay-seconds must be non-negative")
    if args.stationary_speed_rad_s <= 0:
        raise SystemExit("--stationary-speed-rad-s must be positive")

    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"output CSV already exists: {args.output.resolve()} "
            "(choose another --output or pass --overwrite)"
        )
    app = source = dataset = None
    try:
        source = XboxController(args.controller_index)
        print(f"[collector] {source.name}: {source.mapping}")
        dataset = GravityDatasetWriter(args.output.resolve(), args)
        print(
            f"[collector] control loop {args.rate_hz:g} Hz, CSV {args.capture_rate_hz:g} Hz, "
            f"{dataset.capture_rows_target} rows per accepted window"
        )
        if dataset.capture_rows_target < IDENTIFY_DEFAULT_MIN_SAMPLES:
            print(
                f"[collector] note: {dataset.capture_rows_target} rows/window is below the "
                f"default --min-samples {IDENTIFY_DEFAULT_MIN_SAMPLES} of "
                "identify_ft_payload.py; pass a smaller --min-samples there"
            )
        client = make_client(args)
        with edg_session(client, torque_sensor_mode=args.torque_sensor_mode) as client:
            app = JakaJoyServo(client, args, sample_callback=dataset)
            dataset.app = app
            try:
                app.initialize()
                _run_joy(app, source, dataset)
            finally:
                app.shutdown()
    except KeyboardInterrupt:
        print("\n[collector] interrupted by user")
    except (JakaError, RuntimeError, ValueError, OSError) as exc:
        print(f"[collector] FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        if dataset is not None:
            dataset.close()
        if app is not None:
            app.shutdown()
        if source is not None:
            source.close()


if __name__ == "__main__":
    main()
