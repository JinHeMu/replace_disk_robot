#!/usr/bin/env python3
"""Collect static-pose data for six-axis F/T payload gravity identification.

This tool reuses the real JAKA keyboard Cartesian servo.  Move to several
distinct, collision-free tool orientations, release all motion keys, then
press C.  Each C press waits for the arm to settle and records one static
window.  The CSV contains synchronized EDG joint state, raw sensor wrench and
the sensor orientation computed from JAKA forward kinematics.

The script commands real hardware unless --dry-run is supplied.  It does not
power on or enable the robot; use the repository's jaka_start.py first.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import glfw
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIVER_TOOL_DIR = PROJECT_ROOT / "examples" / "jaka_driver_tool"
SRC_DIR = PROJECT_ROOT / "src"
for search_path in (DRIVER_TOOL_DIR, SRC_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from jaka_common import (  # noqa: E402
    DEFAULT_SENSOR_TO_TOOL_ROTATION,
    DEFAULT_TOOL_ARM_M,
    JakaError,
    RateLoop,
    add_network_args,
    edg_session,
    make_client,
)
from jaka_keyboard_servo import (  # noqa: E402
    JakaKeyboardServo,
    _KEY_TO_NAME,
    _window_title,
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


def _parse_args() -> argparse.Namespace:
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
    parser.add_argument("--rate-hz", type=float, default=125.0)
    parser.add_argument("--linear-speed", type=float, default=0.01)
    parser.add_argument("--angular-speed-deg", type=float, default=5.0)
    parser.add_argument("--command-frame", choices=("base", "tool"), default="base")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="optional total duration; 0 runs until Esc")
    parser.add_argument("--dry-run", action="store_true",
                        help="read and capture only; never enable Servo or send commands")
    parser.add_argument("--settle-seconds", type=float, default=0.75,
                        help="continuous low-speed time required before capture")
    parser.add_argument("--capture-seconds", type=float, default=1.0,
                        help="duration of each accepted static window")
    parser.add_argument("--stationary-speed-rad-s", type=float, default=0.01,
                        help="maximum absolute joint velocity during settling/capture")
    parser.add_argument("--max-force-n", type=float, default=10.0)
    parser.add_argument("--max-torque-nm", type=float, default=2.0)
    parser.add_argument("--max-joint-error-rad", type=float, default=0.15)
    parser.add_argument("--torque-sensor-mode", type=int, default=1,
                        help="JAKA sensor mode; it must expose uncompensated EDG wrench")
    parser.add_argument("--tare-samples", type=int, default=50,
                        help="tare used only by the live safety gate, not by saved raw data")
    parser.add_argument("--tare-period-ms", type=float, default=10.0)
    parser.add_argument("--no-tare", action="store_true",
                        help="disable safety-gate tare; saved raw data is never tared")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--deadband-force", type=float, default=1.0)
    parser.add_argument("--deadband-torque", type=float, default=0.2)
    parser.add_argument("--identity-transform", action="store_true",
                        help="use identity sensor-to-tool rotation (diagnostics only)")
    parser.set_defaults(headless=False, plot_wrench=False)
    return parser.parse_args()


class GravityDatasetWriter:
    """Capture-state machine and synchronized CSV writer."""

    def __init__(self, path: Path, args: argparse.Namespace) -> None:
        self.path = path
        self.args = args
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if args.overwrite else "x"
        self.file = self.path.open(mode, newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_COLUMNS)
        self.writer.writeheader()
        self.capture_id = 0
        self.phase = "idle"
        self.stable_since_s: float | None = None
        self.record_start_s = 0.0
        self.recorded_rows = 0
        self.capture_rows: list[dict[str, object]] = []
        self.completed_captures = 0
        self.app: JakaKeyboardServo | None = None

    @property
    def busy(self) -> bool:
        return self.phase != "idle"

    def close(self) -> None:
        if not self.file.closed:
            self.file.flush()
            self.file.close()

    def request_capture(self) -> None:
        if self.app is None:
            return
        if self.busy:
            print(f"[capture] already {self.phase}; wait for this window to finish")
            return
        if self.app.servo.fault is not None:
            print(f"[capture] denied while Servo is stopped: {self.app.servo.fault}")
            return
        self.app.keys.clear()
        self.phase = "settling"
        self.stable_since_s = None
        print(
            f"[capture] request #{self.capture_id}: release the robot; waiting for "
            f"{self.args.settle_seconds:.2f}s of stable feedback"
        )

    def _update_phase(self, elapsed_s: float, max_joint_speed: float) -> None:
        stationary = max_joint_speed <= self.args.stationary_speed_rad_s
        if self.phase == "settling":
            if stationary:
                if self.stable_since_s is None:
                    self.stable_since_s = elapsed_s
                if elapsed_s - self.stable_since_s >= self.args.settle_seconds:
                    self.phase = "recording"
                    self.record_start_s = elapsed_s
                    self.recorded_rows = 0
                    self.capture_rows.clear()
                    print(f"[capture] recording #{self.capture_id}...")
            else:
                self.stable_since_s = None
        elif self.phase == "recording" and not stationary:
            print(
                f"[capture] #{self.capture_id} rejected: arm moved "
                f"({max_joint_speed:.4f} rad/s); press C to retry"
            )
            self.phase = "idle"
            self.recorded_rows = 0
            self.capture_rows.clear()

    def __call__(self, elapsed_s, state, measured, wrench, app) -> None:
        self.app = app
        max_joint_speed = float(np.max(np.abs(state.joint_velocity_rad_s)))
        self._update_phase(elapsed_s, max_joint_speed)

        pose = app.kinematics.forward(measured)
        r_base_tool = rotation_matrix(pose.quaternion_wxyz)
        r_sensor_tool = (
            np.eye(3)
            if self.args.identity_transform
            else np.asarray(DEFAULT_SENSOR_TO_TOOL_ROTATION, dtype=float)
        )
        r_base_sensor = r_base_tool @ r_sensor_tool
        tool_to_sensor = (
            np.zeros(3)
            if self.args.identity_transform
            else np.asarray(DEFAULT_TOOL_ARM_M, dtype=float)
        )
        active_id = self.capture_id if self.phase == "recording" else -1
        row = {
            "time_s": f"{elapsed_s:.9f}",
            "utc_iso": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "capture_id": active_id,
            "capture_phase": self.phase,
            "command_active": int(bool(app.keys.pressed)),
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
        if self.phase == "recording":
            # Keep the tentative window in memory.  If feedback starts moving,
            # the whole attempt is discarded instead of leaving partial rows
            # with a seemingly valid capture_id in the CSV.
            self.capture_rows.append(row)
            self.recorded_rows += 1
            if elapsed_s - self.record_start_s >= self.args.capture_seconds:
                completed_id = self.capture_id
                self.writer.writerows(self.capture_rows)
                self.completed_captures += 1
                self.capture_id += 1
                self.phase = "idle"
                self.capture_rows.clear()
                self.file.flush()
                print(
                    f"[capture] saved #{completed_id}: {self.recorded_rows} rows; "
                    f"total={self.completed_captures}. Move to a new orientation and press C."
                )
        else:
            self.writer.writerow(row)


def _on_key(window, key, scancode, action, mods) -> None:  # noqa: ARG001
    app, dataset = glfw.get_window_user_pointer(window)
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
        return
    if key == glfw.KEY_C and action == glfw.PRESS:
        dataset.request_capture()
        return
    if key == glfw.KEY_SPACE and action == glfw.PRESS:
        app.stop("stopped")
        return
    if key == glfw.KEY_ENTER and action == glfw.PRESS:
        app.resume()
        return
    name = _KEY_TO_NAME.get(key)
    if name is None or dataset.busy:
        return
    if action in (glfw.PRESS, glfw.REPEAT):
        app.keys.press(name)
    elif action == glfw.RELEASE:
        app.keys.release(name)


def _on_focus(window, focused: bool) -> None:
    app, _dataset = glfw.get_window_user_pointer(window)
    app.handle_focus(bool(focused))


def _run_window(app: JakaKeyboardServo, dataset: GravityDatasetWriter) -> None:
    if not glfw.init():
        raise RuntimeError("cannot initialize GLFW display")
    window = None
    try:
        window = glfw.create_window(720, 240, "JAKA F/T gravity data", None, None)
        if window is None:
            raise RuntimeError("cannot create GLFW window")
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        glfw.set_window_user_pointer(window, (app, dataset))
        glfw.set_key_callback(window, _on_key)
        glfw.set_window_focus_callback(window, _on_focus)
        glfw.show_window(window)
        glfw.focus_window(window)
        print(
            "[collector] Move with W/S, A/D, R/F, Q/E and arrow keys.\n"
            "[collector] Release movement keys and press C at each static orientation. "
            "Space stops, Enter resumes, Esc exits. Aim for >=12 diverse orientations."
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
            glfw.set_window_title(
                window,
                f"{_window_title(app)} | capture={dataset.phase} "
                f"saved={dataset.completed_captures}",
            )
            glfw.swap_buffers(window)
            if app.args.seconds > 0 and elapsed_s >= app.args.seconds:
                break
        print(
            f"[collector] finished: {dataset.completed_captures} captures, "
            f"CSV={dataset.path}"
        )
    finally:
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()


def main() -> None:
    args = _parse_args()
    if args.rate_hz <= 0 or args.seconds < 0:
        raise SystemExit("--rate-hz must be positive and --seconds non-negative")
    if args.settle_seconds <= 0 or args.capture_seconds <= 0:
        raise SystemExit("--settle-seconds and --capture-seconds must be positive")
    if args.stationary_speed_rad_s <= 0:
        raise SystemExit("--stationary-speed-rad-s must be positive")

    client = make_client(args)
    try:
        dataset = GravityDatasetWriter(args.output.resolve(), args)
    except FileExistsError as exc:
        raise SystemExit(
            f"output CSV already exists: {args.output.resolve()} "
            "(choose another --output or pass --overwrite)"
        ) from exc
    app = None
    try:
        with edg_session(client, torque_sensor_mode=args.torque_sensor_mode) as client:
            app = JakaKeyboardServo(client, args, sample_callback=dataset)
            dataset.app = app
            try:
                app.initialize()
                _run_window(app, dataset)
            finally:
                app.shutdown()
    except KeyboardInterrupt:
        print("\n[collector] interrupted by user")
    except (JakaError, RuntimeError, ValueError) as exc:
        print(f"[collector] FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        dataset.close()
        if app is not None:
            app.shutdown()


if __name__ == "__main__":
    main()
