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
``JakaRobotAdapter.command_joint_positions()``.  Use ``--dry-run`` first: it
reads EDG/F/T and keyboard state but never enables servo or sends commands.
"""

from __future__ import annotations

import argparse
import sys
import time

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
from replace_disk_robot.control import (
    CartesianJog,
    CartesianServo,
    KeyControl,
    ServoConfig,
)
from replace_disk_robot.core import JointState
from replace_disk_robot.core.rotation import rotation_matrix
from replace_disk_robot.kinematics.jaka import JakaKinematics
from replace_disk_robot.safety import ForceLimitGuard


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--plot-wrench", action="store_true",
                        help="show non-blocking live force/torque curves")
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
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="F/T low-pass filter alpha")
    parser.add_argument("--deadband-force", type=float, default=1.0)
    parser.add_argument("--deadband-torque", type=float, default=0.2)
    parser.add_argument("--identity-transform", action="store_true",
                        help="skip F/T frame/arm compensation; useful for raw diagnostics")
    return parser.parse_args()


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


class JakaKeyboardServo:
    """State machine for real JAKA keyboard Cartesian servo."""

    def __init__(self, client, args: argparse.Namespace) -> None:
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
        q0 = self.arm.read_joint_state()
        print(
            f"[keyboard] initial q={fmt_array(q0.position_rad, 4)} "
            f"base={self.base_frame} command_frame={self.command_frame}"
        )

        if not self.args.no_tare:
            print(f"[keyboard] taring FT with {self.args.tare_samples} samples")
            bias = self.ft.tare(
                samples=self.args.tare_samples,
                period_s=self.args.tare_period_ms / 1000.0,
            )
            print(f"[keyboard] FT bias={fmt_array(bias, 4)}")

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
        q_servo = self.arm.read_joint_state()
        self.servo.reset(q_servo)
        self.status = "holding"

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
    def _jog_to_servo(self) -> CartesianJog:
        """Map keyboard keys to Servo's base-linear/intrinsic-TCP convention.

        * tool mode: W/S/A/D and rotations use the current tool0 frame.  R/F
          still follow the current tool0 +Z/-Z insertion axis.
        * base mode: W/S use base +/-Z; A/D use base -X/+X, which corresponds
          to platform left/right for the -90-degree arm mounting; R/F move
          along current tool0 +Z/-Z.  Base-axis rotations are converted to
          intrinsic TCP angular velocity.
        """
        target = self.servo.target
        if target is None:
            target = self.arm.read_joint_state()
        pose = self.kinematics.forward(target)
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
            wrench = self.ft.read_wrench_from(state)
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
        wrench = self.ft.read_wrench_from(state)
        self.last_wrench = wrench

        if not np.isfinite(dt_s) or dt_s <= 0:
            self.stop("invalid_dt", measured)
            return
        if dt_s > self.servo.config.max_dt_s:
            self.stop("loop_timeout", measured)
            return

        jog = self._jog_to_servo()
        if self.args.dry_run:
            self.status = "dry_run"
            self._print_line(elapsed_s, measured, measured.position_rad, wrench)
            return

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
        print(
            f"[keyboard] t={elapsed_s:6.3f}s "
            f"q={fmt_array(measured.position_rad, 4)} "
            f"cmd={fmt_array(command_q, 4)} "
            f"|F|={force:6.3f}N |T|={torque:6.3f}Nm "
            f"{self._status_text()}"
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
    return (
        f"JAKA keyboard servo | {app.command_frame} | "
        f"F={force:.2f} N | T={torque:.2f} Nm | {app._status_text()}"
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
            from replace_disk_robot.visual import ProcessTypePlotter
            plotter = ProcessTypePlotter(window_s=10.0, refresh_hz=10.0)

        print(
            "[keyboard] a small window named 'JAKA keyboard servo' has opened.\n"
            "[keyboard] Click that window, then hold keys there. "
            "Space stop, Enter resume, Esc exit."
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
