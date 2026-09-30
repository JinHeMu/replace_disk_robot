"""Single-loop JAKA keyboard node: assembly, lifecycle and command dispatch."""
from __future__ import annotations
import argparse
import sys
import time
from copy import deepcopy
from collections.abc import Callable
import numpy as np
from replace_disk_robot.adapters.jaka import JakaError, JakaRobotAdapter, fmt_array
from replace_disk_robot.adapters.jaka.session import RateLoop, edg_session, make_client
from replace_disk_robot.control import CartesianServo, ServoConfig
from replace_disk_robot.core import JointState, Pose, Wrench
from replace_disk_robot.kinematics.jaka import JakaKinematics
from replace_disk_robot.safety import ForceLimitGuard
from .config import PROJECT_ROOT, _parse_args, _validate_admittance_args
from .wrench import WrenchPipeline
from .reference import ReferencePipeline
from .log import AsyncRecorder, CycleCapture, capture_stage, session_metadata


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

SampleCallback = Callable[[float, object, JointState, object, "JakaKeyboardServo"], None]

class JakaKeyboardServo:
    """Own the hardware lifecycle and preserve one ordered command path."""
    def __init__(self, client, args, *, sample_callback=None, kinematics=None,
                 recorder=None, tick_callback=None):
        self.client, self.args = client, args
        self.kinematics = JakaKinematics() if kinematics is None else kinematics
        self.arm = JakaRobotAdapter(client, joint_names=self.kinematics.joint_names)
        self.base_frame = self.kinematics.base_frame
        self.tool_frame = self.kinematics.end_effector_frame
        self.command_frame = self.tool_frame if args.command_frame == "tool" else self.base_frame
        self.servo = CartesianServo(
            self.kinematics,
            self.kinematics.joint_limits_rad,
            ServoConfig(
                linear_speed_m_s=args.linear_speed,
                angular_speed_rad_s=np.deg2rad(args.angular_speed_deg),
                max_tracking_error_rad=args.max_joint_error_rad,
            ),
        )
        self.guard = ForceLimitGuard(
            force_limit_n=args.max_force_n,
            torque_limit_nm=args.max_torque_nm,
        )
        self.sample_callback = sample_callback

        self.wrench = WrenchPipeline(client, args, self.kinematics)
        self.reference = ReferencePipeline(args, self.kinematics, self.arm, self.servo)
        self.admittance_enabled = bool(args.admittance)
        self.recorder, self.tick_callback = recorder, tick_callback

        self.last_wrench = None
        self.servo_enabled = False
        self.status = "initializing"
        self._next_print = 0.0
        self._last_now = None
        self._closed = False
        self._tick_sequence = 0
        self._active_capture = None
        self._observer_error_reported = False

    @property
    def ft(self):
        return self.wrench.ft

    @property
    def sensor_frame_id(self):
        return self.wrench.sensor_frame_id

    @property
    def sensor_to_tool_rotation(self):
        return self.wrench.sensor_to_tool_rotation

    @property
    def tool_to_sensor_m(self):
        return self.wrench.tool_to_sensor_m

    @property
    def gravity_compensator(self):
        return self.wrench.gravity_compensator

    @property
    def wrench_processor(self):
        return self.wrench.wrench_processor

    @property
    def payload_mass_kg(self):
        return self.wrench.payload_mass_kg

    @property
    def payload_com_sensor_m(self):
        return self.wrench.payload_com_sensor_m

    @property
    def keys(self):
        return self.reference.keys

    @property
    def motion(self):
        return self.reference.motion

    @property
    def admittance(self):
        return self.reference.admittance

    @property
    def motion_config(self):
        return self.reference.motion_config

    @property
    def last_external_wrench(self):
        return self.reference.last_external_wrench

    @last_external_wrench.setter
    def last_external_wrench(self, value):
        self.reference.last_external_wrench = value

    @property
    def last_corrected_pose(self):
        return self.reference.last_corrected_pose

    @last_corrected_pose.setter
    def last_corrected_pose(self, value):
        self.reference.last_corrected_pose = value

    def _configure_gravity_compensation(self):
        return self.wrench._configure_gravity_compensation()

    def _sensor_pose_in_base(self, measured):
        return self.wrench._sensor_pose_in_base(measured)

    def _tare_compensated_wrench(self, state, measured):
        return self.wrench._tare_compensated_wrench(state, measured)

    def _read_wrench(self, state, measured):
        return self.wrench._read_wrench(state, measured)

    def _gravity_compensated_wrench(self, state, measured):
        return self.wrench._gravity_compensated_wrench(state, measured)

    def _reset_compliance(self, measured):
        reset = self.reference._reset_compliance(measured)
        if reset and self.wrench_processor is not None:
            self.wrench_processor.reset()
        if reset:
            self._record_event("reference_reset")

    def _update_compliance(self, jog, wrench, tcp_pose, dt_s):
        return self.reference._update_compliance(jog, wrench, tcp_pose, dt_s)

    def _clamp_offset(self, corrected, nominal):
        return self.reference._clamp_offset(corrected, nominal)

    def _jog_to_servo(self, reference=None):
        return self.reference._jog_to_servo(reference)

    def admittance_text(self):
        return self.reference.admittance_text()

    def initialize(self) -> None:
        state0 = self.client.read_edg_state()
        q0 = JointState(self.arm.joint_names, state0.joint_position_rad)
        print(
            f"[keyboard] initial q={fmt_array(q0.position_rad, 4)} "
            f"base={self.base_frame} command_frame={self.command_frame}"
        )

        gravity_json = getattr(self.args, "gravity_json", None)
        use_gravity = bool(gravity_json) and getattr(
            self.args, "gravity_compensation_enable", True
        )
        if use_gravity:
            self._configure_gravity_compensation()
            if getattr(self.args, "tare_compensated", False):
                self._tare_compensated_wrench(state0, q0)
        else:
            if getattr(self.args, "tare_compensated", False):
                raise ValueError(
                    "--tare-compensated requires gravity compensation; "
                    "set --gravity-compensation-enable true"
                )
            if getattr(self.args, "tare_enable", True):
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

        if self.recorder is None and getattr(self.args, "log_dir", None) is not None:
            self.recorder = AsyncRecorder(
                self.args.log_dir, session_metadata(self.args, self, PROJECT_ROOT),
                queue_size=getattr(self.args, "log_queue_size", 4096),
            )
        self._record_event("initialized", dry_run=self.args.dry_run)
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
        self._record_event("shutdown")
        if self.recorder is not None:
            try:
                self.recorder.close()
            except Exception as exc:
                self._observer_error(exc)

    def stop(self, reason: str, measured: JointState | None = None) -> None:
        self._record_event("stop", reason=reason)
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
                    self._send_joint_positions(measured)
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

        self._record_event("resume_requested", fault=self.servo.fault)
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
        self._record_event("resumed", previous_fault=fault)
        return True

    def handle_focus(self, focused: bool) -> None:
        self._record_event("focus", focused=focused)
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

    def _observer_error(self, exc):
        try:
            if self.recorder is not None:
                self.recorder.note_observer_error()
        except Exception:
            pass
        if not self._observer_error_reported:
            self._observer_error_reported = True
            print(f"[log] observer failed; control continues: {exc}", file=sys.stderr)

    def _record_event(self, event, **fields):
        if self.recorder is None:
            return
        try:
            self.recorder.publish("events", {
                "event": event, "sequence": self._tick_sequence,
                "monotonic_ns": time.perf_counter_ns(), **fields,
            })
        except Exception as exc:
            self._observer_error(exc)

    def record_input_event(self, key, action):
        self._record_event("key", key=key, action=action)

    def _send_joint_positions(self, command):
        attempt = None
        if self._active_capture is not None:
            attempt = {"q_rad": command.position_rad.copy(),
                       "start_monotonic_ns": time.perf_counter_ns(), "accepted": False}
            self._active_capture.data.setdefault("send_attempts", []).append(attempt)
        try:
            self.arm.command_joint_positions(command)
        except Exception as exc:
            if attempt is not None:
                attempt["error"] = f"{type(exc).__name__}: {exc}"
            self._record_event("command_rejected", error=str(exc))
            raise
        else:
            if attempt is not None:
                attempt["accepted"] = True
                self._active_capture.data["command_sent"] = True
        finally:
            if attempt is not None:
                attempt["end_monotonic_ns"] = time.perf_counter_ns()
        self._record_event("command_accepted", q_rad=command.position_rad.copy())

    def tick(self, elapsed_s: float, dt_s: float) -> None:
        capture = None
        self._tick_sequence += 1
        self.wrench.capture_stages = self.recorder is not None or self.tick_callback is not None
        if self.wrench.capture_stages:
            try:
                capture = CycleCapture(self, self._tick_sequence, elapsed_s, dt_s)
            except Exception as exc:
                self._observer_error(exc)
        self._active_capture = capture
        error = None
        try:
            self._tick(elapsed_s, dt_s, capture)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._active_capture = None
            if capture is not None:
                try:
                    record = capture.finish(self, error)
                    if self.recorder is not None:
                        self.recorder.publish("samples", record)
                        if getattr(self.recorder, "error", None):
                            self._observer_error(self.recorder.error)
                    if self.tick_callback is not None:
                        self.tick_callback(deepcopy(record))
                except Exception as exc:
                    self._observer_error(exc)

    def _tick(self, elapsed_s: float, dt_s: float, capture=None) -> None:
        if capture is not None:
            capture.data["feedback_read_start_monotonic_ns"] = time.perf_counter_ns()
        state = self.client.read_edg_state()
        measured = JointState(self.arm.joint_names, state.joint_position_rad)
        capture_stage(capture, "feedback", self, self, state, measured)
        try:
            wrench = self._read_wrench(state, measured)
        except (ValueError, RuntimeError) as exc:
            self.stop("invalid_wrench", measured)
            print(f"[keyboard] invalid wrench: {exc}", file=sys.stderr)
            return
        self.last_wrench = wrench
        if capture is not None:
            capture.data["wrench_stages"] = deepcopy(self.wrench.stages)

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
        capture_stage(capture, "jog", self, jog)
        if self.args.dry_run:
            # Still integrate the compliance chain so the printed offset and
            # the sign of the external force can be verified before enabling
            # Servo; no command is submitted in dry-run.
            if self.admittance_enabled and self.servo.fault is None:
                self._update_compliance(
                    jog, wrench, self.kinematics.forward(measured), dt_s
                )
                capture_stage(capture, "reference", self, self, wrench)
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
                capture_stage(capture, "reference", self, self, wrench)
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
        if capture is not None:
            capture.data["servo_target_q_rad"] = target.position_rad.copy()
            capture.data["force_gate_tripped"] = bool(tripped)
        if tripped:
            self.stop("force_limit", measured)
            safe_q = measured.position_rad
        else:
            self.status = self.servo.status

        if capture is not None:
            capture.data["final_command_q_rad"] = np.asarray(safe_q).copy()
        self._send_joint_positions(
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
                    from .frontend import _run_window
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
