"""Xbox frontend for the existing JAKA feedback/admittance/servo/force chain."""
from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time
import numpy as np

from replace_disk_robot.adapters.jaka.session import RateLoop, edg_session, make_client
from replace_disk_robot.core import CartesianJog, JointState, Pose
from replace_disk_robot.core.rotation import rotation_matrix
from replace_disk_robot.control import JoyControl, JoySample
from .config import parse_args
from ..common.node import JakaServo


class JakaJoyServo(JakaServo):
    input_label = "joy"
    def __init__(self, client, args, **kwargs):
        super().__init__(client, args, **kwargs)
        self.joy = JoyControl(args.linear_speed, np.deg2rad(args.angular_speed_deg),
                              base_frame=self.command_frame,
                              deadzone=args.joy_deadzone,
                              precision_scale=args.joy_precision_scale,
                              timeout_s=args.joy_timeout_s)
        self._joy_now = 0.
        self._buttons = frozenset()
        self._released = False
        self._active = False
        self.resume_required = False
        self.exit_requested = False
        self._linear_jog_active = False
        self._angular_jog_active = False
        self._joy_measured = None

    @property
    def joy_sample(self):
        return self.joy.sample

    def input_snapshot(self):
        return {"joy": {"sample": self.joy_sample, "active": self._active,
                        "resume_required": self.resume_required}}

    def input_source_paths(self):
        from replace_disk_robot.control import joy_control
        from replace_disk_robot.adapters import joy
        return (Path(joy_control.__file__), Path(joy.__file__))

    def initialize(self):
        super().initialize()
        self.stop("stopped")
        print("[joy] Release LB and center sticks/triggers before arming. "
              "Hold LB to move; RB precision; B stop; A resume; Back exit.")

    def stop(self, reason, measured=None):
        self._active = False
        self._linear_jog_active = self._angular_jog_active = False
        super().stop(reason, measured)

    def _read_wrench(self, state, measured):
        self._joy_measured = measured
        return super()._read_wrench(state, measured)

    def _update_compliance(self, jog, wrench, tcp_pose, dt_s):
        linear = bool(np.any(jog.linear_m_s))
        angular = bool(np.any(jog.angular_rad_s))
        released_linear = self._linear_jog_active and not linear
        released_angular = self._angular_jog_active and not angular
        if released_linear or released_angular:
            self.motion.reanchor(tcp_pose, translation=released_linear, rotation=released_angular)
            # Discard the outstanding joint-position lead too, using the exact
            # feedback packet already read by this tick; no second EDG read.
            self.servo.reset(self._joy_measured)
            self._record_event("joy_centered", translation=released_linear, rotation=released_angular)
        self._linear_jog_active, self._angular_jog_active = linear, angular
        return super()._update_compliance(jog, wrench, tcp_pose, dt_s)

    def resume(self):
        # Stops reset the compliance filter. Do not let a newly seeded filter
        # attenuate a still-present load and authorize recovery prematurely.
        previous_capture = self.wrench.capture_stages
        try:
            self.wrench.capture_stages = True
            state = self.client.read_edg_state()
            measured = JointState(self.arm.joint_names, state.joint_position_rad)
            self._read_wrench(state, measured)
            raw = np.asarray(self.wrench.stages["external_tcp_unfiltered"], dtype=float)
            if (raw.shape != (6,) or not np.isfinite(raw).all()
                    or np.linalg.norm(raw[:3]) > .8 * self.args.max_force_n
                    or np.linalg.norm(raw[3:]) > .8 * self.args.max_torque_nm):
                self._record_event("joy_resume_denied", reason="load_present")
                return False
        except (ValueError, RuntimeError, KeyError, OSError):
            return False
        finally:
            self.wrench.capture_stages = previous_capture
        return super().resume()

    def apply_sample(self, sample, now):
        self.joy.update(sample)
        self._joy_now = now
        valid = self.joy.valid(now)
        buttons = sample.buttons if valid else frozenset()
        edges = buttons - self._buttons
        self._buttons = buttons
        self._record_event("joy", axes=sample.axes, buttons=sorted(buttons),
                           connected=sample.connected, valid=valid)
        if not valid:
            if self._active or not self.resume_required:
                self.stop("stopped")
                self._record_event("joy_unavailable")
            self.resume_required, self._released = True, False
            return
        if "back" in buttons:
            self.exit_requested = True
        if "b" in buttons or self.exit_requested:
            self.stop("stopped")
            self.resume_required, self._released = True, False
            return
        if "lb" not in buttons:
            if self._active:
                self.stop("stopped")
            self._released = self.joy.neutral()
            if "a" in edges and self._released:
                # Reuse force-aware recovery; never hide the load by re-taring.
                if self.resume():
                    self.resume_required = False
                    self.stop("stopped")
            return
        if "lb" in edges and self._released and not self.resume_required:
            if self.servo.fault in (None, "stopped") and self.resume():
                self._active = True
            self._released = False

    def tick(self, elapsed_s, dt_s):
        # Caller uses the same monotonic domain for poll and tick (simulation may
        # supply simulated time). Missing samples cannot keep the last jog alive.
        self._joy_now = elapsed_s
        if not self.joy.valid(elapsed_s):
            self.apply_sample(self.joy_sample, elapsed_s)
        super().tick(elapsed_s, dt_s)
        if self.servo.fault not in (None, "stopped"):
            self._active = False
            self.resume_required = True

    def _jog_to_servo(self, reference=None):
        if not self._active or self.servo.fault is not None:
            return CartesianJog(self.base_frame, np.zeros(3), np.zeros(3))
        if reference is None:
            reference = self.kinematics.forward(self.servo.target or self.arm.read_joint_state())
        elif not isinstance(reference, Pose):
            reference = self.kinematics.forward(reference)
        base_from_tool = rotation_matrix(reference.quaternion_wxyz)
        if self.args.command_frame == "base":
            # JAKA mounting convention: left/right in base X, up/down in base
            # Z; triggers always insert/retract along current tool0 Z.
            axes = np.column_stack(([1., 0., 0.], [0., 0., 1.], base_from_tool[:, 2]))
            command = self.joy.command(self._joy_now, translation_axes=axes)
            return CartesianJog(self.base_frame, command.linear_m_s,
                                base_from_tool.T @ command.angular_rad_s)
        command = self.joy.command(self._joy_now)
        return CartesianJog(self.base_frame, base_from_tool @ command.linear_m_s,
                            command.angular_rad_s)

    def _status_text(self):
        if self.resume_required:
            return f"{self.servo.fault or 'holding'}: release LB, center inputs, press A"
        return "LB active" if self._active else "holding: center inputs, release then hold LB"




def main():
    args = parse_args()
    if args.validate:
        from .joy_validation import validate
        report = validate(args)
        result = json.dumps(report, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result + "\n")
        print(result)
        if not report["passed"]:
            raise SystemExit(1)
        return
    from replace_disk_robot.adapters.joy import XboxController
    source, app = None, None
    try:
        source = XboxController(args.controller_index)
        print(f"[joy] {source.name}: {source.mapping}")
        if args.probe:
            for elapsed, _ in RateLoop(10):
                sample = source.poll()
                print(sample, flush=True)
                if "back" in sample.buttons or (args.seconds > 0 and elapsed >= args.seconds):
                    break
            return
        if args.backend == "sim":
            from .joy_simulation import make_sim_app
            app = make_sim_app(args)
            client = app.client
            session = nullcontext(client)
        else:
            client = make_client(args)
            session = edg_session(client, torque_sensor_mode=args.torque_sensor_mode)
        with session:
            if app is None:
                app = JakaJoyServo(client, args)
            try:
                app.initialize()
                from .joy_frontend import run
                run(app, source)
            finally:
                app.shutdown()
    except KeyboardInterrupt:
        print("\n[joy] interrupted")
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"[joy] FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        if app is not None:
            app.shutdown()
        if source is not None:
            source.close()
