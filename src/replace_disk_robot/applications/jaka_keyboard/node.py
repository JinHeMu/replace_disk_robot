"""Keyboard input application over the common JAKA servo loop."""
from __future__ import annotations
import sys
from pathlib import Path
from replace_disk_robot.adapters.jaka import JakaError
from replace_disk_robot.adapters.jaka.session import RateLoop, edg_session, make_client
from ..common.node import JakaServo, _RECOVERABLE_FAULTS
from .config import _parse_args, _validate_admittance_args
from .reference import ReferencePipeline


class JakaKeyboardServo(JakaServo):
    input_label = "keyboard"
    reference_type = ReferencePipeline

    @property
    def keys(self):
        return self.reference.keys

    def _clear_input(self):
        self.keys.clear()

    def input_source_paths(self):
        from replace_disk_robot.control import key_control
        return (Path(key_control.__file__),)

    def _jog_to_servo(self, reference=None):
        return self.reference._jog_to_servo(reference)

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

    def record_input_event(self, key, action):
        self._record_event("key", key=key, action=action)


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
