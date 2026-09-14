#!/usr/bin/env python3
"""Hold/sine test for JAKA EDG servo at 125 Hz without ROS.

Default behavior is a position hold: read the current joint position, enable
servo mode, then send that same absolute joint target every 8 ms.

Motion is only produced if ``--sine-joint`` is set explicitly, for example::

    python examples/jaka_driver_tool/jaka_edg_servo.py \
        --sine-joint 0 --sine-amplitude-deg 2 --seconds 5

Safety checks abort the run if the measured force/torque or joint tracking
error exceeds the configured bounds.  Use ``--dry-run`` to exercise the EDG
read/parse/adapters/timing loop without enabling servo mode or sending commands.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from jaka_common import (
    JakaError,
    JakaRobotAdapter,
    JakaWristFTAdapter,
    RateLoop,
    add_network_args,
    edg_session,
    fmt_array,
    make_client,
)
from replace_disk_robot.core import JointState


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_network_args(parser)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--rate-hz", type=float, default=125.0,
                        help="EDG servo rate; 125 Hz sends step_num=1")
    parser.add_argument("--sine-joint", type=int, default=-1,
                        help="joint index 0..5 for a sine test; negative means hold position")
    parser.add_argument("--sine-amplitude-deg", type=float, default=2.0)
    parser.add_argument("--sine-frequency-hz", type=float, default=0.2)
    parser.add_argument("--max-force-n", type=float, default=20.0)
    parser.add_argument("--max-torque-nm", type=float, default=5.0)
    parser.add_argument("--max-joint-error-rad", type=float, default=0.5)
    parser.add_argument("--no-tare", action="store_true",
                        help="skip the pre-servo FT tare")
    parser.add_argument("--dry-run", action="store_true",
                        help="read EDG but do not enable servo or send commands")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.seconds <= 0:
        raise SystemExit("--seconds must be positive")
    if args.rate_hz <= 0:
        raise SystemExit("--rate-hz must be positive")
    if args.sine_joint >= 6:
        raise SystemExit("--sine-joint must be -1 or 0..5")
    if args.sine_joint >= 0 and args.sine_amplitude_deg <= 0:
        raise SystemExit("--sine-amplitude-deg must be positive for sine mode")

    client = make_client(args)
    try:
        with edg_session(client) as client:
            arm = JakaRobotAdapter(client)
            ft = JakaWristFTAdapter(client)

            if not args.no_tare:
                print("[servo] taring FT before enabling servo")
                print(f"[servo] bias={fmt_array(ft.tare(), 4)}")

            q0 = client.read_edg_state().joint_position_rad.copy()
            print(f"[servo] initial q={fmt_array(q0, 4)}")

            servo_enabled = False
            try:
                if args.dry_run:
                    print("[servo] dry-run: not enabling servo mode")
                else:
                    client.servo_move_enable(True)
                    servo_enabled = True
                    print("[servo] servo_move_enable(True)")

                sine_joint = int(args.sine_joint)
                sine_amplitude_rad = math.radians(float(args.sine_amplitude_deg))
                sine_omega = 2.0 * math.pi * float(args.sine_frequency_hz)
                print_period = 0.5
                next_print = 0.0
                samples = 0
                loop = RateLoop(args.rate_hz)

                for t, _now in loop:
                    state = client.read_edg_state()
                    wrench = ft.read_wrench_from(state)
                    force_norm = float(np.linalg.norm(wrench.force_n))
                    torque_norm = float(np.linalg.norm(wrench.torque_nm))

                    if force_norm > args.max_force_n:
                        raise RuntimeError(
                            f"force limit: |F|={force_norm:.3f} N > "
                            f"{args.max_force_n:.3f} N"
                        )
                    if torque_norm > args.max_torque_nm:
                        raise RuntimeError(
                            f"torque limit: |T|={torque_norm:.3f} Nm > "
                            f"{args.max_torque_nm:.3f} Nm"
                        )

                    command = q0.copy()
                    if sine_joint >= 0:
                        command[sine_joint] += sine_amplitude_rad * math.sin(
                            sine_omega * t
                        )

                    error = float(np.max(np.abs(state.joint_position_rad - command)))
                    if error > args.max_joint_error_rad:
                        raise RuntimeError(
                            f"joint tracking error {error:.4f} rad > "
                            f"{args.max_joint_error_rad:.4f} rad"
                        )

                    if not args.dry_run:
                        arm.command_joint_positions(
                            JointState(arm.joint_names, command)
                        )

                    samples += 1
                    if t >= next_print:
                        print(
                            f"[servo] t={t:6.3f}s "
                            f"q={fmt_array(state.joint_position_rad, 4)} "
                            f"cmd={fmt_array(command, 4)} "
                            f"|F|={force_norm:6.3f}N "
                            f"|T|={torque_norm:6.3f}Nm"
                        )
                        next_print += print_period

                    if t >= args.seconds:
                        break

                print(
                    f"[servo] done: {samples} samples, "
                    f"overruns={loop.overruns}"
                )
            finally:
                if servo_enabled:
                    try:
                        client.servo_move_enable(False)
                        print("[servo] servo_move_enable(False)")
                    except Exception as exc:  # noqa: BLE001 - best-effort stop
                        print(f"[servo] servo-disable warning: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n[servo] interrupted by user")
    except (JakaError, RuntimeError, ValueError) as exc:
        print(f"[servo] FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
