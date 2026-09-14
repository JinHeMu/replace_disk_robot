#!/usr/bin/env python3
"""Prepare a JAKA robot for standalone Python tests.

Sequence (mirroring the WBMM reference driver):

1. ``login()``
2. ``power_on()`` and wait
3. ``enable_robot()`` and wait
4. joint servo LPF and torque-sensor mode setup

This script does **not** enable servo mode or send motion commands.  EDG is
also not enabled here because it needs a process that keeps reading the UDP
stream; use ``jaka_ft_test.py`` or ``jaka_edg_servo.py`` for that.
"""

from __future__ import annotations

import argparse
import sys
import time

from jaka_common import JakaError, add_network_args, fmt_array, make_client


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_network_args(parser)
    parser.add_argument("--read-only", action="store_true",
                        help="only login/read state; do not power on, enable, or change gains")
    parser.add_argument("--power-wait", type=float, default=8.0,
                        help="seconds to wait after power_on")
    parser.add_argument("--enable-wait", type=float, default=4.0,
                        help="seconds to wait after enable_robot")
    parser.add_argument("--joint-lpf-hz", type=float, default=2.0,
                        help="joint servo LPF cutoff; negative value disables this call")
    parser.add_argument("--torque-sensor-mode", type=int, default=1,
                        help="torque-sensor mode passed to the SDK; negative value disables")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.power_wait < 0 or args.enable_wait < 0:
        raise SystemExit("--power-wait and --enable-wait must be non-negative")

    client = make_client(args)
    try:
        print(f"[start] logging in to {args.robot_ip}")
        client.login()
        print("[start] login OK")

        if args.read_only:
            print("[start] read-only mode: no power/enable/gain changes")
        else:
            print("[start] power_on")
            client.power_on()
            time.sleep(args.power_wait)
            print(f"[start] power_on OK, waited {args.power_wait:g}s")

            print("[start] enable_robot")
            client.enable_robot()
            time.sleep(args.enable_wait)
            print(f"[start] enable_robot OK, waited {args.enable_wait:g}s")

            if args.joint_lpf_hz >= 0:
                print(f"[start] servo_move_use_joint_LPF({args.joint_lpf_hz:g})")
                client.set_joint_lpf(args.joint_lpf_hz)
            else:
                print("[start] joint LPF setup skipped")

            if args.torque_sensor_mode >= 0:
                print(f"[start] set_torque_sensor_mode({args.torque_sensor_mode})")
                client.set_torque_sensor_mode(args.torque_sensor_mode)
            else:
                print("[start] torque-sensor mode setup skipped")

        q = client.get_joint_position()
        print(f"[start] joint position rad: {fmt_array(q)}")
        try:
            tcp = client.get_tcp_position()
            print(f"[start] TCP mm/rad:       {fmt_array(tcp)}")
        except JakaError as exc:
            print(f"[start] TCP read skipped: {exc}", file=sys.stderr)

        print("[start] JAKA startup sequence completed")
    except JakaError as exc:
        print(f"[start] FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        try:
            client.logout()
        except JakaError as exc:
            print(f"[start] logout warning: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
