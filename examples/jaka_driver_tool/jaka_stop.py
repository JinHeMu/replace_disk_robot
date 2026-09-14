#!/usr/bin/env python3
"""Stop and disconnect a JAKA robot from a standalone Python process.

Default sequence:

1. ``servo_move_enable(False)``
2. disable EDG
3. ``disable_robot()``
4. ``power_off()``
5. ``logout()``

Use ``--read-only`` to only disable EDG and logout, without changing
power/enable state.  Use ``--shutdown`` only if you really want to call the
controller ``shut_down()`` API as well.
"""

from __future__ import annotations

import argparse
import sys

from jaka_common import JakaError, add_network_args, make_client


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_network_args(parser)
    parser.add_argument("--read-only", action="store_true",
                        help="only disable EDG and logout; skip servo/disable/power-off")
    parser.add_argument("--shutdown", action="store_true",
                        help="also call the controller shut_down() API after power_off")
    return parser.parse_args()


def _best_effort(label: str, action) -> None:
    try:
        action()
        print(f"[stop] {label} OK")
    except JakaError as exc:
        print(f"[stop] {label} warning: {exc}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - best-effort shutdown
        print(f"[stop] {label} warning: {exc}", file=sys.stderr)


def main() -> None:
    args = _parse_args()
    client = make_client(args)
    try:
        print(f"[stop] logging in to {args.robot_ip}")
        client.login()
        _best_effort("servo_move_enable(False)", lambda: client.servo_move_enable(False))
        _best_effort("EDG disable", lambda: client.edg_disable(ignore_errors=True))

        if args.read_only:
            print("[stop] read-only mode: leaving power/enable state unchanged")
        else:
            _best_effort("disable_robot()", client.disable_robot)
            _best_effort("power_off()", client.power_off)
            if args.shutdown:
                _best_effort("shut_down()", client.shut_down)

        print("[stop] JAKA stop sequence completed")
    except JakaError as exc:
        print(f"[stop] FAILED during login/stop: {exc}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        _best_effort("logout()", client.logout)


if __name__ == "__main__":
    main()
