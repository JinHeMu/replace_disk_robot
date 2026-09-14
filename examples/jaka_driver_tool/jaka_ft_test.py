#!/usr/bin/env python3
"""Read and log JAKA EDG force/torque data without ROS.

The loop runs at 125 Hz by default, mirrors the WBMM hardware-interface
compensation chain, and can optionally write a CSV for offline analysis.

This is a read-only tool: it never enables servo mode and never sends motion.
"""

from __future__ import annotations

import argparse
import csv
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from jaka_common import (
    DEFAULT_SENSOR_TO_TOOL_ROTATION,
    DEFAULT_TOOL_ARM_M,
    JakaError,
    JakaWristFTAdapter,
    RateLoop,
    add_network_args,
    edg_session,
    fmt_array,
    make_client,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_network_args(parser)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--rate-hz", type=float, default=125.0,
                        help="EDG polling rate; 125 Hz matches step_num=1")
    parser.add_argument("--print-hz", type=float, default=5.0)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--frame-id", default="jaka_tool")
    parser.add_argument("--tare-samples", type=int, default=50)
    parser.add_argument("--tare-period-ms", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--deadband-force", type=float, default=1.0)
    parser.add_argument("--deadband-torque", type=float, default=0.2)
    parser.add_argument("--torque-sensor-mode", type=int, default=1,
                        help="passed to set_torque_sensor_mode before EDG; negative disables")
    parser.add_argument("--identity-transform", action="store_true",
                        help="skip R_sensor_to_tool and r_t_s (for raw diagnostics)")
    return parser.parse_args()


def _csv_context(path: Path | None):
    if path is None:
        return nullcontext(None)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", newline="", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if args.seconds <= 0:
        raise SystemExit("--seconds must be positive")
    if args.rate_hz <= 0:
        raise SystemExit("--rate-hz must be positive")
    if args.print_hz <= 0 or args.print_hz > args.rate_hz:
        raise SystemExit("--print-hz must be in (0, --rate-hz]")

    rotation = np.eye(3) if args.identity_transform else DEFAULT_SENSOR_TO_TOOL_ROTATION
    tool_arm = np.zeros(3) if args.identity_transform else DEFAULT_TOOL_ARM_M
    client = make_client(args)

    try:
        with edg_session(client, torque_sensor_mode=args.torque_sensor_mode) as client, \
                _csv_context(args.csv) as csv_file:
            writer = csv.writer(csv_file) if csv_file is not None else None
            if writer is not None:
                writer.writerow([
                    "time_s",
                    "joint0_rad", "joint1_rad", "joint2_rad",
                    "joint3_rad", "joint4_rad", "joint5_rad",
                    "raw_fx_n", "raw_fy_n", "raw_fz_n",
                    "raw_tx_nm", "raw_ty_nm", "raw_tz_nm",
                    "fx_n", "fy_n", "fz_n", "tx_nm", "ty_nm", "tz_nm",
                ])
                print(f"[ft] writing CSV: {args.csv}")

            ft = JakaWristFTAdapter(
                client,
                frame_id=args.frame_id,
                sensor_to_tool_rotation=rotation,
                tool_arm_m=tool_arm,
                filter_alpha=args.alpha,
                deadband_force_n=args.deadband_force,
                deadband_torque_nm=args.deadband_torque,
            )
            print(f"[ft] taring with {args.tare_samples} samples...")
            bias = ft.tare(
                samples=args.tare_samples,
                period_s=args.tare_period_ms / 1000.0,
            )
            print(f"[ft] bias: {fmt_array(bias, 4)}")

            print_period = 1.0 / args.print_hz
            next_print = 0.0
            rows = 0
            loop = RateLoop(args.rate_hz)
            for t, _now in loop:
                state = client.read_edg_state()
                wrench = ft.read_wrench_from(state)
                rows += 1

                if writer is not None:
                    writer.writerow([
                        f"{t:.6f}",
                        *[f"{value:.9f}" for value in state.joint_position_rad],
                        *[f"{value:.9f}" for value in state.torque_sensor],
                        *[f"{value:.9f}" for value in wrench.as_vector()],
                    ])

                if t >= next_print:
                    print(
                        f"[ft] t={t:6.3f}s "
                        f"raw={fmt_array(state.torque_sensor, 4)} "
                        f"filt={fmt_array(wrench.as_vector(), 4)}"
                    )
                    next_print += print_period

                if t >= args.seconds:
                    break

            elapsed = t if rows else 0.0
            print(
                f"[ft] done: {rows} samples in {elapsed:.3f}s "
                f"({rows / elapsed if elapsed else 0.0:.1f} Hz requested "
                f"{args.rate_hz:.1f} Hz, overruns={loop.overruns})"
            )
    except KeyboardInterrupt:
        print("\n[ft] interrupted by user")
    except JakaError as exc:
        print(f"[ft] FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
