#!/usr/bin/env python3
"""Plot raw and offline gravity-compensated F/T data from a capture CSV.

The compensation is reconstructed from the payload-identification JSON:

    f_comp = f_raw - b_f - R_base_sensor.T @ h_base
    t_comp = t_raw - b_t - r_com_sensor x (R_base_sensor.T @ h_base)

By default, accepted static captures (capture_id >= 0) are preferred. If the
CSV has no accepted captures at all (for example, a slow quasi-static session
where every row is capture_id=-1), all rows are used automatically unless
``--capture-id`` was supplied. The CSV's processed_* columns are intentionally
not used: they may also include frame transforms, filtering and deadbands from
the online controller.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CSV = Path(__file__).with_name("gravity_test01.csv")
RAW_FORCE_COLUMNS = ("raw_fx_n", "raw_fy_n", "raw_fz_n")
RAW_TORQUE_COLUMNS = ("raw_tx_nm", "raw_ty_nm", "raw_tz_nm")
ROTATION_COLUMNS = tuple(
    f"r_base_sensor_{row}{col}" for row in range(3) for col in range(3)
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=DEFAULT_CSV,
        help=f"capture CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--identified",
        type=Path,
        default=None,
        help="payload fit JSON (default: <CSV stem>_identified.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output (default: <CSV stem>_gravity_comparison.png)",
    )
    parser.add_argument(
        "--capture-id",
        type=int,
        default=None,
        help="plot one accepted static capture only",
    )
    parser.add_argument(
        "--include-noncapture",
        action="store_true",
        help="also plot rows outside accepted static capture windows",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="open an interactive plot window after saving the PNG",
    )
    return parser.parse_args()


def _load_rows(
    csv_path: Path,
    *,
    capture_id: int | None,
    include_noncapture: bool,
) -> dict[str, Any]:
    required = {
        "time_s",
        "capture_id",
        *RAW_FORCE_COLUMNS,
        *RAW_TORQUE_COLUMNS,
        *ROTATION_COLUMNS,
    }
    all_rows: list[tuple[int, dict[str, str]]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            try:
                row_capture_id = int(row["capture_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid capture_id on CSV line {line_number}") from exc
            all_rows.append((row_capture_id, row))

    selected = [
        row
        for row_capture_id, row in all_rows
        if (capture_id is None or row_capture_id == capture_id)
        and (include_noncapture or row_capture_id >= 0)
    ]
    auto_included_noncapture = False
    if not selected and capture_id is None and not include_noncapture:
        # Common case for slow-motion datasets: the collector never marked an
        # accepted window, so every row has capture_id=-1.  Fall back to all
        # rows automatically instead of failing with "no rows".
        selected = [row for _row_capture_id, row in all_rows]
        auto_included_noncapture = True

    if not selected:
        target = f"capture_id={capture_id}" if capture_id is not None else "the selected rows"
        raise ValueError(
            f"no CSV rows found for {target}; if the CSV uses capture_id=-1, "
            "pass --include-noncapture"
        )

    def values(columns: tuple[str, ...]) -> np.ndarray:
        try:
            array = np.asarray([[float(row[name]) for name in columns] for row in selected])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"CSV contains an invalid value in {columns}") from exc
        if not np.isfinite(array).all():
            raise ValueError(f"CSV contains non-finite data in {columns}")
        return array

    times = values(("time_s",)).ravel()
    ids = np.asarray([int(row["capture_id"]) for row in selected], dtype=int)
    rotations = values(ROTATION_COLUMNS).reshape(-1, 3, 3)
    result = {
        "time_s": times,
        "capture_id": ids,
        "auto_included_noncapture": auto_included_noncapture,
        "rotation_base_sensor": rotations,
        "raw_force_n": values(RAW_FORCE_COLUMNS),
        "raw_torque_nm": values(RAW_TORQUE_COLUMNS),
    }
    extrinsic_columns = tuple(f"r_sensor_tool_{i}{j}" for i in range(3) for j in range(3))
    if all(name in selected[0] for name in extrinsic_columns):
        result["rotation_sensor_to_tool_recorded"] = values(extrinsic_columns).reshape(-1, 3, 3)
    return result


def _load_payload_fit(path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            fit = json.load(stream)
        if not isinstance(fit, dict):
            raise ValueError("payload fit root must be a JSON object")
        result = {
            "gravity_force_base_n": np.asarray(
                fit["signed_gravity_force_base_n"], dtype=float
            ),
            "center_of_mass_sensor_m": np.asarray(
                fit["center_of_mass_sensor_m"], dtype=float
            ),
            "force_bias_sensor_n": np.asarray(fit["force_bias_sensor_n"], dtype=float),
            "torque_bias_sensor_nm": np.asarray(fit["torque_bias_sensor_nm"], dtype=float),
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot read payload identification file {path}: {exc}") from exc
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in result.values()):
        raise ValueError(f"payload fit in {path} must contain finite 3D vectors")
    result["stale_tool0_extrinsics_repaired"] = fit.get("stale_tool0_extrinsics_repaired") is True
    warnings = fit.get("warnings", [])
    if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        raise ValueError(f"warnings in {path} must be a list of strings")
    return result, warnings


def _compensate(
    data: dict[str, Any], fit: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    rotations = data["rotation_base_sensor"]
    if fit.get("stale_tool0_extrinsics_repaired"):
        if "rotation_sensor_to_tool_recorded" not in data:
            raise ValueError("repaired payload fit requires the original CSV sensor extrinsics")
        from identify_ft_payload import _project_rotation
        rotations = np.array([
            _project_rotation(base) @ _project_rotation(old).T @ np.diag([-1., -1., 1.]) @ _project_rotation(old)
            for base, old in zip(rotations, data["rotation_sensor_to_tool_recorded"])
        ])
    gravity_force_sensor = np.einsum(
        "nji,j->ni", rotations, fit["gravity_force_base_n"]
    )
    gravity_torque_sensor = np.cross(
        fit["center_of_mass_sensor_m"][None, :], gravity_force_sensor
    )
    force_compensated = (
        data["raw_force_n"] - fit["force_bias_sensor_n"] - gravity_force_sensor
    )
    torque_compensated = (
        data["raw_torque_nm"] - fit["torque_bias_sensor_nm"] - gravity_torque_sensor
    )
    return force_compensated, torque_compensated


def _plot(
    data: dict[str, Any],
    force_compensated: np.ndarray,
    torque_compensated: np.ndarray,
    output_path: Path,
    *,
    show: bool,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    time_s = data["time_s"]
    capture_ids = data["capture_id"]
    time_s = time_s - float(time_s[0])
    channels = (
        ("F_x", "N", data["raw_force_n"][:, 0], force_compensated[:, 0], 1.0),
        ("F_y", "N", data["raw_force_n"][:, 1], force_compensated[:, 1], 1.0),
        ("F_z", "N", data["raw_force_n"][:, 2], force_compensated[:, 2], 1.0),
        ("T_x", "N·m", data["raw_torque_nm"][:, 0], torque_compensated[:, 0], 0.1),
        ("T_y", "N·m", data["raw_torque_nm"][:, 1], torque_compensated[:, 1], 0.1),
        ("T_z", "N·m", data["raw_torque_nm"][:, 2], torque_compensated[:, 2], 0.1),
    )
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for axis, (name, unit, raw, compensated, minimum_half_range) in zip(
        axes.flat, channels, strict=True
    ):
        segment_starts = np.r_[0, np.flatnonzero(capture_ids[1:] != capture_ids[:-1]) + 1]
        segment_ends = np.r_[segment_starts[1:], len(capture_ids)]
        for segment_index, (start, end) in enumerate(
            zip(segment_starts, segment_ends, strict=True)
        ):
            axis.plot(time_s[start:end], raw[start:end], color="tab:blue", linewidth=1.8, alpha=0.9)
            axis.plot(
                time_s[start:end],
                compensated[start:end],
                color="tab:orange",
                linewidth=1.8,
                alpha=0.95,
            )
            if segment_index + 1 < len(segment_starts):
                next_start = segment_starts[segment_index + 1]
                axis.plot(
                    time_s[end - 1 : next_start + 1],
                    raw[end - 1 : next_start + 1],
                    color="tab:blue",
                    linewidth=1.5,
                    linestyle=":",
                    alpha=0.7,
                )
                axis.plot(
                    time_s[end - 1 : next_start + 1],
                    compensated[end - 1 : next_start + 1],
                    color="tab:orange",
                    linewidth=1.5,
                    linestyle=":",
                    alpha=0.7,
                )
        axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
        axis.set_title(name)
        axis.set_ylabel(unit)
        half_range = max(
            minimum_half_range,
            1.05 * float(np.max(np.abs(np.concatenate((raw, compensated))))),
        )
        axis.set_ylim(-half_range, half_range)
        axis.grid(True, alpha=0.25)

    axes[-1, 0].set_xlabel("Time from first selected sample (s)")
    axes[-1, 1].set_xlabel("Time from first selected sample (s)")
    pose_count = len(np.unique(capture_ids[capture_ids >= 0]))
    fig.suptitle(
        f"F/T before and after payload gravity compensation "
        f"({pose_count} static captures, {len(time_s)} samples)",
        fontsize=14,
    )
    fig.legend(
        handles=(
            Line2D([0], [0], color="tab:blue", lw=2.2, label="Before: raw sensor wrench"),
            Line2D(
                [0],
                [0],
                color="tab:orange",
                lw=2.2,
                label="After: raw - fitted gravity - sensor bias",
            ),
            Line2D([0], [0], color="gray", lw=1.2, linestyle=":", label="Between captures (no samples)"),
        ),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.95),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    csv_path = args.csv.expanduser().resolve()
    identified_path = (
        args.identified.expanduser().resolve()
        if args.identified
        else csv_path.with_name(f"{csv_path.stem}_identified.json")
    )
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else csv_path.with_name(f"{csv_path.stem}_gravity_comparison.png")
    )
    if not csv_path.is_file():
        raise FileNotFoundError(f"capture CSV not found: {csv_path}")
    if not identified_path.is_file():
        raise FileNotFoundError(f"payload fit JSON not found: {identified_path}")

    data = _load_rows(
        csv_path,
        capture_id=args.capture_id,
        include_noncapture=args.include_noncapture,
    )
    if data.get("auto_included_noncapture"):
        print(
            "[plot] no capture_id>=0 rows found; automatically using all rows "
            "(same effect as --include-noncapture)"
        )
    fit, warnings = _load_payload_fit(identified_path)
    force_compensated, torque_compensated = _compensate(data, fit)
    for warning in warnings:
        print(f"[fit warning] {warning}")
    _plot(
        data,
        force_compensated,
        torque_compensated,
        output_path,
        show=args.show,
    )

    force_rms_before = float(np.sqrt(np.mean(np.sum(data["raw_force_n"] ** 2, axis=1))))
    force_rms_after = float(np.sqrt(np.mean(np.sum(force_compensated ** 2, axis=1))))
    torque_rms_before = float(np.sqrt(np.mean(np.sum(data["raw_torque_nm"] ** 2, axis=1))))
    torque_rms_after = float(np.sqrt(np.mean(np.sum(torque_compensated ** 2, axis=1))))
    print(f"Saved plot: {output_path}")
    print("RMS values describe the plotted samples and are not a substitute for an independent validation capture.")
    print(
        f"Selected-sample RMS |F|: {force_rms_before:.3f} -> "
        f"{force_rms_after:.3f} N; |T|: {torque_rms_before:.3f} -> "
        f"{torque_rms_after:.3f} N·m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
