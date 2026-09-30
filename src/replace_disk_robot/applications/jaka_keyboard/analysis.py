"""Offline, hardware-free checks of logged discrete admittance and tracking.

Checks use independent algebra and actual per-cycle dt. Measured contact force
is treated as an input; this does not predict a new hardware contact trajectory.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from replace_disk_robot.core.rotation import rotation_matrix, rotation_vector_from_matrix


def load_session(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported log schema version")
    records = []
    malformed = 0
    with (directory / "samples.jsonl").open() as file:
        for line in file:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            records.append(row)
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    return metadata, records, summary, malformed


def analyze_session(directory):
    metadata, records, summary, malformed = load_session(directory)
    args = metadata["effective_args"]
    mass = np.asarray(args["adm_mass"], dtype=float)
    damping = np.asarray(args["adm_damping"], dtype=float)
    stiffness = np.asarray(args["adm_stiffness"], dtype=float)
    limit = np.asarray(args["adm_max_velocity"], dtype=float)
    residuals, position_errors, angle_errors, masked_input_errors = [], [], [], []
    inconsistent_inputs, skipped, clipped, clamped, observation_errors = 0, 0, 0, 0, 0
    cycle_metrics = []
    for row in records:
        observation_errors += bool(row.get("observer_errors") or
                                   row.get("wrench_stages", {}).get("capture_error"))
        before, after = row.get("admittance_before"), row.get("admittance_integrated")
        residual = None
        if row.get("admittance_updated") and before and after and not row.get("observer_errors"):
            try:
                x0, v0 = np.asarray(before["offset"], float), np.asarray(before["velocity"], float)
                force = np.asarray(row["admittance_input_wrench"], float)
                dt = float(row["dt_s"])
                x1, v1 = np.asarray(after["offset"], float), np.asarray(after["velocity"], float)
                if not np.all(np.isfinite(np.r_[x0, v0, force, x1, v1, dt])) or dt <= 0:
                    raise ValueError("invalid replay input")
                predicted_v = np.minimum(np.maximum(v0+dt*(force-damping*v0-stiffness*x0)/mass, -limit), limit)
                predicted_x = x0+dt*predicted_v
                residual = np.r_[x1-predicted_x, v1-predicted_v]
                residuals.append(residual)
                processed = np.asarray(row["wrench_stages"]["processed"], float)
                masked_error = force-processed*np.asarray(before["enabled_axes"], bool)
                masked_input_errors.append(masked_error)
                inconsistent_inputs += bool(np.max(np.abs(masked_error)) > 1e-12)
                clipped += bool(np.any(row.get("velocity_clipped_axes", [])))
                clamped += bool(row.get("translation_offset_clamped") or row.get("rotation_offset_clamped"))
            except (KeyError, TypeError, ValueError):
                skipped += 1
        else:
            skipped += 1
        target, actual = row.get("corrected_pose"), row.get("measured_tcp_pose")
        tracking_mm, tracking_deg = None, None
        if (row.get("admittance_updated") and target and actual
                and target["frame_id"] == actual["frame_id"]):
            try:
                delta = np.asarray(target["position_m"], float)-np.asarray(actual["position_m"], float)
                rt = rotation_matrix(np.asarray(target["quaternion_wxyz"], float))
                ra = rotation_matrix(np.asarray(actual["quaternion_wxyz"], float))
                angle = np.linalg.norm(rotation_vector_from_matrix(rt @ ra.T))
                if np.all(np.isfinite(delta)) and np.isfinite(angle):
                    position_errors.append(delta)
                    angle_errors.append(angle)
                    tracking_mm = float(np.linalg.norm(delta)*1000)
                    tracking_deg = float(np.rad2deg(angle))
            except (ValueError, TypeError):
                pass
        cycle_metrics.append({
            "sequence": row["sequence"], "elapsed_s": row["elapsed_s"], "dt_s": row["dt_s"],
            "admittance_updated": bool(row.get("admittance_updated")),
            "max_discrete_residual": None if residual is None else float(np.max(np.abs(residual))),
            "reference_tracking_error_mm": tracking_mm,
            "reference_tracking_error_deg": tracking_deg,
            "fault": row.get("fault"), "command_sent": bool(row.get("command_sent")),
        })
    sequences = [r["sequence"] for r in records]
    gaps = sum(max(0, b-a-1) for a, b in zip(sequences, sequences[1:]))
    time_order_ok = all(b["elapsed_s"] > a["elapsed_s"] for a, b in zip(records, records[1:]))
    sequence_order_ok = all(b > a for a, b in zip(sequences, sequences[1:]))
    dts = np.asarray([r["dt_s"] for r in records], dtype=float)
    dts = dts[np.isfinite(dts) & (dts > 0)]
    max_residual = float(np.max(np.abs(residuals))) if residuals else None
    sample_count_matches = (len(records) == summary["written_samples"]
                            if summary and "written_samples" in summary else None)
    complete = bool(summary and summary.get("complete") and not malformed and not gaps
                    and time_order_ok and sequence_order_ok and not observation_errors
                    and sample_count_matches is not False)
    report = {
        "scope": "offline algebra/record integrity/reference tracking; no hardware acceptance",
        "rows": len(records), "checked_admittance_cycles": len(residuals),
        "skipped_admittance_cycles": skipped, "malformed_lines": malformed,
        "missing_sequences_between_rows": gaps, "time_order_ok": time_order_ok,
        "sequence_order_ok": sequence_order_ok, "observation_error_cycles": observation_errors,
        "recording_complete": complete, "writer_summary": summary,
        "sample_count_matches_summary": sample_count_matches,
        "max_discrete_residual": max_residual,
        "max_masked_input_error": float(np.max(np.abs(masked_input_errors))) if masked_input_errors else None,
        "inconsistent_input_cycles": inconsistent_inputs,
        "discrete_equation_matches": None if max_residual is None else max_residual < 1e-10,
        "velocity_clipped_cycles": clipped, "offset_clamped_cycles": clamped,
        "fault_cycles": sum(bool(r.get("fault")) for r in records),
        "dt_ms": None if not len(dts) else {
            "median": float(np.median(dts)*1000), "p95": float(np.percentile(dts,95)*1000),
            "max": float(np.max(dts)*1000),
        },
        "reference_tracking_rms_mm": (float(np.sqrt(np.mean(np.sum(np.square(position_errors),axis=1)))*1000)
                                       if position_errors else None),
        "reference_tracking_rms_deg": (float(np.rad2deg(np.sqrt(np.mean(np.square(angle_errors)))))
                                        if angle_errors else None),
    }
    return report, cycle_metrics, records


def make_plots(records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    t = [r["elapsed_s"] for r in records]
    raw = np.asarray([r.get("raw_sensor_wrench", [None]*6) for r in records], dtype=float)
    processed = np.asarray([r.get("wrench_stages", {}).get("processed", [None]*6) for r in records], float)
    for i, name in enumerate(("Fx", "Fy", "Fz")):
        # Raw and processed axes may differ: distinct panels/labels avoid equating them.
        axes[0].plot(t, raw[:, i], label=f"raw sensor {name}", alpha=.65)
        axes[1].plot(t, processed[:, i], label=f"processed TCP {name}")
    for i, name in enumerate(("x", "y", "z")):
        offset = [((r.get("admittance_integrated") or {}).get("offset", [np.nan]*6))[i]*1000
                  for r in records]
        axes[2].plot(t, offset, label=f"offset {name}")
    axes[3].plot(t, [r["dt_s"]*1000 for r in records], label="actual dt")
    for row in records:
        if row.get("fault"):
            axes[3].axvline(row["elapsed_s"], color="red", alpha=.15)
    for ax, unit in zip(axes, ("sensor force [N]", "processed force [N]", "offset [mm]", "dt [ms]")):
        ax.set_ylabel(unit)
        ax.legend(loc="upper right")
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("elapsed [s]")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    report, metrics, records = analyze_session(args.session)
    output = args.output_dir or args.session / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "cycle_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        if metrics:
            writer = csv.DictWriter(file, fieldnames=metrics[0])
            writer.writeheader()
            writer.writerows(metrics)
    if args.plots and records:
        make_plots(records, output / "control_overview.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
