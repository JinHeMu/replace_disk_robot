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
    controls, _, _ = control_analysis(records, export_samples=False)
    report.update(controls)
    return report, cycle_metrics, records


def _six_axis_series(records, getter):
    """Return a NaN-padded N x 6 array for a possibly absent logged signal."""
    values = np.full((len(records), 6), np.nan, dtype=float)
    for index, row in enumerate(records):
        try:
            value = np.asarray(getter(row), dtype=float)
            if value.shape == (6,):
                values[index] = value
        except (TypeError, ValueError):
            pass
    return values

AXES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
MOTION_AXES = ("x", "y", "z", "rx", "ry", "rz")


def _signal(records, getter, frame_getter, label):
    values = _six_axis_series(records, getter)
    frames = {frame_getter(row) or "unknown" for row, value in zip(records, values)
              if np.any(np.isfinite(value))}
    if len(frames) > 1:
        raise ValueError(f"{label} changes coordinate frame: {sorted(frames)}")
    return values, next(iter(frames), "unknown")


def _pose_series(records, getter, label):
    positions = np.full((len(records), 3), np.nan)
    rotations = np.full((len(records), 3, 3), np.nan)
    frames = set()
    for index, row in enumerate(records):
        pose = getter(row)
        if not pose:
            continue
        try:
            position = np.asarray(pose["position_m"], float)
            rotation = rotation_matrix(np.asarray(pose["quaternion_wxyz"], float))
            if position.shape != (3,) or not np.isfinite(np.r_[position, rotation.ravel()]).all():
                continue
            positions[index], rotations[index] = position, rotation
            frames.add(pose.get("frame_id") or "unknown")
        except (KeyError, TypeError, ValueError):
            continue
    if len(frames) > 1:
        raise ValueError(f"{label} changes coordinate frame: {sorted(frames)}")
    return positions, rotations, next(iter(frames), "unknown")


def _pose_delta(position, rotation, origin_position, origin_rotation):
    """Translation and spatial rotation vector, both in the pose's base axes."""
    result = np.full((len(position), 6), np.nan)
    result[:, :3] = position - origin_position
    for index, (current, origin) in enumerate(zip(rotation, origin_rotation)):
        if np.isfinite(np.r_[current.ravel(), origin.ravel()]).all():
            result[index, 3:] = rotation_vector_from_matrix(current @ origin.T)
    return result


def _display_motion(values):
    result = values.copy()
    result[:, :3] *= 1000
    result[:, 3:] = np.rad2deg(result[:, 3:])
    return result


def _statistics(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "rms": None, "min": None,
                "max": None, "peak_abs": None}
    return {"count": len(values), "mean": float(np.mean(values)),
            "rms": float(np.sqrt(np.mean(values**2))), "min": float(np.min(values)),
            "max": float(np.max(values)), "peak_abs": float(np.max(np.abs(values)))}


def control_analysis(records, *, export_samples=True):
    """Prepare frame-checked data shared by tables, CSV export and plots.

    Pose comparisons are spatial rotation vectors, not Euler angle subtraction.
    Missing signals stay NaN internally and become empty CSV cells / JSON null.
    """
    stage = lambda row: row.get("wrench_stages") or {}
    raw, raw_frame = _signal(records, lambda row: row.get("raw_sensor_wrench"),
                             lambda row: row.get("raw_sensor_frame_id"), "raw wrench")
    comp, comp_frame = _signal(records, lambda row: stage(row).get("compensated_sensor"),
                              lambda row: stage(row).get("compensated_sensor_frame"),
                              "compensated sensor wrench")
    processed, wrench_frame = _signal(records, lambda row: stage(row).get("processed"),
                                      lambda row: stage(row).get("frame_id"), "processed wrench")
    external, external_frame = _signal(
        records, lambda row: stage(row).get("external_tcp_unfiltered"),
        lambda row: stage(row).get("frame_id"), "compensated TCP wrench")
    actual_input, input_frame = _signal(
        records, lambda row: row.get("admittance_input_wrench") if row.get("admittance_updated") else None,
        lambda row: (row.get("admittance_integrated") or row.get("admittance_before") or {}).get("frame_id"),
        "admittance input")
    integrated = lambda row: (row.get("admittance_integrated") or {}) if row.get("admittance_updated") else {}
    offset, adm_frame = _signal(records, lambda row: integrated(row).get("offset"),
                               lambda row: integrated(row).get("frame_id"), "admittance offset")
    velocity, _ = _signal(records, lambda row: integrated(row).get("velocity"),
                         lambda row: integrated(row).get("frame_id"), "admittance velocity")
    known = {frame for values, frame in ((external, external_frame), (processed, wrench_frame),
             (actual_input, input_frame), (offset, adm_frame))
             if np.any(np.isfinite(values))}
    if len(known) > 1:
        raise ValueError(f"external wrench and admittance axes do not match: {sorted(known)}")

    warnings = []
    comp_source = "compensated_sensor"
    if not np.any(np.isfinite(comp)):
        comp, comp_frame, comp_source = processed.copy(), wrench_frame, "processed"
        warnings.append("未记录传感器补偿阶段，力补偿表使用处理后 TCP 数据（包含滤波/死区），坐标系单独标注。")
    paired = np.isfinite(raw) & np.isfinite(comp)
    comparable = raw_frame == comp_frame and raw_frame != "unknown"
    raw_available = bool(np.isfinite(raw).any())
    paired_available = bool(comparable and paired.any())
    if np.any(np.isfinite(raw)) and np.any(np.isfinite(comp)) and not comparable:
        warnings.append("原始与补偿信号坐标系不同或未知，仅分别统计，不计算差值、不叠加曲线。")
    force_table = []
    for index, axis in enumerate(AXES):
        pair = paired[:, index] if comparable else np.zeros(len(records), bool)
        force_table.append({"axis": axis, "unit": "N" if index < 3 else "N·m",
                            "raw_frame": raw_frame, "compensated_frame": comp_frame,
                            "raw": _statistics(raw[:, index]),
                            "compensated": _statistics(comp[:, index]),
                            "compensated_minus_raw": _statistics(comp[pair, index] - raw[pair, index])})

    pa, ra, base_frame = _pose_series(records, lambda row: row.get("measured_tcp_pose"), "measured TCP")
    pt, rt, target_frame = _pose_series(records, lambda row: row.get("corrected_pose"), "corrected TCP")
    pn, rn, nominal_frame = _pose_series(records, lambda row: integrated(row).get("nominal_pose"), "nominal TCP")
    pose_frames = {frame for pos, frame in ((pa, base_frame), (pt, target_frame), (pn, nominal_frame))
                   if np.any(np.isfinite(pos))}
    if len(pose_frames) > 1:
        raise ValueError(f"cannot compare TCP poses across different frames: {sorted(pose_frames)}")
    if pose_frames == {"unknown"}:
        pt[:], rt[:], pn[:], rn[:] = np.nan, np.nan, np.nan, np.nan
        warnings.append("TCP 位姿坐标系未知，跳过目标/名义/实测的相对位移比较。")
    base_frame = next(iter(pose_frames), "unknown")
    valid_actual = np.flatnonzero(np.isfinite(pa).all(axis=1) & np.isfinite(ra).all(axis=(1, 2)))
    origin_index = int(valid_actual[0]) if len(valid_actual) else None
    p0 = np.full_like(pa, np.nan) if origin_index is None else np.broadcast_to(pa[origin_index], pa.shape)
    r0 = np.full_like(ra, np.nan) if origin_index is None else np.broadcast_to(ra[origin_index], ra.shape)
    signals = {
        "external_tcp": external, "processed": processed, "input": actual_input,
        "offset": _display_motion(offset), "velocity": _display_motion(velocity),
        "actual_displacement_base": _display_motion(_pose_delta(pa, ra, p0, r0)),
        "target_displacement_base": _display_motion(_pose_delta(pt, rt, p0, r0)),
        "nominal_displacement_base": _display_motion(_pose_delta(pn, rn, p0, r0)),
        "limited_compliance_base": _display_motion(_pose_delta(pt, rt, pn, rn)),
        "actual_relative_nominal_base": _display_motion(_pose_delta(pa, ra, pn, rn)),
        "tracking_error_base": _display_motion(_pose_delta(pt, rt, pa, ra)),
    }
    adm_table = []
    for index, axis in enumerate(MOTION_AXES):
        adm_table.append({"axis": axis, "wrench_unit": "N" if index < 3 else "N·m",
                          "motion_unit": "mm" if index < 3 else "deg",
                          "external_tcp": _statistics(external[:, index]),
                          "processed": _statistics(processed[:, index]),
                          "input": _statistics(actual_input[:, index]),
                          **{name: _statistics(signals[name][:, index]) for name in (
                              "offset", "actual_displacement_base", "limited_compliance_base",
                              "actual_relative_nominal_base", "tracking_error_base")}})
    sections = {
        "force_control": {"compensated_source": comp_source, "raw_available": raw_available,
                          "compensated_available": bool(np.isfinite(comp).any()),
                          "paired_comparison_available": paired_available,
                          "gravity_compensation_cycles": sum(stage(row).get("gravity_compensation") is True for row in records),
                          "table": force_table, "warnings": warnings},
        "admittance_control": {"wrench_frame": wrench_frame, "offset_frame": adm_frame,
                               "pose_frame": base_frame,
                               "displacement_origin_sequence": None if origin_index is None else records[origin_index]["sequence"],
                               "table": adm_table,
                               "definitions": {
                                   "offset": "积分后、位姿限幅前的导纳偏移；沿该周期实测 TCP 轴",
                                   "limited_compliance_base": "限幅后 corrected - nominal；基座坐标系",
                                   "actual_relative_nominal_base": "同周期 measured - nominal；包含跟踪滞后，不能认定为纯导纳响应",
                                   "actual_displacement_base": "实测 TCP 相对首个有效实测位姿；包含键盘运动",
                                   "rotation": "log(R_current R_reference^T) 的旋转向量，沿基座轴，非欧拉角差",
                                   "timing": "实测反馈在本周期下发命令之前采集；同周期误差不是命令执行后的响应",
                               }},
    }
    force_rows, adm_rows = [], []
    for index, row in enumerate(records if export_samples else []):
        common = {"sequence": row["sequence"], "elapsed_s": row["elapsed_s"],
                  "admittance_updated": bool(row.get("admittance_updated")),
                  "command_sent": bool(row.get("command_sent")), "fault": row.get("fault")}
        force_row = {**common, "raw_frame": raw_frame, "compensated_frame": comp_frame,
                     "compensated_source": comp_source,
                     "gravity_compensation": stage(row).get("gravity_compensation")}
        for axis_index, axis in enumerate(AXES):
            unit = "N" if axis_index < 3 else "Nm"
            if raw_available:
                force_row[f"raw_{axis}_{unit}"] = _csv_value(raw[index, axis_index])
            force_row[f"compensated_{axis}_{unit}"] = _csv_value(comp[index, axis_index])
            if paired_available:
                force_row[f"compensated_minus_raw_{axis}_{unit}"] = _csv_value(comp[index, axis_index] - raw[index, axis_index])
        force_rows.append(force_row)
        adm_row = {**common, "wrench_frame": wrench_frame, "offset_frame": adm_frame, "pose_frame": base_frame}
        for name, values in signals.items():
            for axis_index, axis in enumerate(AXES if name in ("external_tcp", "processed", "input") else MOTION_AXES):
                unit = ("N" if axis_index < 3 else "Nm") if name in ("external_tcp", "processed", "input") else ("mm" if axis_index < 3 else "deg")
                if name == "velocity":
                    unit += "_s"
                adm_row[f"{name}_{axis}_{unit}"] = _csv_value(values[index, axis_index])
        adm_rows.append(adm_row)
    # Plot arrays live separately from the JSON report.
    data = {**signals, "raw": raw, "compensated": comp, "raw_frame": raw_frame,
            "compensated_frame": comp_frame, "comparable": comparable,
            "wrench_frame": wrench_frame, "offset_frame": adm_frame, "pose_frame": base_frame}
    return sections, (force_rows, adm_rows), data


def _csv_value(value):
    return float(value) if np.isfinite(value) else None


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        if rows:
            writer = csv.DictWriter(file, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)


def _format_number(value):
    return "—" if value is None else f"{value:.4g}"


def write_readable_report(report, output):
    force, adm = report["force_control"], report["admittance_control"]
    show_raw = force["raw_available"]
    lines = ["# JAKA 控制日志分析", "",
             f"采样 {report['rows']} 周期；记录完整：{report['recording_complete']}；"
             f"离散导纳公式一致：{report['discrete_equation_matches']}。", "",
             "## 1. 力补偿", "",
             f"补偿数据来源：`{force['compensated_source']}`。"
             f"重力补偿开启周期：{force['gravity_compensation_cycles']}。"
             "统计覆盖全部有效采样；缺失值不作为零参与统计。", ""]
    if show_raw:
        lines += ["|轴|单位|原始坐标系|原始样本数|原始均值 / RMS / 峰值绝对值|补偿坐标系|补偿样本数|补偿均值 / RMS / 峰值绝对值|配对样本数|补偿−原始均值|",
                  "|---|---|---|---:|---|---|---:|---|---:|---:|"]
    else:
        lines += ["未记录原始数据，仅展示补偿数据。", "",
                  "|轴|单位|补偿坐标系|有效样本数|均值|RMS|峰值绝对值|",
                  "|---|---|---|---:|---:|---:|---:|"]
    for row in force["table"]:
        stats = row["compensated"]
        if show_raw:
            raw, diff = row["raw"], row["compensated_minus_raw"]
            triple = lambda s: " / ".join(_format_number(s[key]) for key in ("mean", "rms", "peak_abs"))
            lines.append(f"|{row['axis']}|{row['unit']}|{row['raw_frame']}|{raw['count']}|{triple(raw)}|{row['compensated_frame']}|{stats['count']}|{triple(stats)}|{diff['count']}|{_format_number(diff['mean'])}|")
        else:
            lines.append(f"|{row['axis']}|{row['unit']}|{row['compensated_frame']}|{stats['count']}|{_format_number(stats['mean'])}|{_format_number(stats['rms'])}|{_format_number(stats['peak_abs'])}|")
    lines += ["", *force["warnings"], "",
              "补偿−原始表示被移除的信号差，不是补偿精度或已知真值误差。", "",
              "逐周期数据：`force_control_samples.csv`；曲线：`force_control_overview.png`（请求绘图时生成）。", "",
              "## 2. 导纳控制", "", "### 2.1 外力与控制器输出", "",
              f"外力与力矩在 `{adm['wrench_frame']}`，积分偏移在 `{adm['offset_frame']}`。"
              "分别保留补偿后 TCP 外力（滤波前）、处理后外力（滤波/死区后）、轴屏蔽后的实际导纳输入。", "",
              "|轴|力/矩单位|补偿 TCP 外力峰值|处理后外力峰值|实际输入峰值|位移单位|积分偏移峰值|",
              "|---|---|---:|---:|---:|---|---:|"]
    for row in adm["table"]:
        cells = [row['axis'], row['wrench_unit'], *[_format_number(row[key]['peak_abs']) for key in ('external_tcp', 'processed', 'input')], row['motion_unit'], _format_number(row['offset']['peak_abs'])]
        lines.append("|" + "|".join(cells) + "|")
    lines += ["", "曲线：`admittance_control_overview.png`（外力、积分偏移、速度）。", "",
              "### 2.2 机械臂实际运动", "",
              f"以下全部在 `{adm['pose_frame']}`。位移起点为首个有效实测位姿"
              f"（序号 {adm['displacement_origin_sequence']}）。实测 TCP 来自关节反馈的正运动学。", "",
              "|轴|单位|实际相对起点位移峰值|限幅后柔顺目标峰值|实测相对名义位姿峰值|同周期跟踪误差 RMS|",
              "|---|---|---:|---:|---:|---:|"]
    for row in adm["table"]:
        cells = [row['axis'], row['motion_unit'], *[_format_number(row[key]['peak_abs']) for key in ('actual_displacement_base', 'limited_compliance_base', 'actual_relative_nominal_base')], _format_number(row['tracking_error_base']['rms'])]
        lines.append("|" + "|".join(cells) + "|")
    lines += ["", *[f"- `{name}`：{meaning}。" for name, meaning in adm["definitions"].items()], "",
              "曲线：`admittance_tracking_overview.png`；逐周期数据：`admittance_control_samples.csv`。", "",
              "故障区间以红色阴影标记；积分输出或修正目标存在不代表命令已经发送，CSV 保留 command_sent/fault。", "",
              "公式检查及同周期参考误差属于离线分析，不作为接触稳定或实机验收结论。", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


def _shade_fault_intervals(axes, records):
    fault_indices = [i for i, row in enumerate(records) if row.get("fault")]
    if not fault_indices:
        return
    groups = np.split(fault_indices, np.flatnonzero(np.diff(fault_indices) > 1) + 1)
    for group in groups:
        first, last = int(group[0]), int(group[-1])
        start = float(records[first]["elapsed_s"])
        end = float(records[last]["elapsed_s"] + records[last].get("dt_s", 0.0))
        for ax in axes:
            ax.axvspan(start, end, color="red", alpha=0.09, linewidth=0)


def _plot_canvas(rows, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(rows, 2, figsize=(15, rows * 3.1), sharex=True, squeeze=False)
    fig.suptitle(title)
    return plt, fig, axes


def _finish_plot(plt, fig, axes, records, output):
    for ax in axes.ravel():
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend(fontsize=8, loc="best", ncol=3 if len(ax.lines) > 3 else 1)
        if not any(np.isfinite(np.asarray(line.get_ydata(), float)).any() for line in ax.lines):
            ax.text(.5, .5, "No valid logged samples", transform=ax.transAxes, ha="center")
    for ax in axes[-1]:
        ax.set_xlabel("elapsed [s]")
    _shade_fault_intervals(axes.ravel(), records)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def make_force_control_plot(records, output, data=None):
    data = control_analysis(records)[2] if data is None else data
    t = [row["elapsed_s"] for row in records]
    show_raw = np.isfinite(data["raw"]).any()
    # Separate panels when coordinates cannot be safely compared.
    separate = show_raw and not data["comparable"]
    plt, fig, axes = _plot_canvas(6 if separate else 3, "1. Force compensation: sensor wrench")
    for index, name in enumerate(AXES):
        ax = axes.ravel()[index * 2] if separate else axes.ravel()[index]
        if show_raw:
            ax.plot(t, data["raw"][:, index], label=f"raw ({data['raw_frame']})", alpha=.7)
        if separate:
            ax.set_title(name)
            ax.set_ylabel("N" if index < 3 else "N m")
            ax = axes.ravel()[index * 2 + 1]
        ax.plot(t, data["compensated"][:, index], label=f"compensated ({data['compensated_frame']})")
        ax.set_title(name)
        ax.set_ylabel("N" if index < 3 else "N m")
    _finish_plot(plt, fig, axes, records, output)


def _plot_three(ax, t, values, names, prefix="", style="-"):
    for index, name in enumerate(names):
        ax.plot(t, values[:, index], style, color=f"C{index}",
                label=f"{prefix}{name}", linewidth=1)


def make_admittance_control_plot(records, output, data=None):
    data = control_analysis(records)[2] if data is None else data
    t = [row["elapsed_s"] for row in records]
    plt, fig, axes = _plot_canvas(3, "2.1 Admittance: compensated external wrench and controller output")
    for column, (names, sl, unit, motion_unit) in enumerate(((AXES[:3], slice(0, 3), "N", "mm"),
                                                          (AXES[3:], slice(3, 6), "N m", "deg"))):
        ax = axes[0, column]
        for name, prefix, style in (("external_tcp", "compensated TCP ", ":"),
                                    ("processed", "processed ", "-"), ("input", "masked input ", "--")):
            if np.isfinite(data[name][:, sl]).any():
                _plot_three(ax, t, data[name][:, sl], names, prefix, style)
        ax.set_title(f"External wrench / input ({data['wrench_frame']})")
        ax.set_ylabel(unit)
        for row, name, label, suffix in ((1, "offset", "Integrated offset BEFORE pose clamp", ""),
                                          (2, "velocity", "Integrated velocity", "/s")):
            _plot_three(axes[row, column], t, data[name][:, sl], MOTION_AXES[sl])
            axes[row, column].set_title(f"{label} ({data['offset_frame']}, measured TCP axes)")
            axes[row, column].set_ylabel(motion_unit + suffix)
    _finish_plot(plt, fig, axes, records, output)


def make_admittance_tracking_plot(records, output, data=None):
    data = control_analysis(records)[2] if data is None else data
    t = [row["elapsed_s"] for row in records]
    plt, fig, axes = _plot_canvas(4, f"2.2 Robot motion: all vectors in {data['pose_frame']}")
    for column, (sl, unit) in enumerate(((slice(0, 3), "mm"), (slice(3, 6), "deg"))):
        names = MOTION_AXES[sl]
        for name, prefix, style in (("limited_compliance_base", "limited target ", "--"),
                                    ("actual_relative_nominal_base", "measured ", "-")):
            _plot_three(axes[0, column], t, data[name][:, sl], names, prefix, style)
        axes[0, column].set_title("Compliance relative to CURRENT nominal pose")
        for name, prefix, style in (("actual_displacement_base", "measured ", "-"),
                                    ("target_displacement_base", "corrected ", "--"),
                                    ("nominal_displacement_base", "nominal ", ":")):
            _plot_three(axes[1, column], t, data[name][:, sl], names, prefix, style)
        axes[1, column].set_title("Displacement from FIRST measured pose (includes keyboard motion)")
        _plot_three(axes[2, column], t, data["tracking_error_base"][:, sl], names)
        axes[2, column].set_title("Same-cycle corrected minus measured (feedback PRECEDES command)")
        for row in range(3):
            axes[row, column].set_ylabel(unit)
    axes[3, 0].plot(t, [row.get("dt_s", np.nan) * 1000 for row in records], label="recorded dt")
    axes[3, 0].set_title("Control period (red shading: fault)")
    axes[3, 0].set_ylabel("ms")
    axes[3, 1].step(t, [bool(row.get("command_sent")) for row in records], where="post", label="command sent")
    axes[3, 1].step(t, [bool(row.get("admittance_updated")) for row in records], where="post", label="admittance updated", alpha=.6)
    axes[3, 1].set_title("Recorded command / update status")
    axes[3, 1].set_ylabel("0 / 1")
    _finish_plot(plt, fig, axes, records, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plots", action="store_true", help="generate force, admittance and robot-motion plots")
    parser.add_argument("--plot-force-control", action="store_true", help="generate force compensation plot")
    parser.add_argument("--plot-admittance-control", action="store_true", help="generate both admittance and robot-motion plots")
    args = parser.parse_args()
    report, metrics, records = analyze_session(args.session)
    _, (force_rows, adm_rows), data = control_analysis(records)
    output = args.output_dir or args.session / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readable_report(report, output / "report.md")
    _write_csv(output / "cycle_metrics.csv", metrics)
    _write_csv(output / "force_control_samples.csv", force_rows)
    _write_csv(output / "admittance_control_samples.csv", adm_rows)
    if records and (args.plots or args.plot_force_control):
        make_force_control_plot(records, output / "force_control_overview.png", data)
    if records and (args.plots or args.plot_admittance_control):
        make_admittance_control_plot(records, output / "admittance_control_overview.png", data)
        make_admittance_tracking_plot(records, output / "admittance_tracking_overview.png", data)
    print((output / "report.md").read_text(encoding="utf-8"))
    print(f"\n分析报告（力补偿表 / 导纳输入输出 / 实际运动）：{output / 'report.md'}")


if __name__ == "__main__":
    main()
