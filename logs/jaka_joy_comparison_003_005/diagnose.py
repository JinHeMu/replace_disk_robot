"""Offline audit of recorded forces, references, model geometry and tracking.

Reads captured samples only. Does not create a robot client or send commands.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "logs/force_direction_audit"))
from reproduce_audit import fk, recorded_urdf
from replace_disk_robot.core.rotation import rotation_matrix, rotation_vector_from_matrix


def pose(rows, key):
    return np.array([r[key]["position_m"] if r.get(key) else [np.nan]*3 for r in rows])


def wrench_series(rows, key):
    return np.array([r.get("wrench_stages", {}).get(key, [np.nan]*6) for r in rows])


def load(trial):
    directory = ROOT / "logs" / f"jaka_joy_trial_{trial:03d}"
    m = json.loads((directory / "metadata.json").read_text())
    raw = (directory / "samples.jsonl").read_bytes()
    rows, malformed = [], 0
    for line in raw.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            malformed += 1
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    xml = recorded_urdf(m)["content"]
    q = np.array([r["measured_q_rad"] for r in rows])
    p, R, _ = fk(xml, q)
    active = np.array([r.get("admittance_updated", False) and not r.get("fault") for r in rows])
    nominal = np.array([r["admittance_integrated"]["nominal_pose"]["position_m"]
                        if r.get("admittance_updated") else [np.nan]*3 for r in rows])
    return dict(trial=trial, rows=rows, m=m, xml=xml, p=p, R=R, active=active,
                nominal=nominal, reference=pose(rows, "corrected_pose"),
                command=pose(rows, "final_command_tcp_pose"),
                t=np.array([r["elapsed_s"] for r in rows]),
                offset=np.array([r["admittance_after"]["offset"] for r in rows]),
                raw=np.array([r["raw_sensor_wrench"] for r in rows]),
                comp=wrench_series(rows, "compensated_sensor"),
                external=wrench_series(rows, "external_tcp_unfiltered"),
                filtered=wrench_series(rows, "filtered_before_deadband"),
                processed=wrench_series(rows, "processed"),
                input_bytes=len(raw), input_sha256=hashlib.sha256(raw).hexdigest(),
                malformed=malformed, summary=summary)


def audit(d):
    m, rows, R, p = d["m"], d["rows"], d["R"], d["p"]
    args, calibration = m["effective_args"], m["runtime_calibration"]
    gc = calibration["gravity_compensation_config"]
    rts, lever = np.array(m["sensor_to_tool_rotation"]), np.array(m["tool_to_sensor_m"])
    rbs = R @ rts
    gravity = gc["load_sign"]*gc["payload_mass_kg"]*np.array(gc["gravity_m_s2"])
    fg = np.einsum("nij,i->nj", rbs, gravity)
    predicted_comp = d["raw"] - np.array(calibration["gravity_bias_sensor"]) - np.c_[
        fg, np.cross(np.array(gc["payload_com_sensor_m"]), fg)]
    f = d["comp"][:, :3] @ rts.T
    direct = d["comp"][:, 3:] @ rts.T
    shifted = np.cross(lever, f)
    predicted_ext = np.c_[f, direct + shifted]
    predicted_out = d["filtered"].copy()
    predicted_out[:, :3][np.abs(predicted_out[:, :3]) < args["deadband_force"]] = 0
    predicted_out[:, 3:][np.abs(predicted_out[:, 3:]) < args["deadband_torque"]] = 0
    filter_residual, dynamics, application = [], [], []
    mass, damping, stiffness, limits = [np.array(args[k]) for k in (
        "adm_mass", "adm_damping", "adm_stiffness", "adm_max_velocity")]
    for i, row in enumerate(rows):
        if i and d["active"][i] and d["active"][i-1]:
            fb = R[i-1] @ d["filtered"][i-1, :3]
            tb = R[i-1] @ d["filtered"][i-1, 3:] + np.cross(p[i-1]-p[i], fb)
            previous = np.r_[R[i].T @ fb, R[i].T @ tb]
            prediction = args["alpha"]*d["external"][i] + (1-args["alpha"])*previous
            filter_residual.append(prediction-d["filtered"][i])
        if not row.get("admittance_updated"):
            continue
        before, after = row["admittance_before"], row["admittance_integrated"]
        x, v = np.array(before["offset"]), np.array(before["velocity"])
        dt, force = row["dt_s"], np.array(row["admittance_input_wrench"])
        v1 = np.clip(v+dt*(force-damping*v-stiffness*x)/mass, -limits, limits)
        dynamics.append(np.r_[x+dt*v1-np.array(after["offset"]), v1-np.array(after["velocity"])])
        application.append(d["reference"][i]-d["nominal"][i]-R[i] @ np.array(after["offset"][:3]))
    edge = np.diff(np.r_[False, d["active"], False].astype(int))
    starts, ends = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)-1
    command_error = (d["reference"]-d["command"])[d["active"]]*1000
    timestamps = {}
    for label, a, b in [
        ("feedback_read", "feedback_read_start_monotonic_ns", "feedback_received_monotonic_ns"),
        ("force_processing", "wrench_start_monotonic_ns", "wrench_ready_monotonic_ns"),
        ("force_ready_to_reference", "wrench_ready_monotonic_ns", "reference_ready_monotonic_ns"),
        ("reference_to_servo", "reference_ready_monotonic_ns", "servo_ready_monotonic_ns"),
        ("tick", "tick_start_monotonic_ns", "tick_end_monotonic_ns")]:
        values = [(r[b]-r[a])/1e6 for r in rows if a in r and b in r]
        timestamps[label] = dict(zip(("median", "p95", "max"), np.percentile(values, [50, 95, 100]).tolist()))
    mismatches = [path for path, info in m["sources"].items()
                  if Path(path).exists() and hashlib.sha256(Path(path).read_bytes()).hexdigest() != info["sha256"]]
    return dict(rows=len(rows), duration_s=d["t"][-1], input_sha256=d["input_sha256"],
                input_bytes=d["input_bytes"], malformed_lines=d["malformed"], summary=d["summary"],
                incomplete_wrench_rows=[{"sequence": r["sequence"], "error": r.get("error")}
                                       for r in rows if "wrench_stages" not in r],
                active_windows_s=[[d["t"][a], d["t"][b]] for a, b in zip(starts, ends)],
                force_stop_times_s=[r["elapsed_s"] for r in rows if r.get("force_gate_tripped")],
                servo_mode=m.get("servo_pose_tracking_mode"), source_mismatches=mismatches,
                independent_fk_max_position_error_m=float(np.max(np.abs(p-pose(rows, "measured_tcp_pose")))),
                compensation_max_residual=float(np.nanmax(np.abs(predicted_comp-d["comp"]))),
                wrench_transform_max_residual=float(np.nanmax(np.abs(predicted_ext-d["external"]))),
                transported_filter_max_residual=float(np.max(np.abs(filter_residual))),
                deadband_max_residual=float(np.nanmax(np.abs(predicted_out-d["processed"]))),
                admittance_max_residual=float(np.max(np.abs(dynamics))),
                offset_application_max_residual_m=float(np.max(np.abs(application))),
                reference_to_command_rms_mm=float(np.sqrt(np.mean(np.sum(command_error**2, axis=1)))),
                timing_ms=timestamps)


def window(d, start, end):
    ix = np.flatnonzero((d["t"] >= start) & (d["t"] < end) & d["active"])
    i, j = ix[[0, -1]]
    t, R, offset = d["t"][ix], d["R"][ix], d["offset"][ix]
    ref, nom, cmd, pos = [d[k][ix]*1000 for k in ("reference", "nominal", "command", "p")]
    filtered, processed = d["filtered"][ix], d["processed"][ix]
    angular = np.array([rotation_vector_from_matrix(r @ R[0].T) for r in R])
    relative = t-t[0]
    line = np.c_[np.ones(len(t)), relative]
    harmonic = np.c_[line, np.sin(2*np.pi*relative), np.cos(2*np.pi*relative)]
    peak_to_peak = {}
    for key, values in (("actual_xyz_mm", pos), ("reference_xyz_mm", ref), ("command_xyz_mm", cmd),
                        ("actual_rotation_deg", np.rad2deg(angular))):
        fit = np.linalg.lstsq(harmonic, values, rcond=None)[0]
        peak_to_peak[key] = (2*np.hypot(fit[-2], fit[-1])).tolist()
    # Exact difference: d(R*x) = R_mid*dx + dR*x_mid.
    increments = np.diff(offset[:, :3], axis=0)
    rmid, xmid = (R[1:]+R[:-1])/2, (offset[1:, :3]+offset[:-1, :3])/2
    offset_motion = np.einsum("nij,nj->ni", rmid, increments)*1000
    rotation_motion = np.einsum("nij,nj->ni", np.diff(R, axis=0), xmid)*1000
    accumulated_rot_motion = np.cumsum(np.vstack([np.zeros(3), rotation_motion]), axis=0)
    residual_rot_motion = accumulated_rot_motion-line @ np.linalg.lstsq(line, accumulated_rot_motion, rcond=None)[0]
    direct = d["comp"][ix, 3:] @ np.array(d["m"]["sensor_to_tool_rotation"]).T
    lever = np.cross(np.array(d["m"]["tool_to_sensor_m"]), d["external"][ix, :3])
    lag = []
    if len(ix) >= 500:
        command_q = np.array([d["rows"][k]["final_command_q_rad"] for k in ix])
        measured_q = np.array([d["rows"][k]["measured_q_rad"] for k in ix])
        command_q -= line @ np.linalg.lstsq(line, command_q, rcond=None)[0]
        measured_q -= line @ np.linalg.lstsq(line, measured_q, rcond=None)[0]
        for axis in range(6):
            if np.std(command_q[:, axis]) < 1e-5:
                lag.append(None)
                continue
            correlations = [np.corrcoef(command_q[:-k, axis] if k else command_q[:, axis],
                                        measured_q[k:, axis])[0, 1] for k in range(50)]
            k = int(np.nanargmax(correlations))
            lag.append(dict(lag_ms=float(k*np.median(np.diff(t))*1000),
                            correlation=float(correlations[k])))
    return dict(window_s=[float(t[0]), float(t[-1])], rows=len(ix),
                actual_change_mm=(pos[-1]-pos[0]).tolist(), nominal_change_mm=(nom[-1]-nom[0]).tolist(),
                compliance_change_mm=((ref-nom)[-1]-(ref-nom)[0]).tolist(),
                reference_change_mm=(ref[-1]-ref[0]).tolist(),
                command_change_mm=(cmd[-1]-cmd[0]).tolist(),
                vertical_compliance_change_by_tool_xyz_mm=((R[-1, 2]*offset[-1, :3]-R[0, 2]*offset[0, :3])*1000).tolist(),
                filtered_wrench_mean=filtered.mean(0).tolist(), processed_wrench_mean=processed.mean(0).tolist(),
                filtered_force_peak_abs_n=np.max(np.abs(filtered[:, :3]), axis=0).tolist(),
                force_nonzero_percent=((processed[:, :3] != 0).mean(0)*100).tolist(),
                max_offset_translation_mm=(np.max(np.abs(offset[:, :3]), axis=0)*1000).tolist(),
                max_offset_rotation_deg=np.max(np.abs(np.rad2deg(offset[:, 3:])), axis=0).tolist(),
                reference_to_command_rms_mm=float(np.sqrt(np.mean(np.sum((ref-cmd)**2, axis=1)))),
                left_stick_vertical_peak_abs=float(max(abs(d["rows"][k]["joy"]["sample"]["axes"][1]) for k in ix)),
                angular_jog_peak_rad_s=float(max(np.linalg.norm(d["rows"][k]["jog"]["angular_rad_s"]) for k in ix)),
                mean_tool_axes_in_base=R.mean(0).tolist(),
                one_hz_harmonic_peak_to_peak=peak_to_peak,
                effective_command_to_measured_lag=lag,
                rotation_of_existing_offset_detrended_span_mm=np.ptp(residual_rot_motion, axis=0).tolist(),
                offset_update_accumulated_mm=offset_motion.sum(0).tolist(),
                rotation_of_offset_accumulated_mm=rotation_motion.sum(0).tolist(),
                direct_torque_std_nm=direct.std(0).tolist(), lever_torque_std_nm=lever.std(0).tolist())


def geometry(d):
    p, R, _ = fk(d["xml"], np.zeros((1, 6)))
    ps, Rs, _ = fk(d["xml"], np.zeros((1, 6)), end="jk_se_vi_200_link")
    model_rts, model_lever = R[0].T @ Rs[0], R[0].T @ (ps[0]-p[0])
    logged_rts = np.array(d["m"]["sensor_to_tool_rotation"])
    delta = logged_rts @ model_rts.T
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(ROOT / "simulation/mujoco/models/jaka/tracer_jaka_zu5_robot.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tcp, sensor = data.body("tool0"), data.site("tcp_fts_site")
    rt = tcp.xmat.reshape(3, 3)
    return dict(urdf_sensor_link="jk_se_vi_200_link", xml_sensor_site="tcp_fts_site",
                urdf_sensor_to_tool_rotation=model_rts.tolist(),
                urdf_tool_to_sensor_m=model_lever.tolist(),
                xml_sensor_to_tool_rotation=(rt.T @ sensor.xmat.reshape(3, 3)).tolist(),
                xml_tool_to_sensor_m=(rt.T @ (sensor.xpos-tcp.xpos)).tolist(),
                logged_sensor_to_tool_rotation=logged_rts.tolist(),
                logged_tool_to_sensor_m=d["m"]["tool_to_sensor_m"],
                rotation_difference_axis_angle_deg=np.rad2deg(rotation_vector_from_matrix(delta)).tolist(),
                model_pure_axial_minus5N_in_logged_tool_axes=(logged_rts @ model_rts.T @ [0., 0., -5.]).tolist(),
                user_measured_sensor_surface_to_disk_edge_m=.195,
                measurement_caveat="Surface is not necessarily sensor measurement origin; lateral offset/axes still require verification.")


def plots(d3, d5):
    ix = (d3["t"] >= 11.7) & (d3["t"] < 13.1)
    t = d3["t"][ix]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    for key, label in (("nominal", "nominal"), ("reference", "corrected reference"),
                       ("command", "final command"), ("p", "measured")):
        z = d3[key][ix, 2]*1000
        axes[0].plot(t, z-z[0], label=label)
    axes[0].set_ylabel("base Z change (mm)")
    axes[0].set_title("003: upward reference already exists before Servo; no up-stick / rotation command")
    for axis, label in ((0, "tool X: lateral"), (1, "tool Y: approximately up")):
        axes[axis+1].plot(t, d3["filtered"][ix, axis], label=label+" filtered")
        axes[axis+1].plot(t, d3["processed"][ix, axis], label="after hard deadband", linestyle="--")
        axes[axis+1].axhline(1, color="gray", linestyle=":")
        axes[axis+1].axhline(-1, color="gray", linestyle=":")
        axes[axis+1].set_ylabel("N")
    for ax in axes:
        ax.legend(); ax.grid(alpha=.25)
    axes[-1].set_xlabel("elapsed s")
    fig.savefig(OUT / "trial003_upward_source.png", dpi=150)
    plt.close(fig)
    ix = (d5["t"] >= 70.85) & (d5["t"] < 111.07)
    t, R = d5["t"][ix], d5["R"][ix]
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True, constrained_layout=True)
    for axis in range(3):
        origin = d5["p"][ix][0, axis]
        for key, label in (("nominal", "nominal"), ("reference", "corrected reference"),
                           ("command", "final command"), ("p", "measured")):
            axes[axis].plot(t, (d5[key][ix, axis]-origin)*1000, label=label, lw=1)
        axes[axis].set_ylabel(f"base {'XYZ'[axis]} (mm)")
    axes[0].set_title("005 insertion: large nominal / compliance cancellation; references are not converged")
    for axis, label in enumerate(("tool Fx", "tool Fy", "tool Fz")):
        axes[3].plot(t, d5["processed"][ix, axis], label=label, lw=.8)
    axes[3].set_ylabel("processed N")
    angles = np.array([rotation_vector_from_matrix(r @ R[0].T) for r in R])*180/np.pi
    for axis, label in enumerate(("base Rx", "base Ry", "base Rz")):
        axes[4].plot(t, angles[:, axis], label=label)
    axes[4].set_ylabel("measured deg")
    axes[4].set_xlabel("elapsed s")
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(loc="upper left", ncol=4)
    fig.savefig(OUT / "trial005_insertion_pipeline.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    data = {trial: load(trial) for trial in (3, 4, 5)}
    results = {"scope": "Offline recorded-data audit; measured contact location and sensor physical axes are not identified by this log.",
               "sessions": {str(k): audit(d) for k, d in data.items()}, "geometry": geometry(data[5]),
               "windows": {"003_upward": window(data[3], 11.7, 13.1),
                           "003_late_contact": window(data[3], 55, 62.7),
                           "004_downward_input_cancelled": window(data[4], 20, 27),
                           "005_insertion": window(data[5], 70.856, 111.064),
                           "005_no_rotation_input": window(data[5], 94, 110)}}
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plots(data[3], data[5])
    print(json.dumps(results, indent=2))
