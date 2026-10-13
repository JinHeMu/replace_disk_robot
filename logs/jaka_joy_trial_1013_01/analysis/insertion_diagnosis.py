"""Offline insertion cancellation and rotation audit, using recorded references.

No robot client is constructed. Pipeline/FK checks reuse the previous audit.
"""
from pathlib import Path
import collections
import hashlib
import importlib.util
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SESSION = OUT.parent
sys.path.insert(0, str(ROOT / "src"))
from replace_disk_robot.core.rotation import rotation_matrix, rotation_vector_from_matrix

spec = importlib.util.spec_from_file_location(
    "previous_audit", ROOT / "logs/jaka_joy_comparison_003_005/diagnose.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def load():
    metadata = json.loads((SESSION / "metadata.json").read_text())
    raw = (SESSION / "samples.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    xml = helpers.recorded_urdf(metadata)["content"]
    p, R, _ = helpers.fk(xml, np.array([r["measured_q_rad"] for r in rows]))
    nominal = np.array([r["admittance_integrated"]["nominal_pose"]["position_m"]
                        if r.get("admittance_updated") else [np.nan]*3 for r in rows])
    return dict(rows=rows, m=metadata, xml=xml, p=p, R=R, nominal=nominal,
                active=np.array([r.get("admittance_updated", False) and not r.get("fault") for r in rows]),
                reference=helpers.pose(rows, "corrected_pose"),
                command=helpers.pose(rows, "final_command_tcp_pose"),
                t=np.array([r["elapsed_s"] for r in rows]),
                offset=np.array([r["admittance_after"]["offset"] for r in rows]),
                raw=np.array([r["raw_sensor_wrench"] for r in rows]),
                comp=helpers.wrench_series(rows, "compensated_sensor"),
                external=helpers.wrench_series(rows, "external_tcp_unfiltered"),
                filtered=helpers.wrench_series(rows, "filtered_before_deadband"),
                processed=helpers.wrench_series(rows, "processed"),
                input_bytes=len(raw), input_sha256=hashlib.sha256(raw).hexdigest(), malformed=0,
                summary=json.loads((SESSION / "summary.json").read_text()))


def cancellation(d, start, end):
    ix = np.flatnonzero((d["t"] >= start) & (d["t"] < end) & d["active"])
    rr = [d["rows"][k] for k in ix]
    t, R = d["t"][ix], d["R"][ix]
    nominal_R = rotation_matrix(rr[0]["admittance_integrated"]["nominal_pose"]["quaternion_wxyz"])
    direction = nominal_R[:, 2]
    x = np.array([r["admittance_integrated"]["offset"] for r in rr])
    v = np.array([r["admittance_integrated"]["velocity"] for r in rr])
    jog = np.array([r["jog"]["linear_m_s"] for r in rr])
    dt = np.array([r["dt_s"] for r in rr])
    duration = t[-1]-t[0]
    nominal, ref, command, actual = [d[k][ix] for k in ("nominal", "reference", "command", "p")]
    changes = {"nominal": nominal[-1]-nominal[0], "compliance": (ref-nominal)[-1]-(ref-nominal)[0],
               "corrected_reference": ref[-1]-ref[0], "final_command": command[-1]-command[0],
               "actual": actual[-1]-actual[0]}
    rmid, xmid = (R[1:]+R[:-1])/2, (x[1:, :3]+x[:-1, :3])/2
    dx = np.einsum("nij,nj->ni", rmid, np.diff(x[:, :3], axis=0))
    dr = np.einsum("nij,nj->ni", np.diff(R, axis=0), xmid)
    result = dict(window_s=[float(t[0]), float(t[-1])], rows=len(ix),
                  insertion_direction_base=direction.tolist(),
                  axial_displacement_mm={k: float(value @ direction*1000) for k, value in changes.items()},
                  axial_average_speed_mm_s={k: float(value @ direction/duration*1000) for k, value in changes.items()},
                  forward_jog_average_mm_s=float(np.average(jog @ direction, weights=dt)*1000),
                  admittance_tool_z_average_mm_s=float(np.average(v[:, 2], weights=dt)*1000),
                  axial_velocity_clipped_percent=float(np.mean([r["velocity_clipped_axes"][2] for r in rr])*100),
                  axial_offset_start_end_mm=(x[[0, -1], 2]*1000).tolist(),
                  force_tool_z_mean_n=float(np.mean(d["processed"][ix, 2])),
                  offset_update_axial_average_mm_s=float(dx.sum(0) @ direction/duration*1000),
                  rotation_of_existing_offset_axial_average_mm_s=float(dr.sum(0) @ direction/duration*1000),
                  guard_trip_count=sum(bool(r.get("force_gate_tripped")) for r in rr),
                  status_counts=dict(collections.Counter(r["status"] for r in rr)))
    result["cancellation_percent"] = float(-result["axial_average_speed_mm_s"]["compliance"]
                                           / result["axial_average_speed_mm_s"]["nominal"]*100)
    return result


def twist(d, start, end):
    result = helpers.window(d, start, end)
    ix = np.flatnonzero((d["t"] >= start) & (d["t"] < end) & d["active"])
    t, R = d["t"][ix], d["R"][ix]
    relative = t-t[0]
    line = np.c_[np.ones(len(t)), relative]
    actual_angles = np.array([rotation_vector_from_matrix(R[0].T @ r) for r in R])
    pos = d["p"][ix]
    detrended = pos-line @ np.linalg.lstsq(line, pos, rcond=None)[0]
    frequency = np.fft.rfftfreq(len(t), np.median(np.diff(t)))
    spectrum = np.abs(np.fft.rfft(detrended*np.hanning(len(t))[:, None], axis=0))
    band = (frequency >= .4) & (frequency <= 5.)
    peak = float(frequency[band][np.argmax(spectrum[band, 0])])
    fft_peak = peak
    resolution = 1/(t[-1]-t[0])
    candidates = np.linspace(max(.4, peak-resolution), min(5., peak+resolution), 401)
    errors = []
    for candidate in candidates:
        model = np.c_[line, np.sin(2*np.pi*candidate*relative), np.cos(2*np.pi*candidate*relative)]
        errors.append(np.mean((pos[:, 0]-model @ np.linalg.lstsq(model, pos[:, 0], rcond=None)[0])**2))
    peak = float(candidates[np.argmin(errors)])
    basis = np.c_[line, np.sin(2*np.pi*peak*relative), np.cos(2*np.pi*peak*relative)]
    signals = {}
    angle_errors = []
    for key in ("measured_tcp_pose", "corrected_pose", "final_command_tcp_pose"):
        rr = [d["rows"][k][key] for k in ix]
        rotations = np.array([rotation_matrix(r["quaternion_wxyz"]) for r in rr])
        angles = np.array([rotation_vector_from_matrix(R[0].T @ rot) for rot in rotations])
        values = np.c_[np.array([r["position_m"] for r in rr])*1000, np.rad2deg(angles)]
        fit = np.linalg.lstsq(basis, values, rcond=None)[0]
        signals[key] = dict(harmonic_peak_to_peak_xyz_mm_rotation_deg=(2*np.hypot(fit[-2], fit[-1])).tolist())
        if key == "corrected_pose":
            ref_rotations = rotations
        if key == "final_command_tcp_pose":
            angle_errors = [np.linalg.norm(rotation_vector_from_matrix(a @ b.T))
                            for a, b in zip(ref_rotations, rotations)]
    moments = d["processed"][ix, 3:]
    result.update(dominant_lateral_frequency_hz=peak, fft_peak_frequency_hz=fft_peak,
                  harmonic_signals=signals,
                  actual_rotation_span_initial_tool_deg=np.ptp(np.rad2deg(actual_angles), axis=0).tolist(),
                  reference_to_command_rotation_rms_deg=float(np.rad2deg(np.sqrt(np.mean(np.square(angle_errors))))),
                  processed_torque_mean_nm=moments.mean(0).tolist(),
                  processed_torque_peak_abs_nm=np.max(np.abs(moments), axis=0).tolist(),
                  processed_torque_nonzero_percent=((moments != 0).mean(0)*100).tolist())
    return result


def plots(d):
    ix = np.flatnonzero((d["t"] >= 34) & (d["t"] < 55.4) & d["active"])
    t, R = d["t"][ix], d["R"][ix]
    direction = rotation_matrix(d["rows"][ix[0]]["admittance_integrated"]["nominal_pose"]["quaternion_wxyz"])[:, 2]
    origin = d["p"][ix[0]]
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
    for key, label in (("nominal", "nominal"), ("reference", "corrected reference"),
                       ("command", "final command"), ("p", "measured TCP")):
        axes[0].plot(t, (d[key][ix]-origin) @ direction*1000, label=label, lw=1)
    axes[0].set_title("1013_01: contact progressively cancels forward motion")
    axes[0].set_ylabel("axial travel (mm)")
    jog = np.array([d["rows"][k]["jog"]["linear_m_s"] for k in ix]) @ direction*1000
    v = np.array([d["rows"][k]["admittance_integrated"]["velocity"] for k in ix])[:, 2]*1000
    axes[1].plot(t, jog, label="forward jog")
    axes[1].plot(t, v, label="tool-Z admittance velocity")
    axes[1].plot(t, jog+v, label="sum (small frame-rotation terms omitted)")
    axes[1].set_ylabel("mm/s")
    axes[2].plot(t, d["processed"][ix, 2], label="processed tool Fz")
    x = np.array([d["rows"][k]["admittance_integrated"]["offset"][2] for k in ix])
    axes[2].plot(t, -.005*300+30*x, "--", label="Fz for steady vz=-5 mm/s: -1.5+30*x")
    axes[2].set_ylabel("N")
    axes[3].plot(t, x*1000, label="axial admittance offset")
    axes[3].set_ylabel("mm")
    axes[3].set_xlabel("elapsed s")
    for ax in axes:
        ax.axvspan(50, 54, color="gray", alpha=.12)
        ax.legend(loc="best"); ax.grid(alpha=.25)
    fig.savefig(OUT / "insertion_cancellation.png", dpi=150)
    plt.close(fig)
    ix = np.flatnonzero((d["t"] >= 58) & (d["t"] < 64.5) & d["active"])
    t, R = d["t"][ix], d["R"][ix]
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for key, label in (("reference", "corrected reference"), ("command", "final command"), ("p", "measured TCP")):
        axes[0].plot(t, (d[key][ix, 0]-d[key][ix, 0].mean())*1000, label=label)
    axes[0].set_ylabel("centered base X (mm)")
    for key, label in (("corrected_pose", "reference"), ("final_command_tcp_pose", "command"),
                       ("measured_tcp_pose", "actual")):
        angles = np.array([rotation_vector_from_matrix(R[0].T @ rotation_matrix(d["rows"][k][key]["quaternion_wxyz"]))
                           for k in ix])*180/np.pi
        axes[1].plot(t, angles[:, 1], label=label+" initial tool Ry")
    axes[1].set_ylabel("deg")
    axes[1].set_title("No angular jog; torque drives the reference rotation")
    for j, label in enumerate(("Tx", "Ty", "Tz"), 3):
        axes[2].plot(t, d["processed"][ix, j], label=label)
    axes[2].set_ylabel("processed Nm")
    axes[3].plot(t, d["filtered"][ix, 4], label="filtered Ty before deadband")
    axes[3].plot(t, d["processed"][ix, 4], "--", label="Ty after hard deadband")
    axes[3].axhline(.2, color="gray", linestyle=":")
    axes[3].axhline(-.2, color="gray", linestyle=":")
    axes[3].set_ylabel("Nm")
    axes[3].set_xlabel("elapsed s")
    for ax in axes:
        ax.legend(loc="best"); ax.grid(alpha=.25)
    fig.savefig(OUT / "contact_twist.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    d = load()
    result = dict(scope="Recorded data / independent math; not a new contact simulation or hardware acceptance.",
                  audit=helpers.audit(d), geometry=helpers.geometry(d),
                  effective_args=d["m"]["effective_args"],
                  angular_jog_peak_rad_s=float(max(np.linalg.norm(r.get("jog", {}).get("angular_rad_s", [0]*3))
                                                 for r in d["rows"])),
                  cancellation={name: cancellation(d, a, b) for name, a, b in (
                      ("free_motion", 34, 37), ("first_contact", 38, 55.4),
                      ("near_stall", 50, 54), ("stall_after_resume", 59, 62),
                      ("later_insertion", 97.7, 100.4))},
                  twist={"first_contact": twist(d, 38, 55.4), "steady_contact": twist(d, 58, 64.5)})
    for key in ("compensation_max_residual", "wrench_transform_max_residual", "transported_filter_max_residual",
                "deadband_max_residual", "admittance_max_residual", "offset_application_max_residual_m"):
        assert result["audit"][key] < 1e-10, (key, result["audit"][key])
    (OUT / "insertion_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plots(d)
    print(json.dumps(result["cancellation"], indent=2))
