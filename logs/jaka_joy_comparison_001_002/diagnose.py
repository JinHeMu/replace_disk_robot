"""Reproduce the two-trial diagnosis without connecting to robot hardware."""
from pathlib import Path
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from replace_disk_robot.core.rotation import rotation_matrix, rotation_vector_from_matrix

OUT = Path(__file__).resolve().parent


def load(trial):
    directory = ROOT / "logs" / f"jaka_joy_trial_{trial:03d}"
    meta = json.loads((directory / "metadata.json").read_text())
    rows = [json.loads(line) for line in (directory / "samples.jsonl").open()]
    t = np.array([r["elapsed_s"] for r in rows])
    rotation = np.array([rotation_matrix(r["measured_tcp_pose"]["quaternion_wxyz"]) for r in rows])
    pos = np.array([r["measured_tcp_pose"]["position_m"] for r in rows]) * 1000
    angle = np.array([rotation_vector_from_matrix(r @ rotation[0].T) for r in rotation]) * 180 / np.pi
    target = np.full_like(pos, np.nan)
    nominal = np.array([r["admittance_after"]["nominal_pose"]["position_m"] for r in rows]) * 1000
    for i, r in enumerate(rows):
        if r.get("admittance_updated"):
            target[i] = np.array(r["corrected_pose"]["position_m"]) * 1000
    return dict(trial=trial, rows=rows, meta=meta, t=t, pos=pos, angle=angle,
                rotation=rotation, target=target, nominal=nominal,
                wrench=np.array([r["wrench_stages"]["processed"] for r in rows]),
                offset=np.array([r["admittance_after"]["offset"] for r in rows]),
                jog=np.array([r["jog"]["linear_m_s"] for r in rows]) * 1000,
                angular_jog=np.array([r["jog"]["angular_rad_s"] for r in rows]) * 180 / np.pi)


def window_metrics(d, start, end):
    ix = np.flatnonzero((d["t"] >= start) & (d["t"] < end))
    rows = [d["rows"][i] for i in ix]
    t = d["t"][ix]
    rel = t - t[0]
    pos, angle = d["pos"][ix], d["angle"][ix]
    data = np.c_[pos, angle]
    line = np.c_[rel, np.ones(len(t))]
    residual = data - line @ np.linalg.lstsq(line, data, rcond=None)[0]
    frequency = np.fft.rfftfreq(len(t), np.median(np.diff(t)))
    spectrum = np.abs(np.fft.rfft(residual * np.hanning(len(t))[:, None], axis=0))
    band = (frequency >= .7) & (frequency <= 1.5)
    peak = frequency[band][np.argmax(spectrum[band, 0])]
    basis = np.c_[line, np.sin(2*np.pi*peak*rel), np.cos(2*np.pi*peak*rel)]
    fit = np.linalg.lstsq(basis, data, rcond=None)[0]
    amplitude_pp = 2 * np.hypot(fit[-2], fit[-1])
    one_hz_basis = np.c_[line, np.sin(2*np.pi*rel), np.cos(2*np.pi*rel)]
    actual_fit = np.linalg.lstsq(one_hz_basis, pos, rcond=None)[0]
    reference_fit = np.linalg.lstsq(one_hz_basis, d["target"][ix], rcond=None)[0]
    transfer = ((actual_fit[-2] + 1j*actual_fit[-1]) /
                (reference_fit[-2] + 1j*reference_fit[-1]))
    command = np.array([r["final_command_q_rad"] for r in rows])
    measured = np.array([r["measured_q_rad"] for r in rows])
    lag = []
    for j in range(6):
        c = command[:, j] - line @ np.linalg.lstsq(line, command[:, j], rcond=None)[0]
        m = measured[:, j] - line @ np.linalg.lstsq(line, measured[:, j], rcond=None)[0]
        correlations = [np.corrcoef(c[:-k] if k else c, m[k:])[0, 1] for k in range(64)]
        best = int(np.argmax(correlations))
        lag.append(dict(lag_ms=best*np.median(np.diff(t))*1000, correlation=correlations[best]))
    sensor = np.array([r["wrench_stages"]["compensated_sensor"] for r in rows])
    sensor_rotation = np.array(d["meta"]["sensor_to_tool_rotation"])
    direct_moment = sensor[:, 3:] @ sensor_rotation.T
    shifted_moment = np.cross(np.array(d["meta"]["tool_to_sensor_m"]), sensor[:, :3] @ sensor_rotation.T)
    external = np.array([r["wrench_stages"]["external_tcp_unfiltered"] for r in rows])
    return dict(window_s=[start, end], xyz_change_mm=(pos[-1]-pos[0]).tolist(),
                xyz_span_mm=np.ptp(pos, axis=0).tolist(),
                rotation_span_deg=np.ptp(angle, axis=0).tolist(),
                dominant_lateral_frequency_hz=float(peak),
                harmonic_xyz_peak_to_peak_mm=amplitude_pp[:3].tolist(),
                harmonic_rotation_peak_to_peak_deg=amplitude_pp[3:].tolist(),
                one_hz_measured_to_corrected_xyz_amplitude_ratio=np.abs(transfer).tolist(),
                one_hz_measured_relative_to_corrected_xyz_phase_deg=np.rad2deg(np.angle(transfer)).tolist(),
                effective_command_to_feedback_lag=lag,
                max_abs_angular_jog_deg_s=np.max(np.abs(d["angular_jog"][ix]), axis=0).tolist(),
                clipped_axes_percent=(np.mean([r["velocity_clipped_axes"] for r in rows], axis=0)*100).tolist(),
                direct_moment_std_nm=direct_moment.std(0).tolist(),
                lever_moment_std_nm=shifted_moment.std(0).tolist(),
                moment_transform_max_residual_nm=float(np.abs(external[:, 3:]-direct_moment-shifted_moment).max()))


def overview(datasets):
    fig, axes = plt.subplots(5, 2, figsize=(15, 14), sharex="col", constrained_layout=True)
    for column, d in enumerate(datasets):
        t = d["t"]
        for j, label in enumerate("xyz"):
            axes[0, column].plot(t, d["pos"][:, j]-d["pos"][0, j], label=f"measured base {label}", lw=1)
        axes[0, column].plot(t, d["nominal"][:, 1]-d["pos"][0, 1], "--", color="C1", label="nominal base y", lw=1)
        axes[0, column].set_title(f"Trial {d['trial']:03d}: measured / nominal displacement")
        axes[0, column].set_ylabel("mm")
        for j, label in enumerate("xyz"):
            axes[1, column].plot(t, d["jog"][:, j], label=f"jog base {label}", lw=1)
            axes[2, column].plot(t, d["angle"][:, j], label=f"measured base r{label}", lw=1)
        axes[1, column].set_ylabel("Input mm/s")
        axes[2, column].set_ylabel("Rotation deg")
        axes[2, column].set_title("Measured rotation; angular jog = 0 throughout" if d["trial"] == 1
                                  else "Measured rotation; angular jog = 0 from 8.816 s")
        for j, label in enumerate(["Fx", "Fy", "Fz"]):
            axes[3, column].plot(t, d["wrench"][:, j], label=f"tool {label}", lw=.8)
        axes[3, column].set_ylabel("Processed N")
        for j, label in enumerate(["Tx", "Ty", "Tz"]):
            axes[4, column].plot(t, d["wrench"][:, j+3], label=f"tool {label}", lw=.8)
        axes[4, column].set_ylabel("Processed N m")
        axes[4, column].set_xlabel("Elapsed seconds")
        for ax in axes[:, column]:
            ax.grid(alpha=.25)
            ax.legend(loc="upper left", fontsize=8, ncol=2)
            if d["trial"] == 1:
                ax.axvline(22.960, color="red", linestyle="--", lw=1)
                ax.axvspan(22.960, t[-1], color="gray", alpha=.12)
            else:
                ax.axvline(13.336, color="gray", linestyle=":", lw=1)
                ax.axvline(39.968, color="red", linestyle="--", lw=1)
        axes[-1, column].set_xlim(0, d["t"][-1])
    fig.suptitle("Horizontal rectangular slot: first force stop / contact onset / LB release marked; gray trial 001 tail excluded", fontsize=12)
    fig.savefig(OUT / "comparison_overview.png", dpi=150)
    plt.close(fig)


def focus(datasets):
    fig, axes = plt.subplots(4, 2, figsize=(15, 11), sharex="col", constrained_layout=True)
    for column, (d, start, end) in enumerate([(datasets[0], 17.5, 22.5), (datasets[1], 34, 39)]):
        ix = (d["t"] >= start) & (d["t"] < end)
        t = d["t"][ix]
        for j, label in [(0, "x"), (2, "z")]:
            axes[0, column].plot(t, d["pos"][ix, j]-d["pos"][ix, j].mean(), label=f"measured {label}")
            axes[0, column].plot(t, d["target"][ix, j]-np.nanmean(d["target"][ix, j]), "--", label=f"corrected {label}", lw=1)
        for j, label in [(0, "rx"), (2, "rz")]:
            axes[1, column].plot(t, d["angle"][ix, j]-d["angle"][ix, j].mean(), label=label)
        for j, label in [(0, "Fx"), (1, "Fy")]:
            axes[2, column].plot(t, d["wrench"][ix, j], label=f"tool {label}")
        for j, label in [(3, "Tx"), (4, "Ty")]:
            axes[3, column].plot(t, d["wrench"][ix, j], label=f"tool {label}")
        axes[0, column].set_title(f"Trial {d['trial']:03d}: {start}-{end} s; angular jog = 0")
        axes[0, column].set_ylabel("Centered base mm")
        axes[1, column].set_ylabel("Centered base deg")
        axes[2, column].set_ylabel("N")
        axes[3, column].set_ylabel("N m")
        axes[3, column].set_xlabel("Elapsed seconds")
        for ax in axes[:, column]:
            ax.grid(alpha=.3)
            ax.legend(loc="upper right", fontsize=8, ncol=2)
        axes[3, column].set_xlim(start, end)
    fig.suptitle("Approximately 1 Hz coupled contact oscillation, with no commanded rotation", fontsize=13)
    fig.savefig(OUT / "contact_oscillation_zoom.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    datasets = [load(1), load(2)]
    results = {"scope": "Offline observation; lag is cross-correlation, not measured network or hardware latency.",
               "trial_001_17_5_to_22_5": window_metrics(datasets[0], 17.5, 22.5),
               "trial_002_20_to_30": window_metrics(datasets[1], 20, 30),
               "trial_002_34_to_39": window_metrics(datasets[1], 34, 39)}
    (OUT / "diagnostic_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    overview(datasets)
    focus(datasets)
    print(OUT)
