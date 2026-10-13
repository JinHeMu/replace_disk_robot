"""Offline audit of all-row payload identification; never opens a robot client."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "tool"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "logs/force_direction_audit"))
from identify_ft_payload import load_all_rows, identify_payload
from reproduce_audit import fk
from replace_disk_robot.applications.common.wrench import default_tool0_ft_transforms, validate_tool0_ft_extrinsics


def short_fit(f):
    return {k: f[k] for k in ("mass_kg", "gravity_tilt_deg", "unconstrained_gravity_tilt_deg",
                              "center_of_mass_sensor_mm", "force_bias_sensor_n", "torque_bias_sensor_nm")} | {
        "fit": {k: v for k, v in f["fit"].items() if not k.endswith("weights")}}


def residual(f, poses):
    R = np.array([p.rotation_base_sensor for p in poses])
    fg = np.einsum("nji,j->ni", R, f["signed_gravity_force_base_n"])
    force = np.array([p.force_sensor_n for p in poses]) - f["force_bias_sensor_n"] - fg
    torque = (np.array([p.torque_sensor_nm for p in poses]) - f["torque_bias_sensor_nm"]
              - np.cross(f["center_of_mass_sensor_m"], fg))
    return force, torque


def stats(x):
    return dict(component_rms=np.sqrt(np.mean(x*x, axis=0)).tolist(),
                all_component_rms=float(np.sqrt(np.mean(x*x))),
                norm_p95=float(np.percentile(np.linalg.norm(x, axis=1), 95)),
                norm_max=float(np.max(np.linalg.norm(x, axis=1))))


def main():
    for original, name in (("tool/ft_gravity_samples.csv", "input_samples.csv"),
                           ("tool/ft_gravity_samples_identified.json", "identified_all_rows.json")):
        if not (OUT / name).exists():
            shutil.copyfile(ROOT / original, OUT / name)
    path = OUT / "input_samples.csv"
    rows = list(csv.DictReader(path.open()))
    poses, tr = load_all_rows(path, 50)
    fit = json.loads((OUT / "identified_all_rows.json").read_text())
    t = np.array([float(r["time_s"]) for r in rows])
    q = np.array([[float(r[f"q{i}_rad"]) for i in range(1, 7)] for r in rows])
    qd = np.array([[float(r[f"qd{i}_rad_s"]) for i in range(1, 7)] for r in rows])
    speed = np.max(np.abs(qd), axis=1)
    commanded = np.array([int(r["command_active"]) for r in rows])
    validate_tool0_ft_extrinsics(np.array(fit["sensor_to_tool_rotation"]), np.array(fit["tool_to_sensor_m"]))
    urdf = (ROOT / "simulation/mujoco/models/jaka/urdf/tracer_jaka_zu5.urdf").read_text()
    _, R, _ = fk(urdf, q)
    _, Rs, _ = fk(urdf, q, end="jk_se_vi_200_link")
    recorded = np.array([p.rotation_base_sensor for p in poses])
    default_R, lever = default_tool0_ft_transforms()
    force, torque = residual(fit, poses)
    force_tcp = force @ default_R.T
    torque_tcp = torque @ default_R.T + np.cross(lever, force_tcp)
    stopped = (speed <= .002) & (commanded == 0)
    edges = np.diff(np.r_[False, stopped, False].astype(int))
    windows = [[float(t[a]), float(t[b])] for a, b in zip(np.where(edges == 1)[0],
               np.where(edges == -1)[0]-1) if t[b]-t[a] >= .8]
    metrics = dict(csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=len(rows),
        duration_s=float(t[-1]-t[0]), capture_ids=dict(collections.Counter(r["capture_id"] for r in rows)),
        phases=dict(collections.Counter(r["capture_phase"] for r in rows)), command_active_rows=int(commanded.sum()),
        joint_speed_percentiles_rad_s=dict(zip(("median", "p95", "max"), np.percentile(speed, [50, 95, 100]).tolist())),
        low_speed_uncommanded_windows_s=windows,
        recorded_rotation_vs_urdf_max_abs=float(np.max(np.abs(recorded-Rs))),
        rotation_default_vs_urdf_max_abs=float(np.max(np.abs(R @ default_R-Rs))),
        reproduction=short_fit(identify_payload(poses)), current_fit=short_fit(fit),
        residual_sensor_force_n=stats(force), residual_sensor_torque_nm=stats(torque),
        residual_tcp_force_n=stats(force_tcp), residual_tcp_torque_nm=stats(torque_tcp))
    folds = []
    for ix in np.array_split(np.arange(len(poses)), 5):
        held = set(ix.tolist())
        f = identify_payload([p for k, p in enumerate(poses) if k not in held])
        fr, tor = residual(f, [poses[k] for k in ix])
        folds.append(dict(withheld_time_s=t[ix[[0, -1]]].tolist(), training_fit=short_fit(f),
                          withheld_force_n=stats(fr), withheld_torque_nm=stats(tor)))
    metrics["five_contiguous_block_validation"] = folds
    metrics["tilt_bound_sensitivity"] = {str(lim): short_fit(identify_payload(poses, gravity_tilt_limit_deg=lim))
                                        for lim in (0, 2, 4, 6)}
    metrics["free_fit"] = short_fit(identify_payload(poses, gravity_mode="free"))
    metrics["half_fits"] = {name: short_fit(identify_payload(pp)) for name, pp in
                            (("first_half", poses[:510]), ("second_half", poses[510:]))}
    metrics["low_speed_fit"] = short_fit(identify_payload([p for p, v in zip(poses, speed) if v <= .002]))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for i, axis in enumerate("XYZ"):
        axes[0].plot(t, force_tcp[:, i], label=axis, linewidth=.8)
        axes[1].plot(t, torque_tcp[:, i], label=axis, linewidth=.8)
    axes[0].set_ylabel("TCP force residual (N)")
    axes[1].set_ylabel("TCP torque residual (Nm)")
    axes[2].plot(t, speed, label="max absolute joint speed")
    axes[2].fill_between(t, 0, speed.max(), where=commanded.astype(bool), alpha=.12, label="command_active")
    axes[2].set_ylabel("joint speed (rad/s)")
    axes[2].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", ncol=3)
    axes[0].set_title("1021-row gravity fit: in-sample residuals, not independent validation")
    fig.savefig(OUT / "residuals.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"rows": len(rows), "mass_kg": fit["mass_kg"],
                      "frame_residual": metrics["recorded_rotation_vs_urdf_max_abs"],
                      "tcp_torque_peak_nm": metrics["residual_tcp_torque_nm"]["norm_max"]}))


if __name__ == "__main__":
    main()
