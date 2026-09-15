#!/usr/bin/env python3
"""Run deterministic virtual-force experiments on the JAKA low pose.

This entry point injects virtual external wrenches through the same F/T data
path used by ``keyboard_servo.py --admittance``.  The admittance controller
then drives the pose-tracking servo and the real MuJoCo dynamics.  It writes
step-response metrics, plots and an optional render GIF for visual inspection.

Examples::

    python examples/jaka_admittance_experiment.py
    python examples/jaka_admittance_experiment.py --no-render
    python examples/jaka_admittance_experiment.py --output-dir /tmp/jaka_adm

The injected force is a *virtual* external wrench: it is not added as a
MuJoCo actuator force.  It enters at the sensor boundary exactly like a real
force/torque reading, so it exercises the full force-processing and admittance
chain while keeping the experiment deterministic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# The model must see the isolated conda environment before importing MuJoCo.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from keyboard_servo import ServoDemo  # noqa: E402
from replace_disk_robot.core import Wrench  # noqa: E402
from replace_disk_robot.core.rotation import rotation_matrix  # noqa: E402


DIRECTIONS = {
    "+X": np.array([1.0, 0.0, 0.0]),
    "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]),
    "-Y": np.array([0.0, -1.0, 0.0]),
    "+Z": np.array([0.0, 0.0, 1.0]),
    "-Z": np.array([0.0, 0.0, -1.0]),
}
FORCE_MAGNITUDES = (1.0, 2.5, 5.0, 7.5, 10.0)


def _raw_wrench_for_external_force(app: ServoDemo, force_base: np.ndarray) -> Wrench:
    """Return the sensor-frame reading that produces ``force_base``."""
    sensor_pose = app._sensor_pose_in_base()
    rotation = rotation_matrix(sensor_pose.quaternion_wxyz)
    raw_force = rotation.T @ (np.asarray(force_base, dtype=float) / app.wrench_load_sign)
    return Wrench(app.ft.frame_id, raw_force, np.zeros(3))


def _make_camera(app: ServoDemo):
    import mujoco

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    base = app.data.body("jaka_base_link")
    camera.lookat[:] = np.asarray(base.xpos) + [0.0, 0.0, 0.35]
    camera.distance = 1.8
    camera.azimuth, camera.elevation = 135.0, -25.0
    return camera


def _run_case(
    direction_name: str,
    force_base: np.ndarray,
    load_seconds: float,
    release_seconds: float,
    render: bool,
    render_stride: int,
    snapshot: bool = False,
):
    """Run one virtual-force step and return measured time histories."""
    import mujoco

    app = ServoDemo(
        model_name="jaka",
        keyframe="low",
        admittance=True,
        admittance_axes="translation",
        admittance_max_offset=0.02,
        force_filter_alpha=0.2,
        force_deadband_n=0.0,
    )
    app.wrench_processor.reset()
    dt = app.dt
    load_steps = round(load_seconds / dt)
    release_steps = round(release_seconds / dt)

    renderer = None
    camera = None
    frames = []
    snapshots = {}
    if render or snapshot:
        renderer = mujoco.Renderer(app.model, height=360, width=640)
        camera = _make_camera(app)

    def capture(label: str | None = None) -> None:
        if renderer is None:
            return
        renderer.update_scene(app.data, camera=camera)
        frame = renderer.render().copy()
        if label is None:
            frames.append(frame)
        else:
            snapshots[label] = frame

    initial_pose = app.kinematics.forward(app.robot.read_joint_state())
    initial_position = initial_pose.position_m.copy()
    initial_quaternion = initial_pose.quaternion_wxyz.copy()

    records = dict(
        t=[], actual=[], nominal=[], corrected=[], offset=[], external_force=[], status=[]
    )

    zero_wrench = Wrench(app.ft.frame_id, np.zeros(3), np.zeros(3))
    app.ft.read_wrench = lambda: zero_wrench

    # A short pre-roll lets the servo reach its hold target before the step.
    for _ in range(20):
        app.tick()
    if snapshot:
        capture("initial")

    loaded_wrench_fn = lambda: _raw_wrench_for_external_force(app, force_base)  # noqa: E731
    app.ft.read_wrench = loaded_wrench_fn

    for step in range(load_steps + release_steps):
        if step == load_steps:
            app.ft.read_wrench = lambda: zero_wrench
        app.tick()
        actual = app.kinematics.forward(app.robot.read_joint_state())
        state = app.motion.state()
        records["t"].append(step * dt)
        records["actual"].append(actual.position_m.copy())
        records["nominal"].append(app.motion.nominal_pose.position_m.copy())
        records["corrected"].append(app.motion.corrected_pose.position_m.copy())
        records["offset"].append(state.offset[:3].copy())
        records["external_force"].append(app.last_external_wrench.force_n.copy())
        records["status"].append(app.servo.status)
        if renderer is not None and step % render_stride == 0:
            capture()
        if snapshot and step == load_steps - 1:
            capture("loaded")

    final_pose = app.kinematics.forward(app.robot.read_joint_state())
    final_position = final_pose.position_m.copy()

    loaded_index = max(0, load_steps - 1)
    loaded_position = records["actual"][loaded_index]
    loaded_offset = records["offset"][loaded_index]
    displacement = loaded_position - initial_position
    final_residual = final_position - initial_position
    force_unit = force_base / max(float(np.linalg.norm(force_base)), 1e-15)

    metrics = dict(
        direction=direction_name,
        external_force_base_n=force_base.tolist(),
        initial_position_m=initial_position.tolist(),
        loaded_position_m=loaded_position.tolist(),
        displacement_m=displacement.tolist(),
        displacement_along_force_m=float(displacement @ force_unit),
        perpendicular_displacement_m=float(
            np.linalg.norm(displacement - (displacement @ force_unit) * force_unit)
        ),
        steady_offset_m=loaded_offset.tolist(),
        released_position_m=final_position.tolist(),
        released_residual_m=final_residual.tolist(),
        final_offset_m=records["offset"][-1].tolist(),
        status=app.servo.status,
        fault=app.servo.fault,
        contact_count=int(app.data.ncon),
        initial_quaternion_wxyz=initial_quaternion.tolist(),
        command_frame=app.command_frame,
        command_frame_mode=app.command_frame_mode,
    )

    arrays = {
        "t": np.asarray(records["t"]),
        "actual": np.asarray(records["actual"]),
        "nominal": np.asarray(records["nominal"]),
        "corrected": np.asarray(records["corrected"]),
        "offset": np.asarray(records["offset"]),
        "external_force": np.asarray(records["external_force"]),
    }
    if renderer is not None:
        try:
            renderer.close()
        except Exception:
            pass
    return metrics, arrays, frames, snapshots


def _plot_directions(results, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    planes = (
        (0, 1, "X (m)", "Y (m)", "XY"),
        (0, 2, "X (m)", "Z (m)", "XZ"),
        (1, 2, "Y (m)", "Z (m)", "YZ"),
    )
    for axis, (i, j, xlabel, ylabel, title) in zip(axes, planes):
        axis.set_title(f"Actual TCP displacement / force ({title})")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        for result in results:
            force = np.asarray(result["external_force_base_n"], dtype=float)
            displacement = np.asarray(result["displacement_m"], dtype=float)
            scale = 0.025 / max(float(np.linalg.norm(force)), 1e-15)
            axis.arrow(0, 0, force[i] * scale, force[j] * scale,
                       color="tab:blue", alpha=0.35, width=0.0006,
                       length_includes_head=True)
            axis.arrow(0, 0, displacement[i], displacement[j],
                       color="tab:red", width=0.0006, length_includes_head=True)
            axis.text(displacement[i] * 1.05, displacement[j] * 1.05,
                      result["direction"], color="tab:red", fontsize=8)
        limit = 0.03
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.axhline(0.0, color="0.8", linewidth=0.5)
        axis.axvline(0.0, color="0.8", linewidth=0.5)
        axis.grid(alpha=0.3)
    fig.suptitle("JAKA low: blue=injected external force direction, red=actual TCP move")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_step_response(arrays, name: str, path: Path, load_seconds: float) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    t = arrays["t"]
    axes[0].plot(t, arrays["actual"][:, 0] - arrays["actual"][0, 0], label="actual - initial")
    axes[0].plot(t, arrays["nominal"][:, 0] - arrays["nominal"][0, 0], "--", label="nominal - initial")
    axes[0].plot(t, arrays["corrected"][:, 0] - arrays["corrected"][0, 0], ":", label="corrected - initial")
    axes[0].set_ylabel("X displacement (m)")
    axes[0].set_title(f"JAKA low admittance step: {name}")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(t, arrays["external_force"][:, 0], label="external Fx (processed)")
    axes[1].plot(t, arrays["offset"][:, 0], label="admittance offset x")
    axes[1].axvline(load_seconds, color="0.4", linestyle="--", label="force released")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("N / m")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_force_sweep(sweep_results, path: Path) -> None:
    forces = np.array([r["force_n"] for r in sweep_results], dtype=float)
    offsets = np.array([r["steady_offset_m"][0] for r in sweep_results], dtype=float)
    displacements = np.array([r["displacement_m"][0] for r in sweep_results], dtype=float)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(forces, offsets, "o-", label="steady admittance offset")
    axis.plot(forces, displacements, "s--", label="steady actual displacement")
    axis.axhline(0.02, color="tab:red", linestyle=":", label="max offset 0.02 m")
    axis.set_xlabel("virtual external force +X (N)")
    axis.set_ylabel("displacement (m)")
    axis.set_title("JAKA low: force magnitude response (translation admittance)")
    axis.grid(alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_contact_sheet(snapshots: dict, path: Path) -> bool:
    """Arrange the six loaded render frames into one labeled image."""
    if not snapshots:
        return False
    order = [name for name in DIRECTIONS if name in snapshots]
    if not order:
        return False
    images = []
    for name in order:
        image = Image.fromarray(snapshots[name]).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 24), fill=(20, 20, 20))
        draw.text((6, 5), f"virtual external force {name} (base frame)", fill=(255, 255, 255))
        images.append(image)
    columns = 3
    rows = (len(images) + columns - 1) // columns
    width, height = images[0].size
    sheet = Image.new("RGB", (columns * width, rows * height), (30, 30, 30))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * width, (index // columns) * height))
    try:
        sheet.save(path)
        return True
    except Exception:
        return False


def _save_gif(frames, path: Path, duration_ms: int = 100) -> bool:
    if not frames:
        return False
    try:
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
        )
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-n", type=float, default=5.0,
                        help="force magnitude for the six direction checks")
    parser.add_argument("--load-seconds", type=float, default=1.5)
    parser.add_argument("--release-seconds", type=float, default=1.5)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "simulation" / "mujoco" / "reports" / "jaka_admittance_experiment")
    parser.add_argument("--no-render", action="store_true",
                        help="skip MuJoCo render GIF creation")
    parser.add_argument("--render-stride", type=int, default=15)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    directions = [name for name in DIRECTIONS]
    case_results = []
    direction_arrays = {}
    snapshot_by_direction = {}
    gif_frames = []
    started = time.time()

    for index, name in enumerate(directions):
        force = DIRECTIONS[name] * args.force_n
        # Render only the +X case to keep the artifact small; the plots cover all.
        render = (not args.no_render) and name == "+X"
        metrics, arrays, frames, snapshots = _run_case(
            name,
            force,
            args.load_seconds,
            args.release_seconds,
            render=render,
            render_stride=max(1, args.render_stride),
            snapshot=not args.no_render,
        )
        metrics["force_n"] = args.force_n
        case_results.append(metrics)
        direction_arrays[name] = arrays
        gif_frames.extend(frames)
        if "loaded" in snapshots:
            snapshot_by_direction[name] = snapshots["loaded"]
        print(
            f"[{index + 1}/{len(directions)}] {name:>2} force={args.force_n:g}N "
            f"dp={np.round(metrics['displacement_m'], 4).tolist()} "
            f"offset={np.round(metrics['steady_offset_m'], 4).tolist()} "
            f"residual={np.round(metrics['released_residual_m'], 4).tolist()}"
        )

    # Force magnitude sweep along +X.
    sweep_results = []
    for force_n in FORCE_MAGNITUDES:
        metrics, _, _, _ = _run_case(
            "+X",
            np.array([force_n, 0.0, 0.0]),
            args.load_seconds,
            args.release_seconds,
            render=False,
            render_stride=1,
        )
        metrics["force_n"] = force_n
        sweep_results.append(metrics)
        print(
            f"[sweep] Fx={force_n:g}N offset={np.round(metrics['steady_offset_m'], 4).tolist()} "
            f"dp={np.round(metrics['displacement_m'], 4).tolist()}"
        )

    _plot_directions(case_results, args.output_dir / "directions.png")
    _plot_step_response(
        direction_arrays["+X"],
        "+X 5 N",
        args.output_dir / "step_response_plus_x.png",
        args.load_seconds,
    )
    _plot_force_sweep(sweep_results, args.output_dir / "force_sweep.png")
    gif_written = _save_gif(gif_frames, args.output_dir / "step_response_plus_x.gif")
    snapshot_written = False
    if not args.no_render:
        snapshot_written = _save_contact_sheet(
            snapshot_by_direction, args.output_dir / "loaded_poses.png"
        )

    report = dict(
        model="jaka",
        keyframe="low",
        command_frame=case_results[0]["command_frame"],
        command_frame_mode=case_results[0]["command_frame_mode"],
        base_frame="jaka_base_link",
        admittance_axes="translation",
        force_magnitude_n=args.force_n,
        load_seconds=args.load_seconds,
        release_seconds=args.release_seconds,
        directions=case_results,
        force_sweep=sweep_results,
        artifacts=dict(
            directions_png="directions.png",
            step_response_png="step_response_plus_x.png",
            force_sweep_png="force_sweep.png",
            step_response_gif="step_response_plus_x.gif" if gif_written else None,
            loaded_poses_png="loaded_poses.png" if snapshot_written else None,
        ),
        scope="Virtual F/T injection through the admittance chain; MuJoCo dynamics are real, "
              "the external wrench is not an actuator force.",
        elapsed_s=time.time() - started,
    )
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    print(json.dumps(dict(
        passed=all(not r["fault"] for r in case_results + sweep_results),
        directions={r["direction"]: np.round(r["displacement_m"], 4).tolist() for r in case_results},
        force_sweep={r["force_n"]: np.round(r["steady_offset_m"], 4).tolist() for r in sweep_results},
        artifacts=report["artifacts"],
    ), indent=2))


if __name__ == "__main__":
    main()
