#!/usr/bin/env python3
"""Compare raw RRT, pruned RRT, and trajectory optimization.

The script runs the same 2-DOF planar-arm planning problem with the same
random seed:

1. raw RRT-Connect path with ``path_pruning=False``;
2. shortened RRT path with ``path_pruning=True``;
3. the pruned path is resampled along its arc length, then optimized with
   :class:`TrajectoryOptimizer`.

The resulting figure overlays the three paths in joint space and task space,
and reports optimization costs and collision checks.

Run:
    python3 examples/demo_rrt_pruning_compare.py --save
    python3 examples/demo_rrt_pruning_compare.py --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replace_disk_robot.core import JointState, TrajectoryPoint
from replace_disk_robot.kinematics.planar2 import Planar2LinkKinematics
from replace_disk_robot.planning.collision import is_path_collision_free
from replace_disk_robot.planning.optimize import TrajectoryOptimizer
from replace_disk_robot.planning.planar2_collision import (
    CircleObstacle,
    Planar2LinkCollisionChecker,
)
from replace_disk_robot.planning.search import RRTConnectPlanner


def _plot_arm(
    ax: matplotlib.axes.Axes,
    kinematics: Planar2LinkKinematics,
    joints: JointState,
    color: str,
    alpha: float = 1.0,
    label: str | None = None,
    linewidth: float = 2.0,
) -> None:
    q = joints.position_rad
    base = np.array([0.0, 0.0])
    elbow = np.array(
        [
            kinematics.link1_m * np.cos(q[0]),
            kinematics.link1_m * np.sin(q[0]),
        ]
    )
    tip = np.array(
        [
            kinematics.link1_m * np.cos(q[0])
            + kinematics.link2_m * np.cos(q[0] + q[1]),
            kinematics.link1_m * np.sin(q[0])
            + kinematics.link2_m * np.sin(q[0] + q[1]),
        ]
    )
    ax.plot(
        [base[0], elbow[0], tip[0]],
        [base[1], elbow[1], tip[1]],
        "-o",
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        markersize=4,
        label=label,
    )


def _q_array(path: list[TrajectoryPoint]) -> np.ndarray:
    return np.array([point.position_rad for point in path], dtype=float)


def _path_length(path: list[TrajectoryPoint]) -> float:
    q = _q_array(path)
    if q.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1)))


def _tip_array(
    kinematics: Planar2LinkKinematics,
    path: list[TrajectoryPoint],
) -> np.ndarray:
    return np.array(
        [
            kinematics.forward(
                JointState(kinematics.joint_names, point.position_rad)
            ).position_m[:2]
            for point in path
        ],
        dtype=float,
    )


def _resample_path(path: list[TrajectoryPoint], n: int) -> list[TrajectoryPoint]:
    """Resample a path to ``n`` points evenly along its arc length."""
    if len(path) >= n or len(path) < 2:
        return list(path)

    q = _q_array(path)
    times = np.array([point.time_from_start_s for point in path], dtype=float)
    segment_lengths = np.linalg.norm(np.diff(q, axis=0), axis=1)
    distances = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if distances[-1] <= 1e-12:
        return list(path)

    sample_distances = np.linspace(0.0, distances[-1], n)
    q_resampled = np.column_stack(
        [
            np.interp(sample_distances, distances, q[:, joint_index])
            for joint_index in range(q.shape[1])
        ]
    )
    t_resampled = np.interp(sample_distances, distances, times)
    return [
        TrajectoryPoint(
            time_from_start_s=float(t_resampled[i]),
            position_rad=q_resampled[i],
        )
        for i in range(n)
    ]


def _path_is_collision_free(
    checker: Planar2LinkCollisionChecker,
    kinematics: Planar2LinkKinematics,
    path: list[TrajectoryPoint],
    samples: int = 40,
) -> bool:
    for start, goal in zip(path[:-1], path[1:]):
        if not is_path_collision_free(
            checker,
            JointState(kinematics.joint_names, start.position_rad),
            JointState(kinematics.joint_names, goal.position_rad),
            samples=samples,
        ):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    parser.add_argument(
        "--save",
        type=Path,
        nargs="?",
        const=(
            Path(__file__).resolve().parents[1]
            / "simulation"
            / "mujoco"
            / "captures"
            / "rrt_pruning_optimization_compare_planar2.png"
        ),
        help="save the figure to a PNG (default path under simulation/mujoco/captures/)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    kinematics = Planar2LinkKinematics(link1_m=1.0, link2_m=1.0)
    obstacles = [CircleObstacle(x=1.5, y=0.0, radius=0.2)]
    checker = Planar2LinkCollisionChecker(
        kinematics,
        obstacles=obstacles,
        link_radius_m=0.02,
    )

    start = JointState(kinematics.joint_names, [-1.0, 0.0])
    goal = JointState(kinematics.joint_names, [2.0, 0.0])

    common_planner_kwargs = dict(
        sample_bounds=[(-3.2, 3.2), (-3.2, 3.2)],
        step_size_rad=0.2,
        max_iterations=5000,
        max_connection_steps=300,
        random_seed=args.seed,
    )

    print("Planning raw RRT-Connect path ...")
    raw_planner = RRTConnectPlanner(
        checker,
        path_pruning=False,
        **common_planner_kwargs,
    )
    raw_path = list(raw_planner.plan(start, goal))

    print("Planning pruned RRT-Connect path ...")
    pruned_planner = RRTConnectPlanner(
        checker,
        path_pruning=True,
        path_prune_step_rad=0.05,
        **common_planner_kwargs,
    )
    pruned_path = list(pruned_planner.plan(start, goal))

    # The pruned path can be very short (often only 2-3 waypoints).  Resample it
    # along arc length so the finite-difference optimizer has enough decision
    # variables and remains numerically well behaved.
    optimization_samples = max(16, min(len(raw_path), 40))
    optimization_input = _resample_path(pruned_path, optimization_samples)

    optimizer = TrajectoryOptimizer(
        kinematics,
        collision_checker=checker,
        task_weight=5.0,
        velocity_weight=10.0,
        acceleration_weight=2.0,
        dt_s=0.1,
        method="gradient",
        max_iterations_gradient=80,
    )
    print("Optimizing resampled pruned trajectory ...")
    optimized_path = list(optimizer.optimize(optimization_input))

    raw_length = _path_length(raw_path)
    pruned_length = _path_length(pruned_path)
    optimization_input_length = _path_length(optimization_input)
    optimized_length = _path_length(optimized_path)
    length_reduction = (
        100.0 * (raw_length - pruned_length) / raw_length if raw_length > 0.0 else 0.0
    )

    before_cost = optimizer.evaluate(optimization_input)
    after_cost = optimizer.evaluate(optimized_path)
    total_improvement = (
        100.0 * (before_cost["total"] - after_cost["total"]) / before_cost["total"]
        if before_cost["total"] > 0.0
        else 0.0
    )
    raw_path_free = _path_is_collision_free(checker, kinematics, raw_path)
    pruned_path_free = _path_is_collision_free(checker, kinematics, pruned_path)
    optimized_path_free = _path_is_collision_free(checker, kinematics, optimized_path)

    print(
        f"Raw RRT:             {len(raw_path):3d} waypoints, "
        f"{raw_length:.3f} rad, collision-free={raw_path_free}"
    )
    print(
        f"Pruned RRT:          {len(pruned_path):3d} waypoints, "
        f"{pruned_length:.3f} rad "
        f"({length_reduction:.1f}% shorter), collision-free={pruned_path_free}"
    )
    print(
        f"Optimization input:  {len(optimization_input):3d} waypoints, "
        f"{optimization_input_length:.3f} rad"
    )
    print(
        f"Optimized path:      {len(optimized_path):3d} waypoints, "
        f"{optimized_length:.3f} rad, collision-free={optimized_path_free}"
    )
    print(
        "Optimization cost:   "
        f"{before_cost['total']:.2f} -> {after_cost['total']:.2f} "
        f"({total_improvement:.1f}% lower)"
    )

    raw_q = _q_array(raw_path)
    pruned_q = _q_array(pruned_path)
    optimization_input_q = _q_array(optimization_input)
    optimized_q = _q_array(optimized_path)
    raw_tip = _tip_array(kinematics, raw_path)
    pruned_tip = _tip_array(kinematics, pruned_path)
    optimized_tip = _tip_array(kinematics, optimized_path)

    fig, (ax_joint, ax_task, ax_cost) = plt.subplots(
        1,
        3,
        figsize=(17.0, 5.8),
        gridspec_kw={"width_ratios": [1.05, 1.05, 0.85]},
    )

    # ------------------------------------------------------------------
    # Joint-space comparison
    # ------------------------------------------------------------------
    ax_joint.plot(
        raw_q[:, 0],
        raw_q[:, 1],
        "--o",
        color="tab:gray",
        markersize=3.0,
        linewidth=1.2,
        alpha=0.75,
        label=f"raw RRT ({len(raw_path)} pts, {raw_length:.2f} rad)",
    )
    ax_joint.plot(
        pruned_q[:, 0],
        pruned_q[:, 1],
        "-o",
        color="tab:orange",
        markersize=5.5,
        linewidth=2.0,
        label=f"pruned ({len(pruned_path)} pts, {pruned_length:.2f} rad)",
    )
    ax_joint.plot(
        optimization_input_q[:, 0],
        optimization_input_q[:, 1],
        "--",
        color="tab:cyan",
        linewidth=1.5,
        alpha=0.9,
        label=f"optimization input ({len(optimization_input)} pts)",
    )
    ax_joint.plot(
        optimized_q[:, 0],
        optimized_q[:, 1],
        "-",
        color="tab:blue",
        linewidth=2.5,
        label=f"optimized ({len(optimized_path)} pts, {optimized_length:.2f} rad)",
    )
    ax_joint.plot(
        start.position_rad[0],
        start.position_rad[1],
        "*",
        color="tab:green",
        markersize=14,
        zorder=5,
        label="start",
    )
    ax_joint.plot(
        goal.position_rad[0],
        goal.position_rad[1],
        "*",
        color="tab:blue",
        markersize=14,
        zorder=5,
        label="goal",
    )
    ax_joint.set_title(
        "Joint-space comparison\n"
        f"raw {raw_length:.2f} rad -> pruned {pruned_length:.2f} rad "
        f"({length_reduction:.1f}% shorter)"
    )
    ax_joint.set_xlabel("joint1 [rad]")
    ax_joint.set_ylabel("joint2 [rad]")
    ax_joint.set_aspect("equal", adjustable="box")
    ax_joint.grid(True, alpha=0.3)
    ax_joint.legend(loc="best", fontsize="small")

    # ------------------------------------------------------------------
    # Task-space / end-effector comparison
    # ------------------------------------------------------------------
    for obstacle in obstacles:
        ax_task.add_patch(
            plt.Circle(
                (obstacle.x, obstacle.y),
                obstacle.radius,
                color="red",
                alpha=0.25,
                zorder=1,
            )
        )

    ax_task.plot(
        raw_tip[:, 0],
        raw_tip[:, 1],
        "--o",
        color="tab:gray",
        markersize=2.5,
        linewidth=1.2,
        alpha=0.75,
        label="raw RRT tip path",
    )
    ax_task.plot(
        pruned_tip[:, 0],
        pruned_tip[:, 1],
        "-o",
        color="tab:orange",
        markersize=4.0,
        linewidth=2.0,
        label="pruned tip path",
    )
    ax_task.plot(
        optimized_tip[:, 0],
        optimized_tip[:, 1],
        "-",
        color="tab:blue",
        linewidth=2.5,
        label="optimized tip path",
    )
    _plot_arm(ax_task, kinematics, start, "tab:green", label="start arm")
    _plot_arm(ax_task, kinematics, goal, "tab:blue", label="goal arm")
    ax_task.set_title("Task-space end-effector trajectory")
    ax_task.set_xlabel("x [m]")
    ax_task.set_ylabel("y [m]")
    ax_task.set_aspect("equal", adjustable="box")
    ax_task.grid(True, alpha=0.3)
    ax_task.set_xlim(-2.5, 3.0)
    ax_task.set_ylim(-2.5, 2.5)
    ax_task.legend(loc="upper left", fontsize="small")

    # ------------------------------------------------------------------
    # Optimization cost summary
    # ------------------------------------------------------------------
    ax_cost.axis("off")
    ax_cost.set_title("Trajectory optimization cost")
    cost_table = [
        ["metric", "before opt", "after opt"],
        ["waypoints", str(len(optimization_input)), str(len(optimized_path))],
        ["task", f"{before_cost['task']:.2f}", f"{after_cost['task']:.2f}"],
        ["velocity", f"{before_cost['velocity']:.2f}", f"{after_cost['velocity']:.2f}"],
        [
            "acceleration",
            f"{before_cost['acceleration']:.2f}",
            f"{after_cost['acceleration']:.2f}",
        ],
        ["total", f"{before_cost['total']:.2f}", f"{after_cost['total']:.2f}"],
    ]
    table = ax_cost.table(
        cellText=cost_table,
        colWidths=[0.32, 0.34, 0.34],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)

    ax_cost.text(
        0.5,
        0.05,
        f"total cost: {before_cost['total']:.1f} -> {after_cost['total']:.1f}\n"
        f"({total_improvement:.1f}% lower)\n"
        f"optimized collision-free: {optimized_path_free}",
        transform=ax_cost.transAxes,
        ha="center",
        va="bottom",
        fontsize="small",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    fig.suptitle(
        "RRT-Connect: raw -> pruned -> optimized trajectory comparison",
        fontsize=14,
    )
    fig.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")

    if args.show:
        plt.show()
    else:
        print("Run with --show to open an interactive window (or --save to write PNG).")


if __name__ == "__main__":
    main()
