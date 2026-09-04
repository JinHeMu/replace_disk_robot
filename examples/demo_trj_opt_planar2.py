#!/usr/bin/env python3
"""Visualize trajectory optimization on the 2-DOF planar arm.

Run:
    python3 scripts/demo_trj_opt_planar2.py --save
    python3 scripts/demo_trj_opt_planar2.py --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hard_disk_robot.core import JointState
from hard_disk_robot.kinematics.planar2 import Planar2LinkKinematics
from hard_disk_robot.planning.planar2_collision import (
    CircleObstacle,
    Planar2LinkCollisionChecker,
)
from hard_disk_robot.planning.search import RRTConnectPlanner
from hard_disk_robot.planning.trj_opt import TrajectoryOptimizer


def _tip_path(kinematics, path):
    points = []
    for point in path:
        pose = kinematics.forward(JointState(kinematics.joint_names, point.position_rad))
        points.append(pose.position_m[:2])
    return np.asarray(points)


def _plot_arm(ax, kinematics, joints, color, alpha=0.5, linewidth=1.5):
    q = joints.position_rad
    base = np.array([0.0, 0.0])
    elbow = np.array([
        kinematics.link1_m * np.cos(q[0]),
        kinematics.link1_m * np.sin(q[0]),
    ])
    tip = np.array([
        kinematics.link1_m * np.cos(q[0])
        + kinematics.link2_m * np.cos(q[0] + q[1]),
        kinematics.link1_m * np.sin(q[0])
        + kinematics.link2_m * np.sin(q[0] + q[1]),
    ])
    ax.plot(
        [base[0], elbow[0], tip[0]],
        [base[1], elbow[1], tip[1]],
        "-o",
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        markersize=3,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--save",
        type=Path,
        nargs="?",
        const=Path(__file__).resolve().parents[1]
        / "simulation"
        / "mujoco"
        / "captures"
        / "trj_opt_planar2_demo.png",
        help="save PNG (default path under simulation/mujoco/captures/)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    kinematics = Planar2LinkKinematics(link1_m=1.0, link2_m=1.0)
    checker = Planar2LinkCollisionChecker(
        kinematics,
        obstacles=[CircleObstacle(x=1.5, y=0.0, radius=0.2)],
        link_radius_m=0.02,
    )

    start = JointState(kinematics.joint_names, [-1.0, 0.0])
    goal = JointState(kinematics.joint_names, [2.0, 0.0])

    rrt = RRTConnectPlanner(
        checker,
        sample_bounds=[(-3.2, 3.2), (-3.2, 3.2)],
        step_size_rad=0.2,
        max_iterations=5000,
        max_connection_steps=300,
        random_seed=args.seed,
    )
    print("Planning RRT-Connect ...")
    initial = rrt.plan(start, goal)
    print(f"Initial path: {len(initial)} waypoints")

    optimizer = TrajectoryOptimizer(
        kinematics,
        collision_checker=checker,
        task_weight=5.0,
        velocity_weight=10.0,
        acceleration_weight=2.0,
        dt_s=0.1,
        method="auto",
        max_iterations_gradient=60,
    )
    print("Optimizing trajectory ...")
    optimized = optimizer.optimize(initial)
    print(f"Optimized path: {len(optimized)} waypoints")

    before_cost = optimizer.evaluate(initial)
    after_cost = optimizer.evaluate(optimized)
    print("Cost before:", {k: round(v, 4) for k, v in before_cost.items()})
    print("Cost after: ", {k: round(v, 4) for k, v in after_cost.items()})
    print(f"Total cost improvement: {before_cost['total']:.4f} -> {after_cost['total']:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # before workspace
    ax = axes[0, 0]
    ax.set_title("Before: workspace")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2.5, 3.0)
    ax.set_ylim(-2.5, 2.5)
    circle = plt.Circle((1.5, 0.0), 0.2, color="red", alpha=0.25, zorder=1)
    ax.add_patch(circle)
    tip_before = _tip_path(kinematics, initial)
    ax.plot(tip_before[:, 0], tip_before[:, 1], "--", color="tab:purple", label="tip path")
    for point in initial[::6]:
        _plot_arm(ax, kinematics, JointState(kinematics.joint_names, point.position_rad), "tab:orange", alpha=0.35)
    _plot_arm(ax, kinematics, start, "tab:green", alpha=1.0, linewidth=2.5)
    _plot_arm(ax, kinematics, goal, "tab:blue", alpha=1.0, linewidth=2.5)
    ax.legend(loc="upper left", fontsize="small")

    # after workspace
    ax = axes[0, 1]
    ax.set_title("After: workspace")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2.5, 3.0)
    ax.set_ylim(-2.5, 2.5)
    circle = plt.Circle((1.5, 0.0), 0.2, color="red", alpha=0.25, zorder=1)
    ax.add_patch(circle)
    tip_after = _tip_path(kinematics, optimized)
    ax.plot(tip_after[:, 0], tip_after[:, 1], "--", color="tab:purple", label="tip path")
    for point in optimized[::6]:
        _plot_arm(ax, kinematics, JointState(kinematics.joint_names, point.position_rad), "tab:orange", alpha=0.35)
    _plot_arm(ax, kinematics, start, "tab:green", alpha=1.0, linewidth=2.5)
    _plot_arm(ax, kinematics, goal, "tab:blue", alpha=1.0, linewidth=2.5)
    ax.legend(loc="upper left", fontsize="small")

    # before joint space
    ax = axes[1, 0]
    ax.set_title("Before: joint space")
    ax.set_xlabel("joint1 [rad]")
    ax.set_ylabel("joint2 [rad]")
    ax.grid(True, alpha=0.3)
    q_before = np.array([p.position_rad for p in initial])
    ax.plot(q_before[:, 0], q_before[:, 1], "-o", color="tab:orange", markersize=3, linewidth=1.5, label="trajectory")
    ax.plot(*start.position_rad, "*", color="tab:green", markersize=12, label="start")
    ax.plot(*goal.position_rad, "*", color="tab:blue", markersize=12, label="goal")
    ax.legend(loc="upper right", fontsize="small")

    # after joint space
    ax = axes[1, 1]
    ax.set_title("After: joint space")
    ax.set_xlabel("joint1 [rad]")
    ax.set_ylabel("joint2 [rad]")
    ax.grid(True, alpha=0.3)
    q_after = np.array([p.position_rad for p in optimized])
    ax.plot(q_after[:, 0], q_after[:, 1], "-o", color="tab:orange", markersize=3, linewidth=1.5, label="trajectory")
    ax.plot(*start.position_rad, "*", color="tab:green", markersize=12, label="start")
    ax.plot(*goal.position_rad, "*", color="tab:blue", markersize=12, label="goal")
    ax.legend(loc="upper right", fontsize="small")

    fig.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")

    if args.show:
        plt.show()
    else:
        print("Run with --show to open an interactive window.")


if __name__ == "__main__":
    main()
