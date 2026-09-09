#!/usr/bin/env python3
"""Visualize RRT-Connect on the 2-DOF planar arm.

Run:
    python3 examples/demo_rrt_planar2.py --save
    python3 examples/demo_rrt_planar2.py --show
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
        markersize=4,
        label=label,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    parser.add_argument(
        "--save",
        type=Path,
        nargs="?",
        const=Path(__file__).resolve().parents[1] / "simulation" / "mujoco" / "captures" / "rrt_planar2_demo.png",
        help="save the figure to a PNG (default path under simulation/mujoco/captures/)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # 2-link planar arm
    kinematics = Planar2LinkKinematics(link1_m=1.0, link2_m=1.0)

    obstacles = [
        CircleObstacle(x=1.5, y=0.0, radius=0.2),
    ]
    checker = Planar2LinkCollisionChecker(
        kinematics,
        obstacles=obstacles,
        link_radius_m=0.02,
    )

    start = JointState(kinematics.joint_names, [-1.0, 0.0])
    goal = JointState(kinematics.joint_names, [2.0, 0.0])

    planner = RRTConnectPlanner(
        checker,
        sample_bounds=[(-3.2, 3.2), (-3.2, 3.2)],
        step_size_rad=0.2,
        max_iterations=5000,
        max_connection_steps=300,
        random_seed=args.seed,
    )

    print("Planning RRT-Connect ...")
    path = planner.plan(start, goal)
    print(f"Path found: {len(path)} waypoints")

    # Sample a few arm configurations along the path for visualization.
    sample_indices = np.linspace(0, len(path) - 1, 8, dtype=int)

    fig, (ax_workspace, ax_joint) = plt.subplots(1, 2, figsize=(12, 5.5))

    # ---------- Workspace plot ----------
    ax_workspace.set_title("2-DOF planar arm workspace")
    ax_workspace.set_aspect("equal")
    ax_workspace.grid(True, alpha=0.3)
    ax_workspace.set_xlim(-2.5, 3.0)
    ax_workspace.set_ylim(-2.5, 2.5)

    # obstacles
    for obstacle in obstacles:
        circle = plt.Circle(
            (obstacle.x, obstacle.y),
            obstacle.radius,
            color="red",
            alpha=0.25,
            zorder=1,
        )
        ax_workspace.add_patch(circle)

    # start / goal arms
    _plot_arm(ax_workspace, kinematics, start, "tab:green", label="start")
    _plot_arm(ax_workspace, kinematics, goal, "tab:blue", label="goal")

    # Cartesian tip trajectory
    tip_points = []
    for point in path:
        pose = kinematics.forward(JointState(kinematics.joint_names, point.position_rad))
        tip_points.append(pose.position_m[:2])
    tip_points = np.asarray(tip_points)
    ax_workspace.plot(
        tip_points[:, 0],
        tip_points[:, 1],
        "--",
        color="tab:purple",
        alpha=0.7,
        linewidth=1.2,
        label="end-effector path",
        zorder=2,
    )

    # sampled arm poses
    for index in sample_indices:
        point = path[index]
        alpha = 0.2 + 0.8 * (index / max(len(path) - 1, 1))
        _plot_arm(
            ax_workspace,
            kinematics,
            JointState(kinematics.joint_names, point.position_rad),
            "tab:orange",
            alpha=alpha,
            linewidth=1.2,
        )

    ax_workspace.legend(loc="upper left", fontsize="small")

    # ---------- Joint-space plot ----------
    ax_joint.set_title("Joint-space RRT-Connect path")
    ax_joint.set_xlabel("joint1 [rad]")
    ax_joint.set_ylabel("joint2 [rad]")
    ax_joint.grid(True, alpha=0.3)
    ax_joint.set_xlim(-3.5, 3.5)
    ax_joint.set_ylim(-3.5, 3.5)

    q_path = np.array([point.position_rad for point in path])
    ax_joint.plot(
        q_path[:, 0],
        q_path[:, 1],
        "-o",
        color="tab:orange",
        markersize=3,
        linewidth=1.5,
        label="RRT path",
    )
    ax_joint.plot(
        [start.position_rad[0]],
        [start.position_rad[1]],
        "*",
        color="tab:green",
        markersize=12,
        label="start",
    )
    ax_joint.plot(
        [goal.position_rad[0]],
        [goal.position_rad[1]],
        "*",
        color="tab:blue",
        markersize=12,
        label="goal",
    )
    ax_joint.legend(loc="upper right", fontsize="small")

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
