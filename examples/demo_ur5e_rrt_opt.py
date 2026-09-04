#!/usr/bin/env python3
"""RRT-Connect + trajectory optimization demo for the 6-DOF UR5e arm.

This script uses:
* UR5e Kinematics (Pinocchio wrapper in ``kinematics/ur5e.py``)
* RRTConnectPlanner from ``planning/search.py``
* TrajectoryOptimizer from ``planning/trj_opt.py``

Run with a Pinocchio-compatible Python environment, e.g. the conda env:

    (hard_disk_robot) $ python3 scripts/demo_ur5e_rrt_opt.py --save

Or with interactive window:

    (hard_disk_robot) $ python3 scripts/demo_ur5e_rrt_opt.py --show
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hard_disk_robot.core import CollisionCheckerPort, JointState, TrajectoryPoint
from hard_disk_robot.kinematics.ur5e import UR5eKinematics
from hard_disk_robot.planning.search import RRTConnectPlanner
from hard_disk_robot.planning.trj_opt import TrajectoryOptimizer


@dataclass
class SphereObstacle3D:
    x: float
    y: float
    z: float
    radius: float


class UR5eSphereCollisionChecker:
    """Simple UR5e collision checker based on the end-effector position.

    This is not a full collision model; it is used to demonstrate RRT and
    trajectory optimization with a sparse spherical workspace obstacle.
    """

    def __init__(
        self,
        kinematics: UR5eKinematics,
        obstacles: list[SphereObstacle3D],
        *,
        tool_radius_m: float = 0.05,
    ) -> None:
        self.kinematics = kinematics
        self.obstacles = obstacles
        self.tool_radius_m = tool_radius_m

    def is_collision_free(self, joints: JointState) -> bool:
        return self.minimum_distance(joints) > 0.0

    def minimum_distance(self, joints: JointState) -> float:
        pose = self.kinematics.forward(joints)
        tip = pose.position_m
        distances = []
        for obstacle in self.obstacles:
            center = np.array([obstacle.x, obstacle.y, obstacle.z])
            distances.append(
                float(np.linalg.norm(tip - center))
                - obstacle.radius
                - self.tool_radius_m
            )
        return min(distances) if distances else float("inf")


def _subsample(path: list[TrajectoryPoint], n: int) -> list[TrajectoryPoint]:
    """Resample a trajectory to ``n`` points keeping first/last fixed."""
    if len(path) <= n:
        return path
    indices = np.linspace(0, len(path) - 1, n, dtype=int)
    return [path[i] for i in indices]


def _tip_points(
    kinematics: UR5eKinematics,
    path: list[TrajectoryPoint],
) -> NDArray[np.float64]:
    points = []
    for point in path:
        pose = kinematics.forward(JointState(kinematics.joint_names, point.position_rad))
        points.append(pose.position_m)
    return np.asarray(points)


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
        / "ur5e_rrt_opt_demo.png",
        help="save PNG (default: simulation/mujoco/captures/ur5e_rrt_opt_demo.png)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("Loading UR5e kinematics ...")
    kinematics = UR5eKinematics()
    lower, upper = kinematics.joint_limits_rad

    start_q = np.zeros(6)
    goal_q = np.array([-2.9746, -1.5307, 1.1305, -1.1455, -1.5708, -1.3353])

    start = JointState(kinematics.joint_names, start_q)
    goal = JointState(kinematics.joint_names, goal_q)

    obstacles = [
        SphereObstacle3D(x=0.0, y=0.0, z=0.45, radius=0.22),
    ]
    checker = UR5eSphereCollisionChecker(kinematics, obstacles, tool_radius_m=0.05)

    print("Planning RRT-Connect for UR5e ...")
    rrt = RRTConnectPlanner(
        checker,
        sample_bounds=list(zip(lower, upper)),
        step_size_rad=0.15,
        max_iterations=20000,
        max_connection_steps=500,
        random_seed=args.seed,
    )
    rrt_path = rrt.plan(start, goal)
    print(f"RRT path: {len(rrt_path)} waypoints")

    # Keep optimization cost manageable for the demo.
    initial = _subsample(rrt_path, 16)
    print(f"Subsampled optimization input: {len(initial)} waypoints")

    optimizer = TrajectoryOptimizer(
        kinematics,
        collision_checker=checker,
        task_weight=5.0,
        velocity_weight=8.0,
        acceleration_weight=2.0,
        dt_s=0.1,
        method="gradient",
        max_iterations_gradient=30,
    )
    print("Optimizing trajectory ...")
    optimized = optimizer.optimize(initial)
    print(f"Optimized trajectory: {len(optimized)} waypoints")

    before_cost = optimizer.evaluate(initial)
    after_cost = optimizer.evaluate(optimized)
    print("Before:", {k: round(v, 4) for k, v in before_cost.items()})
    print("After: ", {k: round(v, 4) for k, v in after_cost.items()})

    tip_before = _tip_points(kinematics, initial)
    tip_after = _tip_points(kinematics, optimized)
    q_before = np.array([p.position_rad for p in initial])
    q_after = np.array([p.position_rad for p in optimized])

    fig = plt.figure(figsize=(14, 8))

    # 3D workspace
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.set_title("UR5e end-effector path\nbefore vs after")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.plot(
        tip_before[:, 0],
        tip_before[:, 1],
        tip_before[:, 2],
        "o--",
        color="tab:red",
        alpha=0.5,
        markersize=3,
        label="before",
    )
    ax3d.plot(
        tip_after[:, 0],
        tip_after[:, 1],
        tip_after[:, 2],
        "o-",
        color="tab:blue",
        markersize=3,
        label="after",
    )
    for obstacle in obstacles:
        # Draw a transparent wire sphere.
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        xs = obstacle.x + obstacle.radius * np.outer(np.cos(u), np.sin(v))
        ys = obstacle.y + obstacle.radius * np.outer(np.sin(u), np.sin(v))
        zs = obstacle.z + obstacle.radius * np.outer(np.ones_like(u), np.cos(v))
        ax3d.plot_wireframe(xs, ys, zs, color="red", alpha=0.25)
    ax3d.legend()

    # Joint trajectories before/after
    for i in range(6):
        ax = fig.add_subplot(2, 6, 6 + 1 + i)
        ax.set_title(kinematics.joint_names[i])
        ax.plot(q_before[:, i], "o--", color="tab:red", alpha=0.5, markersize=3)
        ax.plot(q_after[:, i], "o-", color="tab:blue", markersize=3)
        ax.grid(True, alpha=0.3)

    fig.suptitle("UR5e RRT-Connect + trajectory optimization demo")
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
