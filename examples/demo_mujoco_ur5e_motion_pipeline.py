#!/usr/bin/env python3
"""Validate the P2/P3 motion pipeline on the UR5e MuJoCo model.

The script plans a small collision-free joint-space motion from the UR5e home
configuration, assigns quintic segment timing, starts the executor, advances
the MuJoCo dynamics in small control cycles, and reports arrival.

It also repeats the motion and cancels mid-execution to verify the P3 cancel
state.

Run:
    python3 examples/demo_mujoco_ur5e_motion_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replace_disk_robot.adapters.mujoco import (
    MujocoCollisionChecker,
    MujocoRobotAdapter,
    load_model,
    reset_home,
)
from replace_disk_robot.core import JointState
from replace_disk_robot.planning.search import RRTConnectPlanner
from replace_disk_robot.task import MotionPipeline, MotionPipelineState


class _JointNameKinematics:
    """Minimal kinematics handle for joint-space goals in this demo."""

    def __init__(self, joint_names: tuple[str, ...]) -> None:
        self.joint_names = tuple(joint_names)

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def forward(self, joints: JointState):
        raise NotImplementedError(
            "this MuJoCo joint-space demo does not use Cartesian goals"
        )

    def inverse(self, target, seed):
        raise NotImplementedError(
            "this MuJoCo joint-space demo does not use Cartesian goals"
        )


def _joint_limits(
    model: mujoco.MjModel,
    joint_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in joint_names
    ]
    if any(joint_id < 0 for joint_id in joint_ids):
        raise KeyError("model is missing one or more joint names")
    ranges = model.jnt_range[np.asarray(joint_ids, dtype=int)]
    return ranges[:, 0].copy(), ranges[:, 1].copy()


def _make_pipeline(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot: MujocoRobotAdapter,
    checker: MujocoCollisionChecker,
    lower: np.ndarray,
    upper: np.ndarray,
) -> MotionPipeline:
    planner = RRTConnectPlanner(
        checker,
        sample_bounds=list(zip(lower.tolist(), upper.tolist())),
        step_size_rad=0.15,
        max_iterations=5000,
        max_connection_steps=300,
        random_seed=0,
        path_pruning=True,
        path_prune_step_rad=0.05,
    )
    return MotionPipeline(
        _JointNameKinematics(tuple(robot.joint_names)),
        checker,
        planner,
        optimizer=None,
        optimize=False,
        joint_limits_rad=(lower, upper),
        arm=robot,
        velocity_limits_rad_s=np.full(len(robot.joint_names), 0.5),
        acceleration_limits_rad_s2=np.full(len(robot.joint_names), 1.0),
        min_segment_time_s=0.1,
        start_tolerance_rad=0.05,
        tracking_tolerance_rad=0.6,
        goal_tolerance_rad=0.02,
        goal_hold_s=0.1,
        timeout_s=5.0,
    )


def _run_to_goal(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pipeline: MotionPipeline,
    robot: MujocoRobotAdapter,
    goal: JointState,
    *,
    cancel_after_s: float | None = None,
) -> dict[str, float | str | bool | None]:
    plan = pipeline.submit_goal(goal, now_s=float(data.time))
    step = 0
    while pipeline.state == MotionPipelineState.EXECUTING and data.time < 20.0:
        pipeline.update(float(data.time))
        if (
            cancel_after_s is not None
            and float(data.time) >= cancel_after_s
        ):
            pipeline.cancel(float(data.time))
            break
        for _ in range(10):
            mujoco.mj_step(model, data)
        step += 1

    actual = robot.read_joint_state()
    return {
        "state": pipeline.state.value,
        "steps": step,
        "sim_time_s": float(data.time),
        "trajectory_duration_s": pipeline.status.trajectory_duration_s,
        "tracking_error_rad": pipeline.status.tracking_error_rad,
        "goal_error_rad": pipeline.status.goal_error_rad,
        "failure_reason": pipeline.status.last_error,
        "success": pipeline.state == MotionPipelineState.SUCCEEDED,
        "actual_position_rad": actual.position_rad.tolist(),
        "goal_position_rad": goal.position_rad.tolist(),
        "waypoint_count": plan.waypoint_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "simulation"
        / "mujoco"
        / "reports"
        / "ur5e_motion_pipeline_report.json",
        help="optional JSON report path",
    )
    args = parser.parse_args()

    model, data = load_model("ur5e")
    reset_home(model, data)
    robot = MujocoRobotAdapter(model, data)
    checker = MujocoCollisionChecker(model, data, use_separate_planning_data=True)
    lower, upper = _joint_limits(model, tuple(robot.joint_names))

    start = robot.read_joint_state()
    goal_position = start.position_rad.copy()
    goal_position[0] += 0.25
    goal_position[1] -= 0.15
    goal_position[2] += 0.10
    goal = JointState(start.names, goal_position)

    if not checker.is_collision_free(start):
        raise RuntimeError("UR5e start state is in collision")
    if not checker.is_collision_free(goal):
        raise RuntimeError("UR5e goal state is in collision")

    print("Planning and executing UR5e joint goal ...")
    pipeline = _make_pipeline(model, data, robot, checker, lower, upper)
    success_report = _run_to_goal(model, data, pipeline, robot, goal)
    print(json.dumps(success_report, indent=2))

    reset_home(model, data)
    print("Planning and executing UR5e joint goal, then canceling ...")
    pipeline = _make_pipeline(model, data, robot, checker, lower, upper)
    cancel_report = _run_to_goal(
        model,
        data,
        pipeline,
        robot,
        goal,
        cancel_after_s=0.15,
    )
    print(json.dumps(cancel_report, indent=2))

    report = {
        "success_case": success_report,
        "cancel_case": cancel_report,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to {args.report}")


if __name__ == "__main__":
    main()
