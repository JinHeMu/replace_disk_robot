#!/usr/bin/env python3
"""Insertion-only demo: the gripper starts holding the drive and inserts it into the empty server bay.

The MuJoCo scene is designed with:
* one empty high-precision bay (1 mm clearance),
* the replacement drive welded/held by the gripper from the start.

This example only demonstrates the insertion movement; it does not simulate
real drive-latch mechanics.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hard_disk_robot.adapters.mujoco import (  # noqa: E402
    MujocoRobotAdapter,
    load_model,
    reset_home,
)
from hard_disk_robot.core import JointState, Pose  # noqa: E402
from hard_disk_robot.kinematics.ur5e import UR5eKinematics  # noqa: E402

APPROACH_OFFSET_M = 0.20


def _quaternion_from_matrix(rotation: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.roll(quat_xyzw, 1)


def _insertion_pose(model, data) -> Pose:
    """Pre-insertion pose that keeps the gripper outside the narrow bay.

    The held drive is aligned with the empty bay (long axis along world X).
    """
    import mujoco

    slot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_drive_carrier")
    if slot_id < 0:
        raise KeyError("target_drive_carrier is missing from the scene")
    slot_center = data.xpos[slot_id].copy()

    # g_base frame: x -> Y, y -> Z, z -> X, matching the held drive's long axis
    # along world X (insertion direction).
    rotation = np.column_stack(([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]))
    position = slot_center - APPROACH_OFFSET_M * rotation[:, 2]
    return Pose(
        frame_id="world",
        position_m=position,
        quaternion_wxyz=_quaternion_from_matrix(rotation),
    )


def _move_to_q(
    model,
    data,
    robot,
    start_q,
    target_q,
    duration_s,
    viewer,
    joint_names,
) -> None:
    import mujoco

    start_q = np.asarray(start_q, dtype=float)
    target_q = np.asarray(target_q, dtype=float)
    steps = max(1, int(round(duration_s / model.opt.timestep)))
    for step_index in range(steps):
        if viewer is not None and not viewer.is_running():
            return
        progress = (step_index + 1) / steps

        def _quintic(p: float) -> float:
            return 10.0 * p**3 - 15.0 * p**4 + 6.0 * p**5

        commanded = start_q + _quintic(progress) * (target_q - start_q)
        robot.command_joint_positions(JointState(joint_names, commanded))
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - 0.001))


def _hold(model, data, robot, viewer, seconds, target_q) -> None:
    import mujoco

    joint_names = robot.read_joint_state().names
    target_q = np.asarray(target_q, dtype=float)
    steps = max(1, int(round(seconds / model.opt.timestep)))
    for _ in range(steps):
        if viewer is not None and not viewer.is_running():
            return
        robot.command_joint_positions(JointState(joint_names, target_q))
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="open MuJoCo viewer")
    parser.add_argument("--pause-seconds", type=float, default=5.0)
    args = parser.parse_args()

    print("[1/4] Loading server scene with gripper-held drive ...")
    model, data = load_model()
    reset_home(model, data)
    robot = MujocoRobotAdapter(model, data)
    # The gripper is already closed in the home keyframe (ctrl=255).
    print("    gripper already closed around the drive")

    print("[2/4] Computing insertion pose ...")
    kinematics = UR5eKinematics(end_effector_frame="g_base")
    target_pose = _insertion_pose(model, data)
    home_q = robot.arm_position().copy()
    insert_q = kinematics.inverse(
        target_pose,
        JointState(kinematics.joint_names, home_q),
    ).position_rad
    print(f"    insert_q = {insert_q.round(4).tolist()}")

    viewer = None
    if args.show:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data)

    print("[3/4] Moving the held drive into the empty bay ...")
    try:
        _hold(model, data, robot, viewer, args.pause_seconds, home_q)
        _move_to_q(
            model,
            data,
            robot,
            home_q,
            insert_q,
            duration_s=3.0,
            viewer=viewer,
            joint_names=kinematics.joint_names,
        )
        _hold(model, data, robot, viewer, args.pause_seconds, insert_q)

        print("[4/4] Release gripper (demo visual release) ...")
        robot.command_gripper_opening(0.085)
        _hold(model, data, robot, viewer, 2.0, insert_q)
    finally:
        if viewer is not None:
            viewer.close()

    print("INSERTION DEMO COMPLETE")
    print(f"    robot arm q = {robot.arm_position().round(4).tolist()}")


if __name__ == "__main__":
    main()
