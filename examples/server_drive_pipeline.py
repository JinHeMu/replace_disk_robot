#!/usr/bin/env python3
"""Server-scene pipeline: load scene -> select drive -> plan grasp -> run -> close gripper -> pull out.

This is a demonstration pipeline that uses the existing backend-neutral planning
pieces together with the MuJoCo server scene.

Steps
-----
1. Load the MuJoCo server scene and reset to ``home``.
2. Select the manipulable target drive (``target_drive_carrier``).
3. Compute a grasp pose for the UR5e ``g_base`` frame using Pinocchio IK.
4. Plan a collision-free RRT-Connect path from home to the grasp configuration.
5. Execute the planned path in MuJoCo.
6. Close the Robotiq 2F-85 gripper.
7. Command the carrier mechanism to extract the drive.

Run in a Pinocchio-compatible environment:

    (hard_disk_robot) $ python3 examples/server_drive_pipeline.py

For an interactive MuJoCo viewer:

    (hard_disk_robot) $ python3 examples/server_drive_pipeline.py --show
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
    MujocoCollisionChecker,
    MujocoMechanismAdapter,
    MujocoRobotAdapter,
    load_model,
    reset_home,
)
from hard_disk_robot.core import JointState, Pose  # noqa: E402
from hard_disk_robot.kinematics.ur5e import UR5eKinematics  # noqa: E402
from hard_disk_robot.planning.search import RRTConnectPlanner  # noqa: E402

GRASP_OFFSET_M = 0.145  # distance from g_base origin to the pinch site
PULL_EXTRACTION_M = 0.035
PULL_LATCH_PRESS_M = 0.004
PULL_DISTANCE_M = 0.08
REPLACEMENT_LIFT_M = 0.01
WAYPOINT_DURATION_S = 0.04
SETTLE_STEPS = 250


def _quaternion_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to wxyz quaternion."""
    from scipy.spatial.transform import Rotation

    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.roll(quat_xyzw, 1)


def _grasp_pose(
    model,
    data,
    handle_geom_name: str,
    *,
    from_above: bool = False,
) -> Pose:
    """Desired ``g_base`` pose that approaches the chosen drive's handle."""
    import mujoco

    handle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, handle_geom_name)
    if handle_id < 0:
        raise KeyError(f"MuJoCo model is missing geom {handle_geom_name!r}")
    handle_pos = data.geom_xpos[handle_id].copy()

    if from_above:
        # For the free replacement drive on the staging bench, approach from
        # above so the gripper does not collide with the bench surface.
        rotation = np.column_stack(([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]))
    else:
        # Server drive: local +Z is the approach axis and local +Y is between
        # the fingers.  Rotate 90 degrees so fingers close along world Z,
        # grasping the narrow side of the drive.
        rotation = np.column_stack(([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]))

    position_m = handle_pos - GRASP_OFFSET_M * rotation[:, 2]
    if from_above:
        # Keep the gripper/pads just above the bench surface while still
        # reaching the replacement drive.
        position_m = position_m + np.array([0.0, 0.0, 0.01])
    return Pose(
        frame_id="world",
        position_m=position_m,
        quaternion_wxyz=_quaternion_from_matrix(rotation),
    )


def _quintic(progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5


def _execute_path(
    model,
    data,
    robot,
    path,
    viewer,
) -> None:
    """Run a joint-space waypoint path using MuJoCo stepping."""
    import mujoco

    joint_names = robot.read_joint_state().names
    for waypoint in path:
        start_q = robot.arm_position().copy()
        target_q = np.asarray(waypoint.position_rad, dtype=float)
        steps = max(1, int(round(WAYPOINT_DURATION_S / model.opt.timestep)))
        for step_index in range(steps):
            if viewer is not None and not viewer.is_running():
                return
            progress = (step_index + 1) / steps
            commanded = start_q + _quintic(progress) * (target_q - start_q)
            robot.command_joint_positions(JointState(joint_names, commanded))
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, model.opt.timestep - 0.001))


def _hold(
    model,
    data,
    robot,
    viewer,
    seconds: float = 0.5,
    *,
    target_q: np.ndarray | None = None,
) -> None:
    import mujoco

    joint_names = robot.read_joint_state().names
    q = robot.arm_position().copy() if target_q is None else np.asarray(target_q, dtype=float)
    steps = max(1, int(round(seconds / model.opt.timestep)))
    for _ in range(steps):
        if viewer is not None and not viewer.is_running():
            return
        robot.command_joint_positions(JointState(joint_names, q))
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()


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
    """Move from one joint configuration to another with a quintic blend."""
    import mujoco

    start_q = np.asarray(start_q, dtype=float)
    target_q = np.asarray(target_q, dtype=float)
    steps = max(1, int(round(duration_s / model.opt.timestep)))
    for step_index in range(steps):
        if viewer is not None and not viewer.is_running():
            return
        progress = (step_index + 1) / steps
        commanded = start_q + _quintic(progress) * (target_q - start_q)
        robot.command_joint_positions(JointState(joint_names, commanded))
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()


def _wait_for_arm_reach(
    model,
    data,
    robot,
    target_q,
    *,
    viewer,
    tolerance_rad: float = 0.02,
    max_seconds: float = 3.0,
) -> bool:
    """Hold the current joint-position command until the arm tracks close enough."""
    import mujoco

    joint_names = robot.read_joint_state().names
    steps = max(1, int(round(max_seconds / model.opt.timestep)))
    target_q = np.asarray(target_q, dtype=float)
    for step in range(steps):
        if viewer is not None and not viewer.is_running():
            return False
        robot.command_joint_positions(JointState(joint_names, target_q))
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        error = float(np.max(np.abs(robot.arm_position() - target_q)))
        if error < tolerance_rad:
            return True
    return False


def _run_full_replacement(
    model,
    data,
    robot,
    mechanism,
    checker,
    kinematics,
    viewer,
    args,
    home_q,
    grasp_q,
) -> None:
    """Pull bad drive, then pick and insert a good drive from the staging table."""
    joint_names = kinematics.joint_names
    pause = args.pause_seconds

    print("[A] Releasing the pulled-out bad drive ...")
    robot.command_gripper_opening(0.085)
    _hold(model, data, robot, viewer, seconds=pause, target_q=grasp_q)

    print("[B] Returning arm to home ...")
    _move_to_q(
        model,
        data,
        robot,
        grasp_q,
        home_q,
        duration_s=2.0,
        viewer=viewer,
        joint_names=joint_names,
    )
    _hold(model, data, robot, viewer, seconds=pause, target_q=home_q)

    print("[C] Selecting good drive ...")
    if args.good_drive == "replacement_drive":
        good_body = "replacement_drive"
        good_geom = "replacement_drive_bezel"
    else:
        good_body = "replacement_drive_1"
        good_geom = "replacement_drive_1_bezel"
    print(f"    good drive = {good_body}")

    print("[D] Computing good-drive grasp and planning ...")
    good_target_pose = _grasp_pose(model, data, good_geom, from_above=True)
    good_grasp_q = kinematics.inverse(
        good_target_pose,
        JointState(joint_names, home_q),
    ).position_rad
    lower, upper = kinematics.joint_limits_rad
    good_planner = RRTConnectPlanner(
        checker,
        sample_bounds=list(zip(lower, upper)),
        step_size_rad=0.1,
        max_iterations=3000,
        max_connection_steps=300,
        random_seed=args.random_seed,
    )
    good_path = good_planner.plan(
        JointState(joint_names, home_q),
        JointState(joint_names, good_grasp_q),
    )
    print(f"    good path waypoints = {len(good_path)}")
    _execute_path(model, data, robot, good_path, viewer)
    _wait_for_arm_reach(
        model,
        data,
        robot,
        good_grasp_q,
        viewer=viewer,
        tolerance_rad=0.02,
        max_seconds=3.0,
    )

    print("[E] Closing gripper on good drive ...")
    robot.command_gripper_opening(0.0)
    _hold(model, data, robot, viewer, seconds=pause, target_q=good_grasp_q)

    print("[F] Moving good drive to the server bay ...")
    insert_pose = _grasp_pose(model, data, "target_handle")
    insert_q = kinematics.inverse(
        insert_pose,
        JointState(joint_names, good_grasp_q),
    ).position_rad
    _move_to_q(
        model,
        data,
        robot,
        good_grasp_q,
        insert_q,
        duration_s=2.5,
        viewer=viewer,
        joint_names=joint_names,
    )
    _hold(model, data, robot, viewer, seconds=pause, target_q=insert_q)

    print("[G] Releasing good drive ...")
    robot.command_gripper_opening(0.085)
    _hold(model, data, robot, viewer, seconds=pause, target_q=insert_q)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="open an interactive MuJoCo viewer")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--drive",
        choices=("target_drive", "replacement_drive"),
        default="target_drive",
        help="which drive to select in the MuJoCo scene",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="run the full demo: pull bad drive then insert the good drive",
    )
    parser.add_argument(
        "--good-drive",
        choices=("replacement_drive", "replacement_drive_1"),
        default="replacement_drive",
        help="good drive to insert during --replace mode",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=5.0,
        help="seconds to pause at each major pipeline stage",
    )
    args = parser.parse_args()
    if args.pause_seconds < 0.0:
        parser.error("--pause-seconds must be non-negative")

    print("[1/7] Loading MuJoCo server scene ...")
    model, data = load_model()
    reset_home(model, data)
    robot = MujocoRobotAdapter(model, data)
    mechanism = MujocoMechanismAdapter(model, data)
    checker = MujocoCollisionChecker(model, data)

    viewer = None
    if args.show:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data)

    home_q = robot.arm_position().copy()
    _hold(model, data, robot, viewer, seconds=args.pause_seconds, target_q=home_q)

    print("[2/7] Selecting target drive ...")
    if args.drive == "target_drive":
        drive_name = "target_drive_carrier"
        handle_geom_name = "target_handle"
        is_replacement = False
    else:
        drive_name = "replacement_drive"
        handle_geom_name = "replacement_drive_bezel"
        is_replacement = True
    if model.body(drive_name).id < 0:
        raise KeyError(f"model is missing body {drive_name!r}")
    _hold(model, data, robot, viewer, seconds=args.pause_seconds, target_q=home_q)
    print(f"    selected drive = {drive_name} (geom {handle_geom_name})")

    print("[3/7] Computing grasp pose and IK ...")
    kinematics = UR5eKinematics(end_effector_frame="g_base")
    target_pose = _grasp_pose(
        model,
        data,
        handle_geom_name,
        from_above=is_replacement,
    )
    home_q = robot.arm_position().copy()
    grasp_solution = kinematics.inverse(
        target_pose,
        JointState(kinematics.joint_names, home_q),
    )
    grasp_q = grasp_solution.position_rad
    print(f"    grasp quaternion = {grasp_q.round(4).tolist()}")
    _hold(model, data, robot, viewer, seconds=args.pause_seconds, target_q=home_q)

    grasp_state = JointState(kinematics.joint_names, grasp_q)
    if not checker.is_collision_free(grasp_state):
        print("    warning: grasp configuration is in contact; continuing in demo mode")

    print("[4/7] Planning RRT-Connect path in joint space ...")
    lower, upper = kinematics.joint_limits_rad
    planner = RRTConnectPlanner(
        checker,
        sample_bounds=list(zip(lower, upper)),
        step_size_rad=0.1,
        max_iterations=3000,
        max_connection_steps=300,
        random_seed=args.random_seed,
    )
    path = planner.plan(
        JointState(kinematics.joint_names, home_q),
        JointState(kinematics.joint_names, grasp_q),
    )
    print(f"    path waypoints = {len(path)}")

    # The RRT planner may have left data.qpos at a random sampling state.
    # Reset to home so the execution always starts from the initial pose.
    reset_home(model, data)
    home_q = robot.arm_position().copy()
    _hold(model, data, robot, viewer, seconds=args.pause_seconds, target_q=home_q)

    print("[5/7] Executing planned path ...")
    try:
        _execute_path(model, data, robot, path, viewer)
        reached = _wait_for_arm_reach(
            model,
            data,
            robot,
            path[-1].position_rad,
            viewer=viewer,
            tolerance_rad=0.02,
            max_seconds=3.0,
        )
        print(f"    reach_grasp = {'OK' if reached else 'TIMEOUT'}")

        print("[6/7] Closing gripper ...")
        robot.command_gripper_opening(0.0)
        _hold(model, data, robot, viewer, seconds=args.pause_seconds, target_q=path[-1].position_rad)

        if is_replacement:
            print("[7/7] Picking the replacement drive up with the gripper ...")
            pull_pose = Pose(
                frame_id="world",
                position_m=target_pose.position_m + np.array([0.0, 0.0, REPLACEMENT_LIFT_M]),
                quaternion_wxyz=target_pose.quaternion_wxyz,
            )
        else:
            print("[7/7] Pulling the drive out with the gripper ...")
            # Shift the g_base reference away from the server along -X.
            pull_pose = Pose(
                frame_id="world",
                position_m=target_pose.position_m - np.array([PULL_DISTANCE_M, 0.0, 0.0]),
                quaternion_wxyz=target_pose.quaternion_wxyz,
            )

        pull_q = kinematics.inverse(
            pull_pose,
            JointState(kinematics.joint_names, grasp_q),
        ).position_rad
        print(f"    pull_q = {pull_q.round(4).tolist()}")

        # First command the mechanism to release the latch/eject a bit (server
        # drive only), then move the arm to pull/lift.
        if not is_replacement:
            mechanism.command(PULL_EXTRACTION_M, PULL_LATCH_PRESS_M)
            _hold(model, data, robot, viewer, seconds=0.5, target_q=path[-1].position_rad)
            action = "pull"
        else:
            action = "pick"

        _move_to_q(
            model,
            data,
            robot,
            grasp_q,
            pull_q,
            duration_s=2.0,
            viewer=viewer,
            joint_names=kinematics.joint_names,
        )
        _hold(model, data, robot, viewer, seconds=args.pause_seconds, target_q=pull_q)
        print(f"    action = {action}")

        if args.replace and not is_replacement:
            _run_full_replacement(
                model,
                data,
                robot,
                mechanism,
                checker,
                kinematics,
                viewer,
                args,
                home_q,
                grasp_q,
            )
    finally:
        if viewer is not None:
            viewer.close()

    mechanism_positions = mechanism.positions()
    print("PIPELINE COMPLETE")
    print(f"    carrier_extraction = {mechanism_positions[0]:.4f} m")
    print(f"    latch_press        = {mechanism_positions[1]:.4f} m")


if __name__ == "__main__":
    main()
