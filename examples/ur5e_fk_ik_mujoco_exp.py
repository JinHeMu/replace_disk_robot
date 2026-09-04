#!/usr/bin/env python3
"""Visualize UR5e forward/inverse kinematics in the MuJoCo scene.

Data flow::

    q_target -> Pinocchio FK -> target Pose -> Pinocchio IK(seed)
             -> q_ik -> MuJoCo joint-position control

The red ``task_target`` marker shows the FK target of ``wrist_3_link``. The arm
first moves to the IK seed and then to the IK solution. This is a kinematics
experiment; it does not validate force/torque dynamics or real hardware.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
CAMERA_NAMES = ("overview_cam", "server_front_cam", "wrist_cam")
TARGET_OFFSET_RAD = np.array([0.20, -0.15, 0.20, -0.15, 0.15, 0.10])
SEED_OFFSET_RAD = np.array([-0.12, 0.18, -0.16, 0.14, -0.12, -0.10])
IK_POSITION_LIMIT_M = 1e-5
IK_ORIENTATION_LIMIT_RAD = 1e-5
MODEL_POSITION_LIMIT_M = 1e-7
MODEL_ORIENTATION_LIMIT_RAD = 1e-6
TRACKING_JOINT_LIMIT_RAD = 0.025
TRACKING_POSITION_LIMIT_M = 0.025
TRACKING_ORIENTATION_LIMIT_RAD = 0.05


def _pinocchio_worker() -> None:
    """Run Pinocchio only and exchange JSON with the MuJoCo parent process."""
    from hard_disk_robot.core import JointState
    from hard_disk_robot.kinematics import UR5eKinematics

    request = json.load(sys.stdin)
    target_q = np.asarray(request["target_q"], dtype=float)
    seed_q = np.asarray(request["seed_q"], dtype=float)

    kinematics = UR5eKinematics(end_effector_frame="wrist_3_link")
    target = kinematics.forward(JointState(kinematics.joint_names, target_q))
    solution = kinematics.inverse(
        target,
        JointState(kinematics.joint_names, seed_q),
    )
    reached = kinematics.forward(solution)

    response = {
        "target_position_m": target.position_m.tolist(),
        "target_quaternion_wxyz": target.quaternion_wxyz.tolist(),
        "solution_q": solution.position_rad.tolist(),
        "ik_position_error_m": float(
            np.linalg.norm(reached.position_m - target.position_m)
        ),
        "ik_orientation_error_rad": _quaternion_angle(
            reached.quaternion_wxyz,
            target.quaternion_wxyz,
        ),
    }
    print(json.dumps(response))


def _run_pinocchio(target_q: np.ndarray, seed_q: np.ndarray) -> dict[str, Any]:
    """Isolate Pinocchio imports so an ABI error cannot terminate the viewer."""
    request = json.dumps({"target_q": target_q.tolist(), "seed_q": seed_q.tolist()})
    commands = (
        [sys.executable, str(Path(__file__).resolve()), "--pinocchio-worker"],
        [sys.executable, "-s", str(Path(__file__).resolve()), "--pinocchio-worker"],
    )
    failures: list[str] = []
    for command in commands:
        completed = subprocess.run(
            command,
            input=request,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if completed.returncode == 0:
            return json.loads(completed.stdout)
        detail = completed.stderr.strip().splitlines()
        failures.append(detail[-1] if detail else f"exit code {completed.returncode}")
    raise RuntimeError("Pinocchio worker failed: " + " | ".join(failures))


def _quaternion_angle(first_wxyz: np.ndarray, second_wxyz: np.ndarray) -> float:
    dot = float(abs(np.dot(first_wxyz, second_wxyz)))
    return float(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def _rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _rotation_error(first: np.ndarray, second: np.ndarray) -> float:
    cosine = (float(np.trace(first.T @ second)) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _quintic_blend(progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Pinocchio UR5e FK/IK in the MuJoCo scene.",
    )
    parser.add_argument(
        "--camera",
        choices=CAMERA_NAMES,
        default="overview_cam",
    )
    parser.add_argument("--move-seconds", type=float, default=2.5)
    parser.add_argument("--hold-seconds", type=float, default=0.75)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--show-sites", action="store_true")
    args = parser.parse_args()
    if args.move_seconds <= 0 or args.hold_seconds < 0:
        parser.error("--move-seconds must be positive and --hold-seconds non-negative")
    return args


def _set_target_marker(
    model: Any,
    data: Any,
    position_m: np.ndarray,
    quaternion_wxyz: np.ndarray,
) -> None:
    mocap_id = int(model.body("task_target").mocapid[0])
    if mocap_id < 0:
        raise RuntimeError("task_target must be a mocap body")
    data.mocap_pos[mocap_id] = position_m
    data.mocap_quat[mocap_id] = quaternion_wxyz


def _step_to_target(
    model: Any,
    data: Any,
    robot: Any,
    start_q: np.ndarray,
    target_q: np.ndarray,
    duration_s: float,
    viewer: Any | None,
) -> bool:
    import mujoco

    from hard_disk_robot.core import JointState

    if duration_s == 0:
        return True
    step_count = max(1, int(np.ceil(duration_s / model.opt.timestep)))
    for step_index in range(step_count):
        if viewer is not None and not viewer.is_running():
            return False
        step_start = time.monotonic()
        progress = (step_index + 1) / step_count
        commanded_q = start_q + _quintic_blend(progress) * (target_q - start_q)
        robot.command_joint_positions(JointState(ARM_JOINT_NAMES, commanded_q))
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
            remaining = model.opt.timestep - (time.monotonic() - step_start)
            if remaining > 0:
                time.sleep(remaining)
    return True


def _hold(
    model: Any,
    data: Any,
    robot: Any,
    target_q: np.ndarray,
    duration_s: float,
    viewer: Any | None,
) -> bool:
    return _step_to_target(
        model,
        data,
        robot,
        target_q,
        target_q,
        duration_s,
        viewer,
    )


def _mujoco_fk_at_configuration(model: Any, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate MuJoCo FK in isolated data without changing the animation."""
    import mujoco

    check_data = mujoco.MjData(model)
    qpos_ids = np.array([model.joint(name).qposadr[0] for name in ARM_JOINT_NAMES])
    check_data.qpos[qpos_ids] = q
    mujoco.mj_forward(model, check_data)
    body_id = model.body("wrist_3_link").id
    return (
        check_data.xpos[body_id].copy(),
        check_data.xmat[body_id].reshape(3, 3).copy(),
    )


def _print_planning_summary(
    home_q: np.ndarray,
    target_q: np.ndarray,
    seed_q: np.ndarray,
    solution_q: np.ndarray,
    result: dict[str, Any],
    model_position_error_m: float,
    model_orientation_error_rad: float,
) -> None:
    print("\nPinocchio FK/IK result")
    print(f"  home_q       = {home_q.round(4).tolist()}")
    print(f"  target_q     = {target_q.round(4).tolist()}")
    print(f"  ik_seed_q    = {seed_q.round(4).tolist()}")
    print(f"  ik_solution_q= {solution_q.round(4).tolist()}")
    print(f"  IK position error    = {result['ik_position_error_m']:.3e} m")
    print(f"  IK orientation error = {result['ik_orientation_error_rad']:.3e} rad")
    print(f"  Pinocchio/MuJoCo FK position error    = {model_position_error_m:.3e} m")
    print(f"  Pinocchio/MuJoCo FK orientation error = {model_orientation_error_rad:.3e} rad")
    passed = (
        result["ik_position_error_m"] <= IK_POSITION_LIMIT_M
        and result["ik_orientation_error_rad"] <= IK_ORIENTATION_LIMIT_RAD
        and model_position_error_m <= MODEL_POSITION_LIMIT_M
        and model_orientation_error_rad <= MODEL_ORIENTATION_LIMIT_RAD
    )
    print(f"  planning/model gate = {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise RuntimeError("Pinocchio IK or Pinocchio/MuJoCo FK gate failed")


def _print_execution_summary(
    model: Any,
    data: Any,
    robot: Any,
    solution_q: np.ndarray,
    target_position_m: np.ndarray,
    target_rotation: np.ndarray,
) -> None:
    wrist_id = model.body("wrist_3_link").id
    joint_error = float(np.max(np.abs(robot.arm_position() - solution_q)))
    position_error = float(
        np.linalg.norm(data.xpos[wrist_id] - target_position_m)
    )
    orientation_error = _rotation_error(
        data.xmat[wrist_id].reshape(3, 3),
        target_rotation,
    )
    print("\nMuJoCo execution result")
    print(f"  maximum joint tracking error = {joint_error:.3e} rad")
    print(f"  TCP position error           = {position_error:.3e} m")
    print(f"  TCP orientation error        = {orientation_error:.3e} rad")
    passed = (
        joint_error <= TRACKING_JOINT_LIMIT_RAD
        and position_error <= TRACKING_POSITION_LIMIT_M
        and orientation_error <= TRACKING_ORIENTATION_LIMIT_RAD
    )
    print(f"  MuJoCo tracking gate = {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise RuntimeError(
            "MuJoCo tracking gate failed; increase motion/hold duration or inspect control"
        )


def _run_animation(
    model: Any,
    data: Any,
    robot: Any,
    home_q: np.ndarray,
    seed_q: np.ndarray,
    solution_q: np.ndarray,
    move_seconds: float,
    hold_seconds: float,
    viewer: Any | None,
) -> bool:
    phases = (
        ("home hold", home_q, home_q, hold_seconds),
        ("move to IK seed", home_q, seed_q, move_seconds),
        ("IK seed hold", seed_q, seed_q, hold_seconds),
        ("move to IK solution", seed_q, solution_q, move_seconds),
        ("IK solution hold", solution_q, solution_q, hold_seconds),
    )
    for name, start_q, target_q, duration_s in phases:
        print(f"phase: {name}")
        if not _step_to_target(
            model,
            data,
            robot,
            start_q,
            target_q,
            duration_s,
            viewer,
        ):
            return False
    return True


def _run_experiment(args: argparse.Namespace) -> None:
    import mujoco

    from hard_disk_robot.adapters.mujoco import (
        MujocoRobotAdapter,
        load_model,
        reset_home,
    )

    model, data = load_model()
    reset_home(model, data)
    robot = MujocoRobotAdapter(model, data)
    robot.command_gripper_opening(0.085)

    home_q = robot.arm_position()
    target_q = home_q + TARGET_OFFSET_RAD
    seed_q = home_q + SEED_OFFSET_RAD
    result = _run_pinocchio(target_q, seed_q)
    solution_q = np.asarray(result["solution_q"], dtype=float)
    target_position_m = np.asarray(result["target_position_m"], dtype=float)
    target_quaternion = np.asarray(result["target_quaternion_wxyz"], dtype=float)
    target_rotation = _rotation_matrix(target_quaternion)
    _set_target_marker(model, data, target_position_m, target_quaternion)

    mujoco_position, mujoco_rotation = _mujoco_fk_at_configuration(model, solution_q)
    model_position_error = float(np.linalg.norm(mujoco_position - target_position_m))
    model_orientation_error = _rotation_error(mujoco_rotation, target_rotation)
    _print_planning_summary(
        home_q,
        target_q,
        seed_q,
        solution_q,
        result,
        model_position_error,
        model_orientation_error,
    )

    if args.headless:
        completed = _run_animation(
            model,
            data,
            robot,
            home_q,
            seed_q,
            solution_q,
            args.move_seconds,
            args.hold_seconds,
            None,
        )
        if completed:
            _print_execution_summary(
                model,
                data,
                robot,
                solution_q,
                target_position_m,
                target_rotation,
            )
        return

    import mujoco.viewer

    camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        args.camera,
    )
    if camera_id < 0:
        raise KeyError(f"scene is missing camera {args.camera!r}")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id
        if args.show_sites:
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        completed = _run_animation(
            model,
            data,
            robot,
            home_q,
            seed_q,
            solution_q,
            args.move_seconds,
            args.hold_seconds,
            viewer,
        )
        if not completed:
            return
        _print_execution_summary(
            model,
            data,
            robot,
            solution_q,
            target_position_m,
            target_rotation,
        )
        print("Animation complete; close the MuJoCo window to exit.")
        while viewer.is_running():
            step_start = time.monotonic()
            _hold(model, data, robot, solution_q, model.opt.timestep, viewer)
            remaining = model.opt.timestep - (time.monotonic() - step_start)
            if remaining > 0:
                time.sleep(remaining)


def main() -> None:
    if "--pinocchio-worker" in sys.argv:
        _pinocchio_worker()
        return
    _run_experiment(_parse_args())


if __name__ == "__main__":
    main()
