#!/usr/bin/env python3
"""Deterministic model, joint tracking, mechanism, and F/T signal smoke test."""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hard_disk_robot.adapters.mujoco import (
    MujocoMechanismAdapter,
    MujocoRobotAdapter,
    MujocoWristFTAdapter,
    load_model,
    reset_home,
)
from hard_disk_robot.safety import ForceLimitGuard


def step(model: mujoco.MjModel, data: mujoco.MjData, count: int) -> None:
    for _ in range(count):
        mujoco.mj_step(model, data)


def main() -> None:
    model, data = load_model()
    reset_home(model, data)
    robot = MujocoRobotAdapter(model, data)
    ft = MujocoWristFTAdapter(model, data)
    mechanism = MujocoMechanismAdapter(model, data)

    home = robot.arm_position()
    robot.command_arm(home)
    robot.command_gripper_opening(0.085)
    mechanism.command(0.0, 0.0)
    step(model, data, 1500)

    assert np.all(np.isfinite(data.qpos))
    tracking_error = float(np.max(np.abs(robot.arm_position() - home)))
    assert tracking_error < 0.025, tracking_error

    driver_qpos = model.jnt_qposadr[model.joint("right_driver_joint").id]
    gripper_before = float(data.qpos[driver_qpos])
    robot.command_gripper_opening(0.025)
    step(model, data, 600)
    gripper_delta = abs(float(data.qpos[driver_qpos]) - gripper_before)
    assert gripper_delta > 0.05, gripper_delta

    mechanism.command(0.035, 0.004)
    step(model, data, 600)
    mechanism_position = mechanism.positions()
    assert np.allclose(mechanism_position, [0.035, 0.004], atol=[0.004, 0.001])

    # Tare includes tool gravity. Apply a known 12 N world-frame load to the gripper.
    ft.tare()
    gripper_body = model.body("g_base").id
    data.xfrc_applied[gripper_body, :3] = [12.0, 0.0, 0.0]
    step(model, data, 200)
    loaded_wrench = ft.wrench()
    measured_force = float(np.linalg.norm(loaded_wrench[:3]))
    assert 10.0 < measured_force < 14.0, measured_force

    guard = ForceLimitGuard(force_limit_n=8.0, torque_limit_nm=2.0)
    requested = robot.arm_position() + np.array([0.05, 0, 0, 0, 0, 0])
    filtered, tripped = guard.filter_arm_target(loaded_wrench, robot.arm_position(), requested)
    assert tripped
    assert np.allclose(filtered, robot.arm_position())

    print("PASS: model contract and finite-state simulation")
    print(f"PASS: arm max tracking error = {tracking_error:.6f} rad")
    print(f"PASS: gripper driver motion = {gripper_delta:.6f} rad")
    print(f"PASS: mechanism [extract, latch] = {mechanism_position.round(6).tolist()} m")
    print(f"PASS: applied 12 N, measured delta = {measured_force:.6f} N")
    print("PASS: force-limit guard blocked the requested arm target")


if __name__ == "__main__":
    main()
