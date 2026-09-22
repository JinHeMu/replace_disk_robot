#!/usr/bin/env python3
"""JAKA empty-scene admittance demonstration on the repository MuJoCo model.

This mirrors ``demo_ur5e_admittance_force_axes.py`` but uses the existing JAKA
MuJoCo scene and repository adapters:

* ``load_model("jaka")`` / ``reset_keyframe(..., "up")``
* ``MujocoRobotAdapter`` with JAKA joint and actuator names
* ``JakaKinematics`` for FK/Jacobian
* ``AdmittanceController``
* ``CartesianServo``

Every 2 seconds a different external force/torque is applied to the tool in the
JAKA base frame.  The admittance controller produces a compliant Cartesian
offset and the Cartesian servo tracks it with the JAKA position actuators.

Run with the MuJoCo viewer (activate the isolated conda env first):
    conda activate /home/a/replace_disk_robot/.conda_envs/replace_disk_robot
    python examples/demo_jaka_admittance_force_axes.py --viewer

Run headless:
    python examples/demo_jaka_admittance_force_axes.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replace_disk_robot.adapters.mujoco import (
    MujocoRobotAdapter,
    load_model,
    reset_keyframe,
)
from replace_disk_robot.control import (
    AdmittanceConfig,
    AdmittanceController,
    CartesianServo,
    ServoConfig,
)
from replace_disk_robot.core import JointState, Pose, Wrench
from replace_disk_robot.kinematics.jaka import JAKA_JOINT_NAMES, JakaKinematics


JAKA_ACTUATORS = tuple(f"joint_{index}_servo" for index in range(1, 7))
TOOL_BODY_NAME = "tool0_and_camera_link"
BASE_BODY_NAME = "jaka_base_link"

WRENCH_SCHEDULE: tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...] = (
    ("settle", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ("+X force", (5.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ("+Y force", (0.0, 5.0, 0.0), (0.0, 0.0, 0.0)),
    ("+Z force", (0.0, 0.0, 5.0), (0.0, 0.0, 0.0)),
    ("+X torque", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ("+Y torque", (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ("+Z torque", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("release", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
)


def _viewer_preflight() -> tuple[bool, str | None]:
    try:
        import glfw
    except Exception as exc:  # noqa: BLE001 - optional GUI dependency
        return False, str(exc)
    try:
        if not glfw.init():
            return False, "glfw.init() failed"
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(64, 64, "mujoco-viewer-preflight", None, None)
        if not window:
            glfw.terminate()
            return False, "could not create a GLFW/OpenGL window"
        glfw.make_context_current(window)
        glfw.destroy_window(window)
        glfw.terminate()
        return True, None
    except Exception as exc:  # noqa: BLE001 - GUI backend error types vary
        try:
            glfw.terminate()
        except Exception:
            pass
        return False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open the MuJoCo passive viewer while running",
    )
    parser.add_argument("--control-period-s", type=float, default=0.01)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="real-time factor: 1.0 means 16 s of wall-clock time for the schedule",
    )
    parser.add_argument(
        "--no-realtime",
        dest="realtime",
        action="store_false",
        default=True,
        help="run as fast as possible instead of pacing to wall-clock time",
    )
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0.0:
        raise ValueError("--speed must be finite and positive")

    model, data = load_model("jaka")
    reset_keyframe(model, data, "up")
    robot = MujocoRobotAdapter(
        model,
        data,
        compensate_bias=True,
        joint_names=JAKA_JOINT_NAMES,
        actuator_names=JAKA_ACTUATORS,
        gripper_actuator_name=None,
    )
    joint_names = tuple(robot.joint_names)

    kinematics = JakaKinematics()
    lower, upper = kinematics.joint_limits_rad

    servo = CartesianServo(
        kinematics,
        (lower, upper),
        ServoConfig(
            joint_speed_rad_s=0.8,
            max_tracking_error_rad=10.0,
            pose_linear_speed_m_s=0.10,
            pose_angular_speed_rad_s=np.deg2rad(30.0),
            max_dt_s=0.05,
        ),
    )
    initial_joints = robot.read_joint_state()
    servo.reset(initial_joints)
    nominal_pose = kinematics.forward(initial_joints)

    admittance = AdmittanceController(
        AdmittanceConfig(
            frame_id=kinematics.base_frame,
            mass=[1.0, 1.0, 1.0, 0.05, 0.05, 0.05],
            damping=[20.0, 20.0, 20.0, 2.0, 2.0, 2.0],
            stiffness=[50.0, 50.0, 50.0, 5.0, 5.0, 5.0],
            max_velocity=[0.2, 0.2, 0.2, 0.5, 0.5, 0.5],
            max_dt_s=0.05,
        )
    )
    admittance.reset(nominal_pose)

    viewer = None
    if args.viewer:
        viewer_ok, viewer_error = _viewer_preflight()
        if not viewer_ok:
            print(f"MuJoCo viewer unavailable ({viewer_error}); running headless.")
        else:
            try:
                from mujoco import viewer as mujoco_viewer

                viewer = mujoco_viewer.launch_passive(model, data)
                viewer.sync()
                print("MuJoCo viewer opened. Close the viewer window to stop.")
            except Exception as exc:  # noqa: BLE001 - GUI errors vary
                print(f"Could not open MuJoCo viewer ({exc}); running headless.")

    tool_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        TOOL_BODY_NAME,
    )
    base_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        BASE_BODY_NAME,
    )
    if tool_body_id < 0 or base_body_id < 0:
        raise KeyError("JAKA model is missing the tool or base body")

    # The JAKA base is fixed in this demo; this converts base-frame schedule
    # wrenches into world-frame MuJoCo applied forces.
    rotation_world_base = data.body(base_body_id).xmat.reshape(3, 3).copy()

    control_period_s = float(args.control_period_s)
    steps_per_control = max(1, round(control_period_s / model.opt.timestep))
    phase_duration_s = 2.0
    total_duration_s = phase_duration_s * len(WRENCH_SCHEDULE)
    next_control_s = 0.0
    last_phase = -1
    wall_start_s = time.perf_counter()
    sim_start_s = float(data.time)

    print("JAKA admittance schedule (base-frame wrench):")
    for index, (label, force, torque) in enumerate(WRENCH_SCHEDULE):
        print(
            f"  {index * phase_duration_s:4.1f}s - "
            f"{(index + 1) * phase_duration_s:4.1f}s: "
            f"{label:10s} F={force} N, T={torque} N*m"
        )

    try:
        while data.time < total_duration_s:
            phase = int(data.time // phase_duration_s)
            phase = min(phase, len(WRENCH_SCHEDULE) - 1)
            if phase != last_phase:
                if last_phase >= 0:
                    previous_label = WRENCH_SCHEDULE[last_phase][0]
                    print(
                        f"  end of {previous_label:10s}: "
                        f"offset={np.round(admittance.state().offset, 4).tolist()}"
                    )
                label, force, torque = WRENCH_SCHEDULE[phase]
                print(f"[{data.time:5.2f}s] {label}")
                last_phase = phase

            if data.time + 1e-12 >= next_control_s:
                label, force, torque = WRENCH_SCHEDULE[phase]
                force_base = np.asarray(force, dtype=float)
                torque_base = np.asarray(torque, dtype=float)
                external_wrench = Wrench(
                    kinematics.base_frame,
                    force_base,
                    torque_base,
                )

                # Apply the equivalent wrench in the MuJoCo world frame.
                force_world = rotation_world_base @ force_base
                torque_world = rotation_world_base @ torque_base
                data.xfrc_applied[tool_body_id, :3] = force_world
                data.xfrc_applied[tool_body_id, 3:] = torque_world

                actual = robot.read_joint_state()
                corrected = admittance.update(
                    nominal_pose,
                    external_wrench,
                    control_period_s,
                )
                servo.submit_pose(corrected, data.time)
                command = servo.update(actual, control_period_s, data.time)
                robot.command_joint_positions(command)
                next_control_s += control_period_s

                if viewer is not None and not viewer.is_running():
                    print("Viewer closed; stopping.")
                    break

            for _ in range(steps_per_control):
                mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()

            if args.realtime:
                target_wall_s = wall_start_s + (data.time - sim_start_s) / args.speed
                delay_s = target_wall_s - time.perf_counter()
                if delay_s > 0.0:
                    time.sleep(delay_s)
    finally:
        data.xfrc_applied[tool_body_id] = 0.0

    state = admittance.state()
    print(
        "Final admittance offset "
        f"[m, rad]: {np.round(state.offset, 4).tolist()}"
    )
    final_pose = kinematics.forward(robot.read_joint_state())
    print(
        "Final TCP pose in jaka_base_link: "
        f"p={np.round(final_pose.position_m, 4).tolist()}"
    )

    if viewer is not None:
        print("Run finished; close the viewer window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)
        viewer.close()


if __name__ == "__main__":
    main()
