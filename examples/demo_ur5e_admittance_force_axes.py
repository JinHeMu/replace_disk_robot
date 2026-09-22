#!/usr/bin/env python3
"""UR5e empty-scene admittance demonstration.

Every 2 seconds a different external force/torque is applied to the tool.  The
six-axis :class:`AdmittanceController` converts that wrench into a compliant
Cartesian offset, and :class:`CartesianServo` tracks the corrected pose using
the MuJoCo position actuators.

Run with the MuJoCo viewer:
    python3 examples/demo_ur5e_admittance_force_axes.py --viewer

Run headless:
    python3 examples/demo_ur5e_admittance_force_axes.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replace_disk_robot.adapters.mujoco import MujocoRobotAdapter
from replace_disk_robot.adapters.mujoco.collision import DEFAULT_ROBOT_BODIES
from replace_disk_robot.control import (
    AdmittanceConfig,
    AdmittanceController,
    CartesianServo,
    ServoConfig,
)
from replace_disk_robot.core import JointState, Pose, Wrench
from replace_disk_robot.core.rotation import quaternion_from_rotation_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UR5E_EMPTY_XML = (
    PROJECT_ROOT
    / "simulation"
    / "mujoco"
    / "models"
    / "ur5e"
    / "ur5e_with_gripper.xml"
)
ROBOT_BODIES_WITHOUT_DRIVE = tuple(
    name for name in DEFAULT_ROBOT_BODIES if name != "replacement_drive_held"
)

UP_Q = np.array(
    [
        2.45232094662072,
        -1.99294546015135,
        2.53041379518691,
        2.60412431855423,
        -0.88152461982582,
        -1.5707963267949,
    ],
    dtype=float,
)

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


class MujocoUR5eKinematics:
    """Minimal KinematicsPort backed by MuJoCo FK/Jacobian for the ``pinch`` site."""

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_names: tuple[str, ...],
        *,
        site_name: str = "pinch",
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.joint_names = tuple(joint_names)
        self.site_name = site_name
        self.site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
        if self.site_id < 0:
            raise KeyError(f"MuJoCo model is missing site {site_name!r}")
        self._joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ]
        if any(joint_id < 0 for joint_id in self._joint_ids):
            raise KeyError("MuJoCo model is missing one or more UR5e joints")
        self._qpos_ids = [
            int(model.jnt_qposadr[joint_id]) for joint_id in self._joint_ids
        ]
        self._dof_ids = [
            int(model.jnt_dofadr[joint_id]) for joint_id in self._joint_ids
        ]

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def _set_joints(self, joints: JointState) -> None:
        if joints.names != self.joint_names:
            raise ValueError(
                f"expected joint names {self.joint_names}, got {joints.names}"
            )
        for qpos_id, value in zip(self._qpos_ids, joints.position_rad):
            self.data.qpos[qpos_id] = float(value)
        mujoco.mj_forward(self.model, self.data)

    def forward(self, joints: JointState) -> Pose:
        self._set_joints(joints)
        site = self.data.site(self.site_id)
        return Pose(
            "world",
            np.asarray(site.xpos, dtype=float).copy(),
            quaternion_from_rotation_matrix(site.xmat.reshape(3, 3)),
        )

    def jacobian(self, joints: JointState) -> np.ndarray:
        self._set_joints(joints)
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(
            self.model,
            self.data,
            jacp,
            jacr,
            self.site_id,
        )
        return np.vstack(
            (
                jacp[:, self._dof_ids],
                jacr[:, self._dof_ids],
            )
        )

    def inverse(self, target: Pose, seed: JointState) -> JointState:
        raise NotImplementedError(
            "this demo uses CartesianServo's forward/Jacobian tracking path"
        )


def _load_empty_ur5e_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = mujoco.MjSpec.from_file(str(UR5E_EMPTY_XML))
    drive_body = spec.body("replacement_drive_held")
    spec.delete(drive_body)
    model = spec.compile()
    model.opt.timestep = 0.001
    return model, mujoco.MjData(model)


def _set_arm_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: tuple[str, ...],
    q: np.ndarray,
) -> None:
    for name, value in zip(joint_names, q):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = float(value)
    mujoco.mj_forward(model, data)


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

    model, data = _load_empty_ur5e_model()
    robot = MujocoRobotAdapter(model, data)
    joint_names = tuple(robot.joint_names)
    _set_arm_qpos(model, data, joint_names, UP_Q)

    # Keep the planning model separate from the executing data.
    kinematics = MujocoUR5eKinematics(model, joint_names)
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in joint_names
    ]
    lower = model.jnt_range[np.asarray(joint_ids, dtype=int), 0].copy()
    upper = model.jnt_range[np.asarray(joint_ids, dtype=int), 1].copy()

    servo = CartesianServo(
        kinematics,
        (lower, upper),
        ServoConfig(
            joint_speed_rad_s=1.0,
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
            frame_id="world",
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

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g_base")
    if body_id < 0:
        raise KeyError("MuJoCo model is missing g_base body")

    control_period_s = float(args.control_period_s)
    steps_per_control = max(1, round(control_period_s / model.opt.timestep))
    phase_duration_s = 2.0
    total_duration_s = phase_duration_s * len(WRENCH_SCHEDULE)
    next_control_s = 0.0
    last_phase = -1
    wall_start_s = time.perf_counter()
    sim_start_s = float(data.time)

    print("Admittance schedule:")
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
                external_wrench = Wrench(
                    "world",
                    np.asarray(force, dtype=float),
                    np.asarray(torque, dtype=float),
                )
                data.xfrc_applied[body_id, :3] = external_wrench.force_n
                data.xfrc_applied[body_id, 3:] = external_wrench.torque_nm

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
        data.xfrc_applied[body_id] = 0.0

    state = admittance.state()
    print(
        "Final admittance offset "
        f"[m, rad]: {np.round(state.offset, 4).tolist()}"
    )
    print(
        "Final TCP position [m]: "
        f"{np.round(kinematics.forward(robot.read_joint_state()).position_m, 4).tolist()}"
    )

    if viewer is not None:
        print("Run finished; close the viewer window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)
        viewer.close()


if __name__ == "__main__":
    main()
