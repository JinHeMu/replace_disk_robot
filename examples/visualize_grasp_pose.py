#!/usr/bin/env python3
"""Render the UR5e grasp pose for the server drive pipeline.

Outputs two reference images:
    simulation/mujoco/captures/grasp_pose_home.png
    simulation/mujoco/captures/grasp_pose_target.png

The second image places the arm at the exact IK grasp configuration (by setting
MuJoCo qpos directly, not through the joint-position controller) and moves the
``task_target`` marker to the desired g_base location.  This lets us visually
compare the reference pose with an executed (dynamic) pipeline run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hard_disk_robot.adapters.mujoco import load_model, reset_home  # noqa: E402
from hard_disk_robot.adapters.mujoco.interfaces import ARM_JOINTS  # noqa: E402
from hard_disk_robot.core import JointState, Pose  # noqa: E402
from hard_disk_robot.kinematics.ur5e import UR5eKinematics  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "simulation" / "mujoco" / "captures"
GRASP_OFFSET_M = 0.145


def _quaternion_from_matrix(rotation: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.roll(quat_xyzw, 1)


def _grasp_pose(model, data) -> Pose:
    import mujoco

    handle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_handle")
    handle_pos = data.geom_xpos[handle_id].copy()
    rotation = np.column_stack(([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]))
    return Pose(
        frame_id="world",
        position_m=handle_pos - GRASP_OFFSET_M * rotation[:, 2],
        quaternion_wxyz=_quaternion_from_matrix(rotation),
    )


def _set_target_marker(model, data, pose: Pose, visible: bool = True) -> None:
    mocap_id = int(model.body("task_target").mocapid[0])
    if mocap_id < 0:
        raise RuntimeError("task_target must be a mocap body")
    data.mocap_pos[mocap_id] = pose.position_m
    data.mocap_quat[mocap_id] = pose.quaternion_wxyz
    # Show/hide the marker by moving it far away if invisible.
    if not visible:
        data.mocap_pos[mocap_id][:] = [100.0, 100.0, 100.0]


def _render(model, data, camera: str, output: Path) -> None:
    import mujoco
    from PIL import Image

    renderer = mujoco.Renderer(model, width=960, height=720)
    try:
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(output)
    finally:
        renderer.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading scene ...")
    model, data = load_model()
    reset_home(model, data)

    print("Computing grasp IK ...")
    kinematics = UR5eKinematics(end_effector_frame="g_base")
    target_pose = _grasp_pose(model, data)
    grasp_q = kinematics.inverse(
        target_pose,
        JointState(kinematics.joint_names, model.key_qpos[0][:6]),
    ).position_rad
    print(f"grasp_q = {grasp_q.round(4).tolist()}")

    # Reference 1: arm at home, marker at the desired g_base target.
    reset_home(model, data)
    _set_target_marker(model, data, target_pose, visible=True)
    import mujoco

    mujoco.mj_forward(model, data)
    home_path = OUTPUT_DIR / "grasp_pose_home.png"
    _render(model, data, "overview_cam", home_path)
    print(f"Saved {home_path}")

    # Reference 2: arm directly at the exact IK grasp configuration.
    qpos_ids = [model.joint(name).qposadr[0] for name in ARM_JOINTS]
    data.qpos[qpos_ids] = grasp_q
    _set_target_marker(model, data, target_pose, visible=True)
    mujoco.mj_forward(model, data)
    gain_path = OUTPUT_DIR / "grasp_pose_target.png"
    _render(model, data, "overview_cam", gain_path)
    print(f"Saved {gain_path}")

    # Additional close-up front view at grasp configuration.
    front_path = OUTPUT_DIR / "grasp_pose_target_front.png"
    _render(model, data, "server_front_cam", front_path)
    print(f"Saved {front_path}")


if __name__ == "__main__":
    main()
