#!/usr/bin/env python3
"""Display the complete UR5e server-drive scene in MuJoCo.

The example uses the same named robot adapter as the rest of the project. It
does not command a trajectory: the arm holds the model's ``home`` keyframe and
the gripper is opened for visual inspection.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hard_disk_robot.adapters.mujoco import (  # noqa: E402
    MujocoRobotAdapter,
    load_model,
    reset_home,
    scene_path,
)


CAMERA_NAMES = ("overview_cam", "server_front_cam", "wrist_cam")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display the complete UR5e hard-disk replacement scene.",
    )
    parser.add_argument(
        "--camera",
        choices=CAMERA_NAMES,
        default="overview_cam",
        help="fixed camera selected when the interactive viewer opens",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="headless duration; interactive mode runs until the window closes",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="step the same scene without opening a window",
    )
    parser.add_argument(
        "--show-sites",
        action="store_true",
        help="show MuJoCo site coordinate frames in the interactive viewer",
    )
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    return args


def _prepare_scene() -> tuple[mujoco.MjModel, mujoco.MjData, MujocoRobotAdapter]:
    model, data = load_model()
    reset_home(model, data)
    robot = MujocoRobotAdapter(model, data)
    robot.command_joint_positions(robot.read_joint_state())
    robot.command_gripper_opening(0.085)
    return model, data, robot


def _print_scene_summary(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot: MujocoRobotAdapter,
) -> None:
    print(f"scene={scene_path()}")
    print(f"sim_time={data.time:.3f}s")
    print(f"arm_joints={robot.read_joint_state().names}")
    print(f"arm_position_rad={robot.arm_position().round(4).tolist()}")
    print(f"bodies={model.nbody} geoms={model.ngeom} sensors={model.nsensor}")


def _run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot: MujocoRobotAdapter,
    seconds: float,
) -> None:
    step_count = max(1, int(seconds / model.opt.timestep))
    for _ in range(step_count):
        mujoco.mj_step(model, data)
    _print_scene_summary(model, data, robot)


def _run_interactive(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    show_sites: bool,
) -> None:
    import mujoco.viewer

    camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        camera_name,
    )
    if camera_id < 0:
        raise KeyError(f"scene is missing camera {camera_name!r}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id
        if show_sites:
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        print("Close the MuJoCo window to stop the example.")
        while viewer.is_running():
            step_start = time.monotonic()
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.monotonic() - step_start)
            if remaining > 0:
                time.sleep(remaining)


def main() -> None:
    args = _parse_args()
    model, data, robot = _prepare_scene()
    if args.headless:
        _run_headless(model, data, robot, args.seconds)
        return
    _print_scene_summary(model, data, robot)
    _run_interactive(model, data, args.camera, args.show_sites)


if __name__ == "__main__":
    main()
