#!/usr/bin/env python3
"""Render deterministic overview images for visual regression and documentation."""

import argparse
import sys
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hard_disk_robot.adapters.mujoco import MujocoRobotAdapter, load_model, reset_home


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "simulation" / "mujoco" / "captures")
    args = parser.parse_args()

    model, data = load_model()
    reset_home(model, data)
    robot = MujocoRobotAdapter(model, data)
    robot.command_arm(robot.arm_position())
    robot.command_gripper_opening(0.085)
    for _ in range(800):
        mujoco.mj_step(model, data)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    renderer = mujoco.Renderer(model, width=960, height=720)
    try:
        for camera_name in ("overview_cam", "server_front_cam", "wrist_cam"):
            renderer.update_scene(data, camera=camera_name)
            Image.fromarray(renderer.render()).save(args.output_dir / f"{camera_name}.png")
    finally:
        renderer.close()
    print(f"saved 3 images to {args.output_dir}")


if __name__ == "__main__":
    main()
