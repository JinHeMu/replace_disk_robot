#!/usr/bin/env python3
"""Open the server-drive scene or run it headlessly for a bounded duration."""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hard_disk_robot.adapters.mujoco import (
    MujocoRobotAdapter,
    MujocoWristFTAdapter,
    load_model,
    reset_home,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    model, data = load_model()
    reset_home(model, data)
    robot = MujocoRobotAdapter(model, data)
    sensor = MujocoWristFTAdapter(model, data)
    robot.command_arm(robot.arm_position())
    robot.command_gripper_opening(0.085)

    if args.headless:
        steps = max(1, int(args.seconds / model.opt.timestep))
        for _ in range(steps):
            mujoco.mj_step(model, data)
        print(f"sim_time={data.time:.3f}s wrench={sensor.raw().round(4).tolist()}")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
