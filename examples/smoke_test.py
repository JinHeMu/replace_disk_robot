#!/usr/bin/env python3
"""Five-second init hold and known-load wrist signal/limit-gate smoke test."""
import sys
from pathlib import Path
import mujoco
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from replace_disk_robot.adapters.mujoco import load_model, reset_home, MujocoRobotAdapter, MujocoWristFTAdapter
from replace_disk_robot.safety import ForceLimitGuard


def main():
    model, data = load_model()
    reset_home(model, data)
    robot, ft = MujocoRobotAdapter(model, data), MujocoWristFTAdapter(model, data)
    initial = data.site('drive_center').xpos.copy()
    for _ in range(round(5/model.opt.timestep)):
        mujoco.mj_step(model, data)
        assert data.ncon == 0
        assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.sensordata))
    np.testing.assert_allclose(data.site('drive_center').xpos, initial, atol=1e-6)
    ft.tare()
    data.xfrc_applied[model.body('replacement_drive_held').id,:3] = [2,0,0]
    for _ in range(round(.5/model.opt.timestep)):
        mujoco.mj_step(model, data)
    force = np.linalg.norm(ft.wrench()[:3])
    assert 1.8 < force < 2.2, force
    guard = ForceLimitGuard(force_limit_n=1., torque_limit_nm=1.)
    target, tripped = guard.filter_arm_target(ft.wrench(),robot.arm_position(),robot.arm_position()+.01)
    assert tripped
    np.testing.assert_allclose(target,robot.arm_position())
    print(f'PASS: 5 s init hold, no contacts; 2 N drive load measured {force:.4f} N; limit gate passed')


if __name__ == '__main__':
    main()
