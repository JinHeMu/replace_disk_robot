"""MuJoCo integration: keyboard nominal pose plus admittance compliance."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from keyboard_servo import ServoDemo
from replace_disk_robot.core import Wrench
from replace_disk_robot.core.rotation import rotation_matrix


def _raw_wrench_for_external_force(app, external_force_base):
    sensor_pose = app._sensor_pose_in_base()
    rotation = rotation_matrix(sensor_pose.quaternion_wxyz)
    return Wrench(
        app.ft.frame_id,
        rotation.T @ (np.asarray(external_force_base) / app.wrench_load_sign),
        np.zeros(3),
    )


def test_admittance_mode_keeps_keyboard_tracking():
    app = ServoDemo(admittance=True)
    initial = app.kinematics.forward(app.robot.read_joint_state())
    app.keys.press("w")
    for _ in range(40):
        app.tick()
    app.keys.release("w")
    for _ in range(40):
        app.tick()

    end = app.kinematics.forward(app.robot.read_joint_state())
    assert app.servo.fault is None
    assert np.linalg.norm(end.position_m - initial.position_m) > 0.001
    assert np.linalg.norm(app.motion.state().offset) < 1e-9


def test_admittance_mode_moves_with_external_force_and_recovers():
    app = ServoDemo(admittance=True)
    start = app.kinematics.forward(app.robot.read_joint_state()).position_m.copy()
    app.ft.read_wrench = lambda: _raw_wrench_for_external_force(app, [0.0, 0.0, -5.0])
    for _ in range(150):
        app.tick()
    loaded = app.kinematics.forward(app.robot.read_joint_state()).position_m.copy()
    offset = app.motion.state().offset.copy()

    app.ft.read_wrench = lambda: Wrench(app.ft.frame_id, np.zeros(3), np.zeros(3))
    for _ in range(150):
        app.tick()
    released = app.kinematics.forward(app.robot.read_joint_state()).position_m.copy()

    assert app.servo.fault is None
    assert (loaded - start)[2] < -0.003
    assert offset[2] < 0.0
    assert np.linalg.norm(released - start) < 0.003
    assert np.linalg.norm(app.motion.state().offset) < 0.001
