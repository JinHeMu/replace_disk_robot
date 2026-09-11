"""Deterministic contracts for independent key mapping and velocity servo."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import pytest

from replace_disk_robot.core import CartesianJog, JointState, Pose
from replace_disk_robot.control import CartesianServo, KeyControl, ServoConfig
from replace_disk_robot.control.key_control import KEY_AXES


class LinearKinematics:
    joint_names = ('x','y','z','rx','ry','rz')
    dof = 6
    rotation = [1,0,0,0]
    jac = np.eye(6)

    def forward(self, state):
        return Pose('world', state.position_rad[:3], self.rotation)

    def jacobian(self, state):
        return self.jac.copy()


def setup_servo(kin=None, **kwargs):
    kin = kin or LinearKinematics()
    servo = CartesianServo(kin, (-np.ones(6), np.ones(6)), **kwargs)
    state = JointState(kin.joint_names, np.zeros(6))
    servo.reset(state)
    return servo, state


@pytest.mark.parametrize('key', list(KEY_AXES))
def test_key_sign_and_release(key):
    keys = KeyControl()
    keys.press(key.upper())
    command = keys.command()
    linear, angular = KEY_AXES[key]
    np.testing.assert_allclose(command.linear_m_s, np.array(linear)*.01)
    np.testing.assert_allclose(command.angular_rad_s, np.array(angular)*np.deg2rad(5))
    keys.release(key)
    np.testing.assert_array_equal(keys.command().linear_m_s, np.zeros(3))
    np.testing.assert_array_equal(keys.command().angular_rad_s, np.zeros(3))


def test_opposites_repeat_diagonal_and_focus_clear():
    keys = KeyControl()
    for key in ['w','w','s','a','d','r','f','q','e','up','down','left','right']:
        keys.press(key)
    np.testing.assert_array_equal(keys.command().linear_m_s, np.zeros(3))
    np.testing.assert_array_equal(keys.command().angular_rad_s, np.zeros(3))
    keys.clear()
    for key in ['w','a','r','q','left']:
        keys.press(key)
    assert np.isclose(np.linalg.norm(keys.command().linear_m_s), .01)
    assert np.isclose(np.linalg.norm(keys.command().angular_rad_s), np.deg2rad(5))
    keys.clear()
    assert not keys.command().linear_m_s.any()
    assert not keys.command().angular_rad_s.any()


def test_zero_input_and_timeout_hold_last_target_not_measured_drift():
    servo, state = setup_servo()
    servo.submit(CartesianJog('world',[0,0,.01],[0,0,0]), 0.)
    target = servo.update(state,.01,0.)
    assert target.position_rad[2] > 0
    # No new command: after timeout, no additional motion is accumulated.
    expired = servo.update(state,.01,.2)
    np.testing.assert_array_equal(expired.position_rad,target.position_rad)
    assert servo.status == 'command_timeout'
    servo.submit(CartesianJog.zero(), .21)
    np.testing.assert_array_equal(servo.update(state,.01,.21).position_rad,target.position_rad)


def test_named_joint_reordering_and_duplicates():
    servo, state = setup_servo()
    reversed_state = JointState(tuple(reversed(state.names)), np.arange(6)*.01)
    servo.reset(reversed_state)
    np.testing.assert_array_equal(servo.target.position_rad, np.arange(6)[::-1]*.01)
    with pytest.raises(ValueError):
        servo.reset(JointState(('x',)*6, np.zeros(6)))
    with pytest.raises(ValueError):
        servo.reset(JointState(state.names+('x',), np.zeros(7)))


def test_world_translation_and_intrinsic_roll_are_different_frames():
    kin = LinearKinematics()
    kin.rotation = [np.sqrt(.5),0,0,np.sqrt(.5)]  # Tool X now points to world Y.
    servo, state = setup_servo(kin)
    servo.submit(CartesianJog('world',[0,0,.01],[.05,0,0]),0.)
    q = servo.update(state,.01,0.).position_rad
    assert q[2] > 0 and q[4] > 0
    assert abs(q[3]) < 1e-12


def test_singularity_speed_limits_and_joint_limit():
    kin = LinearKinematics()
    kin.jac = np.zeros((6,6))
    servo, state = setup_servo(kin)
    servo.submit(CartesianJog('world',[1,1,1],[1,1,1]),0.)
    np.testing.assert_array_equal(servo.update(state,.01,0.).position_rad,np.zeros(6))
    servo, state = setup_servo(config=ServoConfig(joint_speed_rad_s=.001))
    servo.submit(CartesianJog('world',[1,1,1],[1,1,1]),0.)
    assert np.max(np.abs(servo.update(state,.01,0.).position_rad)) <= .00001+1e-12
    servo.reset(JointState(state.names,[0,0,.999999,0,0,0]))
    servo.submit(CartesianJog('world',[0,0,.01],[0,0,0]),0.)
    servo.update(servo.target,.01,0.)
    assert servo.status == 'joint_limit'


def test_collision_segment_is_checked_and_rejected_without_target_windup():
    class ThinObstacle:
        def __init__(self):
            self.checked = []
        def is_collision_free(self, state):
            x = state.position_rad[0]
            self.checked.append(x)
            return not .00002 < x < .00008
    checker = ThinObstacle()
    servo, state = setup_servo(collision_checker=checker, config=ServoConfig(collision_step_rad=.00001))
    servo.submit(CartesianJog('world',[.01,0,0],[0,0,0]),0.)
    result = servo.update(state,.01,0.)
    np.testing.assert_array_equal(result.position_rad,state.position_rad)
    assert servo.status == 'collision_blocked'
    assert len(checker.checked) > 2


def test_tracking_fault_is_latched_until_reset():
    servo,state = setup_servo()
    moved = JointState(state.names,[.2,0,0,0,0,0])
    servo.update(moved,.01,0.)
    assert servo.fault == 'tracking_error'
    servo.submit(CartesianJog('world',[0,0,.01],[0,0,0]),.01)
    np.testing.assert_array_equal(servo.update(moved,.01,.01).position_rad,moved.position_rad)
    servo.reset(moved)
    assert servo.fault is None


def test_invalid_frames_nonfinite_and_time_rejected():
    with pytest.raises(ValueError):
        CartesianJog('world',[np.nan,0,0],[0,0,0])
    with pytest.raises(ValueError):
        ServoConfig(damping=0)
    servo,state = setup_servo()
    servo.submit(CartesianJog('camera',[0,0,.01],[0,0,0]),0.)
    with pytest.raises(ValueError):
        servo.update(state,.01,0.)
    with pytest.raises(ValueError):
        servo.update(state,.1,.1)
    with pytest.raises(ValueError):
        servo.submit(CartesianJog.zero(),-1.)


def test_control_is_backend_and_window_independent():
    root = Path(__file__).resolve().parents[1]/'src/replace_disk_robot/control'
    source = '\n'.join(p.read_text() for p in root.glob('*.py'))
    for backend in ['import mujoco','import glfw','import pinocchio','import rclpy']:
        assert backend not in source
