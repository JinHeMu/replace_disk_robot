"""Behavioral regressions for delayed execution, not ideal actuator tracking."""
import numpy as np
import pytest

from replace_disk_robot.applications.common.servo_validation import (
    CartesianTestKinematics, DelayedActuator, LegacyMeasuredErrorServo, run_harmonic,
)
from replace_disk_robot.control import CartesianServo, ServoConfig
from replace_disk_robot.core import CartesianJog, JointState, Pose


@pytest.mark.parametrize("delay", [0., .144, .200])
@pytest.mark.parametrize("axis", [0, 5])
@pytest.mark.parametrize("frequency", [.5, 1., 2.])
def test_delayed_pose_trajectory_does_not_amplify_or_add_command_phase_lag(delay, axis, frequency):
    metrics, _ = run_harmonic(delay_s=delay, axis=axis, frequency_hz=frequency)
    assert .98 < metrics["command_amplitude_ratio"] <= 1.02
    assert metrics["actual_amplitude_ratio"] <= 1.02
    assert abs(metrics["command_phase_deg"]) < 1.


def test_old_measured_error_integration_reproduces_one_hz_amplification():
    before, _ = run_harmonic(LegacyMeasuredErrorServo)
    after, _ = run_harmonic()
    assert before["actual_amplitude_ratio"] > 1.5
    assert after["actual_amplitude_ratio"] <= 1.02


def make_servo(**config):
    kin = CartesianTestKinematics()
    state = JointState(kin.joint_names, np.zeros(6))
    servo = CartesianServo(kin, kin.joint_limits_rad, ServoConfig(**config))
    servo.reset(state)
    return servo, state


def test_blocked_actuator_does_not_make_joint_target_integrate_past_fixed_pose():
    servo, actual = make_servo()
    reference = Pose("test_base", [.01, 0, 0], [1, 0, 0, 0])
    commands = []
    for k in range(375):
        servo.submit_pose(reference, k*.008)
        commands.append(servo.update(actual, .008, k*.008).position_rad[0])
    assert servo.fault is None
    assert max(commands) <= .01
    assert commands[-1] > .00998


def test_delayed_step_settles_without_overshoot_and_still_obeys_speed_limit():
    servo, initial = make_servo(pose_linear_speed_m_s=.01)
    plant = DelayedActuator(initial, .144, .008)
    reference = Pose("test_base", [.003, 0, 0], [1, 0, 0, 0])
    trace = []
    for k in range(375):
        servo.submit_pose(reference, k*.008)
        target = servo.update(plant.actual, .008, k*.008)
        trace.append(plant.advance(target).position_rad[0])
    assert min(trace) >= 0
    assert max(trace) <= .003
    assert abs(trace[-1]-.003) < 2e-5
    assert max(np.diff(trace)) <= .01*.008+1e-12


@pytest.mark.parametrize("action", ["reset", "halt", "velocity", "timeout", "submission_gap"])
def test_feedforward_history_is_cleared_across_stops_modes_and_missing_commands(action):
    servo, state = make_servo()
    reference = Pose("test_base", [.001, 0, 0], [1, 0, 0, 0])
    servo.submit_pose(reference, 0.)
    target = servo.update(state, .008, 0.)
    now = .008
    if action == "reset":
        servo.reset(target)
    elif action == "halt":
        servo.halt(target)
        servo.reset(target)
    elif action == "velocity":
        servo.submit(CartesianJog("test_base", [0, 0, 0], [0, 0, 0]), now)
        servo.update(target, .008, now)
        now += .008
    elif action == "timeout":
        held = servo.update(target, .008, .3)
        np.testing.assert_array_equal(held.position_rad, target.position_rad)
        now = .308
    else:
        now = .3
    servo.submit_pose(Pose("test_base", [.01, 0, 0], [1, 0, 0, 0]), now)
    servo.update(target, .008, now)
    np.testing.assert_array_equal(servo.pose_tracking["feedforward_twist_base"], np.zeros(6))


def test_delayed_feedback_is_still_used_for_latched_tracking_fault():
    servo, initial = make_servo(max_tracking_error_rad=.001)
    servo.submit_pose(Pose("test_base", [.01, 0, 0], [1, 0, 0, 0]), 0.)
    for k in range(100):
        target = servo.update(initial, .008, k*.008)
        if servo.fault:
            break
        servo.submit_pose(Pose("test_base", [.01, 0, 0], [1, 0, 0, 0]), (k+1)*.008)
    assert servo.fault == "tracking_error"
    np.testing.assert_array_equal(target.position_rad, initial.position_rad)
    assert servo._previous_pose_reference is None
