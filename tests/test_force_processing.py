"""Contracts for external-wrench transforms, filtering and deadbands."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replace_disk_robot.contact import (
    SensorCompensationConfig,
    SensorWrenchCompensator,
    WrenchProcessor,
    WrenchProcessorConfig,
    external_wrench_at_point,
    external_wrench_at_tcp,
)
from replace_disk_robot.core import Pose, Wrench
from replace_disk_robot.core.rotation import rotation_matrix, rotation_matrix_from_rotation_vector


def test_external_wrench_negates_mujoco_parent_child_load() -> None:
    sensor_pose = Pose("world", [0.2, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    # Sensor reports the load applied by the parent to the child.
    parent_load = Wrench("wrist_ft_site", [0.0, 0.0, -10.0], [0.0, 3.0, 0.0])

    external = external_wrench_at_point(
        parent_load, sensor_pose, [0.5, 0.0, 0.0], load_sign=-1.0
    )

    assert external.frame_id == "world"
    np.testing.assert_allclose(external.force_n, [0.0, 0.0, 10.0])
    np.testing.assert_allclose(external.torque_nm, [0.0, 0.0, 0.0], atol=1e-12)


def test_external_wrench_round_trips_an_arbitrary_external_load() -> None:
    sensor_pose = Pose("base", [0.2, -0.1, 0.3], [1.0, 0.0, 0.0, 0.0])
    tcp = np.array([0.5, 0.2, 0.55])
    external_force = np.array([1.5, -2.0, 3.0])
    external_torque = np.array([0.4, -0.5, 0.6])
    # Invert the transform to build the parent-on-child sensor reading.
    lever = sensor_pose.position_m - tcp
    sensor_force = -external_force
    sensor_torque = -external_torque - np.cross(lever, sensor_force)
    sensor = Wrench("sensor", sensor_force, sensor_torque)

    recovered = external_wrench_at_point(sensor, sensor_pose, tcp, load_sign=-1.0)

    np.testing.assert_allclose(recovered.force_n, external_force, atol=1e-12)
    np.testing.assert_allclose(recovered.torque_nm, external_torque, atol=1e-12)


def test_external_wrench_rotates_sensor_frame_into_output_frame() -> None:
    rotation = rotation_matrix_from_rotation_vector([0.0, 0.0, np.pi / 2.0])
    sensor_pose = Pose(
        "base", [0.0, 0.0, 0.0], [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    )
    sensor = Wrench("sensor", [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    external = external_wrench_at_point(sensor, sensor_pose, [0.0, 0.0, 0.0])

    np.testing.assert_allclose(rotation @ sensor.force_n, [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(external.force_n, [-0.0, -1.0, 0.0], atol=1e-12)


def test_processor_low_pass_and_deadband() -> None:
    pose = Pose("world", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    processor = WrenchProcessor(WrenchProcessorConfig(
        frame_id="world",
        load_sign=-1.0,
        filter_alpha=0.5,
        force_deadband_n=0.1,
        torque_deadband_nm=0.05,
    ))
    raw = Wrench("sensor", [-1.0, 0.0, 0.05], [0.0, 0.0, 0.0])

    first = processor.update(raw, pose, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(first.force_n, [0.5, 0.0, 0.0], atol=1e-12)
    second = processor.update(raw, pose, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(second.force_n, [0.75, 0.0, 0.0], atol=1e-12)

    processor.reset()
    below = processor.update(
        Wrench("sensor", [-0.05, 0.0, 0.0], [0.0, 0.0, 0.0]),
        pose,
        [0.0, 0.0, 0.0],
    )
    np.testing.assert_array_equal(below.force_n, np.zeros(3))


def test_processor_rejects_frame_mismatch_and_bad_configuration() -> None:
    processor = WrenchProcessor(WrenchProcessorConfig(frame_id="world"))
    with pytest.raises(ValueError):
        processor.update(
            Wrench("sensor", [0.0] * 3, [0.0] * 3),
            Pose("tool", [0.0] * 3, [1.0, 0.0, 0.0, 0.0]),
            [0.0] * 3,
        )
    with pytest.raises(ValueError):
        WrenchProcessorConfig(frame_id="world", load_sign=0.0)
    with pytest.raises(ValueError):
        WrenchProcessorConfig(frame_id="world", filter_alpha=0.0)
    with pytest.raises(ValueError):
        WrenchProcessorConfig(frame_id="world", force_deadband_n=-1.0)
    with pytest.raises(ValueError):
        external_wrench_at_point(
            Wrench("sensor", [0.0] * 3, [0.0] * 3),
            Pose("world", [0.0] * 3, [1.0, 0.0, 0.0, 0.0]),
            [0.0, 0.0],
        )



def test_sensor_gravity_and_bias_are_removed_before_frame_transform() -> None:
    config = SensorCompensationConfig(
        frame_id="sensor", gravity_frame_id="base", load_sign=-1.0,
        payload_mass_kg=2.0, payload_com_sensor_m=[0.1, 0.0, 0.0],
    )
    processor = SensorWrenchCompensator(config)
    upright = Pose("base", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    tipped = Pose("base", [0.0, 0.0, 0.0], [np.sqrt(.5), 0.0, np.sqrt(.5), 0.0])
    bias = np.array([.2, -.1, .3, .01, -.02, .03])

    def reading(pose, contact=np.zeros(6)):
        gravity_sensor = rotation_matrix(pose.quaternion_wxyz).T @ config.gravity_m_s2
        force = -2.0 * gravity_sensor
        gravity = np.r_[force, np.cross([.1, 0.0, 0.0], force)]
        value = gravity + bias + contact
        return Wrench("sensor", value[:3], value[3:])

    processor.tare(reading(upright), upright)
    np.testing.assert_allclose(processor.bias_sensor, bias, atol=1e-12)
    np.testing.assert_allclose(
        processor.compensate(reading(tipped), tipped).as_vector(),
        np.zeros(6), atol=1e-12,
    )
    contact = np.array([1.0, 2.0, 3.0, .1, .2, .3])
    np.testing.assert_allclose(
        processor.compensate(reading(tipped, contact), tipped).as_vector(),
        contact, atol=1e-12,
    )
    with pytest.raises(ValueError):
        processor.compensate(reading(tipped), upright.__class__(
            "world", upright.position_m, upright.quaternion_wxyz
        ))


def test_tcp_shift_rotation_and_filter_follow_moving_tcp() -> None:
    sensor = Wrench("sensor", [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    sensor_pose = Pose("base", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    tcp = Pose("base", [0.0, 0.1, 0.0], [np.sqrt(.5), 0.0, 0.0, np.sqrt(.5)])
    transformed = external_wrench_at_tcp(sensor, sensor_pose, tcp, "tcp")
    np.testing.assert_allclose(transformed.force_n, [0.0, -2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(transformed.torque_nm, [0.0, 0.0, .2], atol=1e-12)

    processor = WrenchProcessor(WrenchProcessorConfig(
        frame_id="tcp", filter_alpha=.5, load_sign=-1.0,
    ))
    first = processor.update_tcp(sensor, sensor_pose, tcp)
    np.testing.assert_allclose(first.force_n, [0.0, -1.0, 0.0], atol=1e-12)
    rotated = Pose("base", [0.0, 0.1, 0.0], [1.0, 0.0, 0.0, 0.0])
    second = processor.update_tcp(sensor, sensor_pose, rotated)
    np.testing.assert_allclose(second.force_n, [1.5, 0.0, 0.0], atol=1e-12)



def test_tcp_filter_transports_filtered_moment_to_new_tcp_origin() -> None:
    processor = WrenchProcessor(WrenchProcessorConfig(
        frame_id="tcp", filter_alpha=.5, load_sign=-1.0,
    ))
    sensor_pose = Pose("base", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    sensor = Wrench("sensor", [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    old_tcp = Pose("base", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    new_tcp = Pose("base", [0.0, .1, 0.0], [1.0, 0.0, 0.0, 0.0])
    processor.update_tcp(sensor, sensor_pose, old_tcp)
    second = processor.update_tcp(sensor, sensor_pose, new_tcp)
    np.testing.assert_allclose(second.force_n, [1.5, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(second.torque_nm, [0.0, 0.0, .15], atol=1e-12)
