"""Contracts for external-wrench transforms, filtering and deadbands."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replace_disk_robot.contact import (
    WrenchProcessor,
    WrenchProcessorConfig,
    external_wrench_at_point,
)
from replace_disk_robot.core import Pose, Wrench
from replace_disk_robot.core.rotation import rotation_matrix_from_rotation_vector


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
