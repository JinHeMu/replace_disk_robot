"""Unit-aware adapters and reusable live plot smoke tests."""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replace_disk_robot.core import CartesianJog, JointState, Pose, TrajectoryPoint, Wrench
from replace_disk_robot.visual import LiveTypePlotter, to_plot_sample


@pytest.mark.parametrize(
    ("value", "titles", "units"),
    [
        (Wrench("sensor", [1, 2, 3], [4, 5, 6]), ("Force", "Torque"),
         ("force [N]", "torque [N·m]")),
        (Pose("world", [1, 2, 3], [1, 0, 0, 0]), ("Position", "Quaternion"),
         ("position [m]", "quaternion")),
        (JointState(("a", "b"), [1, 2]), ("Joint position",), ("position [rad]",)),
        (TrajectoryPoint(0.1, [1, 2]), ("Joint trajectory",), ("position [rad]",)),
        (CartesianJog("world", [1, 2, 3], [4, 5, 6]),
         ("Linear velocity", "TCP angular velocity"),
         ("linear [m/s]", "angular [rad/s]")),
    ],
)
def test_supported_types_keep_groups_and_units(value, titles, units):
    sample = to_plot_sample(value)
    assert tuple(group.title for group in sample.groups) == titles
    assert tuple(group.ylabel for group in sample.groups) == units


def test_wrench_has_noise_resistant_minimum_ranges():
    sample = to_plot_sample(Wrench("sensor", [0, 0, 0], [0, 0, 0]))
    assert tuple(group.minimum_half_range for group in sample.groups) == (1.0, 0.1)


def test_unknown_type_is_rejected():
    with pytest.raises(TypeError, match="unsupported visual type"):
        to_plot_sample(object())


def test_live_plotter_updates_prunes_and_saves(tmp_path):
    plotter = LiveTypePlotter(window_s=1.0, refresh_hz=1000.0, show=False)
    plotter.update(Wrench("sensor", [1, 2, 3], [4, 5, 6]), 10.0)
    plotter.update(Wrench("sensor", [2, 3, 4], [5, 6, 7]), 11.5)
    assert list(plotter._times) == [1.5]
    force_ylim = plotter._axes[0].get_ylim()
    torque_ylim = plotter._axes[1].get_ylim()
    assert force_ylim[0] <= -1 and force_ylim[1] >= 1
    assert torque_ylim[0] <= -0.1 and torque_ylim[1] >= 0.1
    output = tmp_path / "wrench.png"
    plotter.save(output)
    assert output.stat().st_size > 0
    plotter.close()


def test_live_plotter_rejects_changed_stream_structure():
    plotter = LiveTypePlotter(show=False)
    plotter.update(JointState(("a", "b"), [0, 0]), 0.0)
    with pytest.raises(ValueError, match="structure changed"):
        plotter.update(JointState(("a", "c"), [0, 0]), 0.1)
    with pytest.raises(ValueError, match="monotonically"):
        # Original structure is required so the timestamp check is reached.
        plotter.update(JointState(("a", "b"), [0, 0]), -0.1)
    plotter.close()
