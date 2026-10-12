"""Reusable joystick state/velocity contracts, without JAKA or SDL devices."""
from dataclasses import replace

import numpy as np
import pytest

from replace_disk_robot.control import JoyControl, JoySample


def test_proportional_input_deadzone_precision_and_norm_limits():
    joy = JoyControl(.01, .1, base_frame="command")
    sample = JoySample(1., (.575, 0., 0., 0., 0., 0.), frozenset({"lb"}))
    joy.update(sample)
    command = joy.command(1.)
    assert command.base_frame == "command"
    np.testing.assert_allclose(command.linear_m_s, [0., -.005, 0.])
    joy.update(replace(sample, buttons=frozenset({"lb", "rb"})))
    np.testing.assert_allclose(joy.command(1.).linear_m_s, [0., -.00125, 0.])
    joy.update(replace(sample, axes=(.1, -.1, .1, -.1, .1, .1)))
    np.testing.assert_array_equal(joy.command(1.).linear_m_s, np.zeros(3))
    np.testing.assert_array_equal(joy.command(1.).angular_rad_s, np.zeros(3))
    joy.update(replace(sample, axes=(1., 1., 1., 1., 0., 1.), buttons=frozenset({"lb", "dpleft"})))
    command = joy.command(1.)
    assert np.linalg.norm(command.linear_m_s) == pytest.approx(.01)
    assert np.linalg.norm(command.angular_rad_s) == pytest.approx(.1)


@pytest.mark.parametrize("sample", [
    JoySample(0., (1., 0., 0., 0., 0., 0.), frozenset({"lb"})),
    JoySample(2., buttons=frozenset({"lb"})),
    JoySample(1., (np.nan, 0., 0., 0., 0., 0.), frozenset({"lb"})),
    JoySample(1., (1.1, 0., 0., 0., 0., 0.), frozenset({"lb"})),
    JoySample(1., (0., 0., 0., 0., -1., 0.), frozenset({"lb"})),
    JoySample(1., connected=False), JoySample(1., axes=(0.,)),
    JoySample(1., (1., 0., 0., 0., 0., 0.)),
])
def test_unusable_input_never_yields_motion(sample):
    joy = JoyControl(.01, .1)
    joy.update(sample)
    command = joy.command(1.)
    np.testing.assert_array_equal(command.linear_m_s, np.zeros(3))
    np.testing.assert_array_equal(command.angular_rad_s, np.zeros(3))


def test_three_independent_translations_and_custom_directions():
    joy = JoyControl(.01, .1)
    velocities = []
    for axis in (0, 1, 5):
        axes = np.zeros(6)
        axes[axis] = 1.
        joy.update(JoySample(1., tuple(axes), frozenset({"lb"})))
        velocities.append(joy.command(1.).linear_m_s)
    assert np.linalg.matrix_rank(velocities) == 3
    joy.update(JoySample(1., (1., 0., 0., 0., 0., 0.), frozenset({"lb"})))
    np.testing.assert_allclose(joy.command(1., translation_axes=np.eye(3)).linear_m_s, [.01, 0., 0.])


def test_clear_and_center_replace_previous_state():
    joy = JoyControl()
    sample = JoySample(1., (1., 0., 0., 0., 0., 0.), frozenset({"lb"}))
    joy.update(sample)
    assert not joy.neutral() and joy.valid(1.)
    joy.update(replace(sample, axes=(0.,)*6))
    assert joy.neutral()
    np.testing.assert_array_equal(joy.command(1.).linear_m_s, np.zeros(3))
    joy.update(sample)
    joy.clear()
    assert not joy.valid(1.)
    assert joy.sample.buttons == frozenset()
    np.testing.assert_array_equal(joy.command(1.).linear_m_s, np.zeros(3))


def test_control_import_does_not_load_device_or_robot_runtimes():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c", "import sys; "
        "from replace_disk_robot.control import JoyControl; "
        "assert not any(m in sys.modules for m in ('pygame', 'mujoco', 'pinocchio')); "
        "assert not any(m.startswith('replace_disk_robot.applications') for m in sys.modules)"],
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
