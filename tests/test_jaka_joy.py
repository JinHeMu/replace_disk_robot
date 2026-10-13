"""Controller mapping and lifecycle regressions; never connect to physical JAKA."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from replace_disk_robot.control import JoyControl, JoySample
from replace_disk_robot.applications.jaka_joy.joy_node import parse_args
from replace_disk_robot.core import Pose
from replace_disk_robot.core.rotation import quaternion_from_rotation_vector, rotation_matrix


def test_insertion_always_uses_tool_z_and_base_rotation_is_intrinsic_tcp():
    from types import SimpleNamespace
    from replace_disk_robot.applications.jaka_joy.joy_node import JakaJoyServo
    pose = Pose("base", [0., 0., 0.], quaternion_from_rotation_vector([0., np.pi/2, 0.]))
    for frame in ("base", "tool"):
        app = object.__new__(JakaJoyServo)
        app.joy = JoyControl(.01, .1, base_frame=frame)
        app.joy.update(JoySample(1., (0., 0., 0., 0., 0., 1.), frozenset({"lb", "dpleft"})))
        app._active, app._joy_now, app.base_frame = True, 1., "base"
        app.servo = SimpleNamespace(fault=None)
        app.args = SimpleNamespace(command_frame=frame)
        command = app._jog_to_servo(pose)
        np.testing.assert_allclose(command.linear_m_s, [.01, 0., 0.], atol=1e-12)
        if frame == "base":
            np.testing.assert_allclose(rotation_matrix(pose.quaternion_wxyz) @ command.angular_rad_s,
                                       [0., 0., .1], atol=1e-12)


def test_entry_help_and_config_override_without_robot(tmp_path):
    entry = Path(__file__).resolve().parents[1] / "src/replace_disk_robot/applications/jaka_joy/jaka_joy_node.py"
    result = subprocess.run([sys.executable, str(entry), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "--backend" in result.stdout and "--validate" in result.stdout
    path = tmp_path / "joy.json"
    path.write_text(json.dumps({"joy_deadzone": .2, "linear_speed": .003, "backend": "sim"}))
    args = parse_args(["--config", str(path), "--joy-deadzone", ".1"])
    assert args.joy_deadzone == .1 and args.linear_speed == .003
    with pytest.raises(SystemExit):
        parse_args(["--backend", "real", "--validate"])


def test_startup_with_lb_pressed_cannot_arm():
    from replace_disk_robot.applications.jaka_joy.joy_simulation import make_sim_app
    app = make_sim_app(parse_args([]))
    try:
        app.initialize()
        sample = JoySample(0., (1., 0., 0., 0., 0., 0.), frozenset({"lb"}))
        app.apply_sample(sample, 0.)
        app.tick(0., app.client.dt)
        assert not app._active
        assert app.servo.fault == "stopped"
        assert not hasattr(app, "keys")
    finally:
        app.shutdown()


def test_dry_run_never_enables_or_sends_even_with_lb():
    from replace_disk_robot.applications.jaka_joy.joy_simulation import make_sim_app
    app = make_sim_app(parse_args(["--dry-run"]))
    try:
        app.initialize()
        app.apply_sample(JoySample(0.), 0.)
        app.apply_sample(JoySample(.008, (1., 0., 0., 0., 0., 0.), frozenset({"lb"})), .008)
        app.tick(.008, .008)
        assert not app.client.enabled
        assert app.client.command_count == 0
    finally:
        app.shutdown()


def test_joy_logging_contains_input_and_source_snapshots(tmp_path):
    from replace_disk_robot.applications.jaka_joy.joy_simulation import make_sim_app
    directory = tmp_path / "session"
    app = make_sim_app(parse_args(["--log-dir", str(directory)]))
    try:
        app.initialize()
        app.apply_sample(JoySample(0.), 0.)
        app.tick(0., app.client.dt)
        app.apply_sample(JoySample(.008, (.575, 0., 0., 0., 0., 0.), frozenset({"lb"})), .008)
        app.tick(.008, .008)
    finally:
        app.shutdown()
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["complete"] and summary["written_samples"] == 2
    samples = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
    assert samples[-1]["joy"]["sample"]["buttons"] == ["lb"]
    assert samples[-1]["joy"]["active"]
    metadata = json.loads((directory / "metadata.json").read_text())
    assert any(path.endswith("control/joy_control.py") for path in metadata["sources"])
    assert any(path.endswith("adapters/joy.py") for path in metadata["sources"])
    assert any(path.endswith("applications/common/node.py") for path in metadata["sources"])
    assert any(path.endswith("applications/common/reference.py") for path in metadata["sources"])
    assert any(path.endswith("applications/jaka_joy/joy_node.py") for path in metadata["sources"])
    assert any(path.endswith("applications/jaka_joy/config.py") for path in metadata["sources"])
    assert not any("/applications/jaka_keyboard/" in path for path in metadata["sources"])


def test_reader_reports_disconnect_before_reopening_same_controller():
    from types import SimpleNamespace
    from replace_disk_robot.adapters.joy import XboxController
    class Device:
        def __init__(self, guid="xbox", attached=True):
            self.guid, self.is_attached, self.closed = guid, attached, False
        def attached(self):
            return self.is_attached
        def as_joystick(self):
            return self
        def get_guid(self):
            return self.guid
        def quit(self):
            self.closed = True
        def get_axis(self, axis):
            return 0
        def get_button(self, button):
            return False
    old, other, new = Device(attached=False), Device("other"), Device()
    reader = XboxController.__new__(XboxController)
    reader.device, reader.guid = old, "xbox"
    reader.api = SimpleNamespace(get_count=lambda: 2, is_controller=lambda i: True,
                                 Controller=lambda i: [other, new][i])
    reader.pygame = SimpleNamespace(event=SimpleNamespace(get=lambda: []), QUIT=-1, error=RuntimeError)
    for name in ("LEFTX", "LEFTY", "RIGHTX", "RIGHTY", "TRIGGERLEFT", "TRIGGERRIGHT"):
        setattr(reader.pygame, "CONTROLLER_AXIS_" + name, 0)
    for name in ("A", "B", "LEFTSHOULDER", "RIGHTSHOULDER", "BACK", "DPAD_LEFT", "DPAD_RIGHT"):
        setattr(reader.pygame, "CONTROLLER_BUTTON_" + name, 0)
    assert not reader.poll().connected
    assert old.closed
    assert reader.poll().connected
    assert reader.device is new and other.closed


@pytest.mark.parametrize("frame", ["base", "tool"])
@pytest.mark.parametrize("axis,sign", [(0, 1.), (0, -1.), (1, 1.), (1, -1.)])
def test_left_stick_center_cancels_pending_motion_while_lb_stays_held(frame, axis, sign):
    from replace_disk_robot.applications.jaka_joy.joy_simulation import make_sim_app
    app = make_sim_app(parse_args(["--linear-speed", ".2", "--command-frame", frame]))
    app._next_print = float("inf")
    t = 0.
    def step(axes, buttons=("lb",)):
        nonlocal t
        app.apply_sample(JoySample(t, tuple(axes), frozenset(buttons)), t)
        app.tick(t, app.client.dt)
        app.client.step()
        t += app.client.dt
    try:
        app.initialize()
        step(np.zeros(6), ())
        axes = np.zeros(6)
        axes[axis] = sign
        for _ in range(125):
            step(axes)
        release_position = app.client.data.body("tool0").xpos.copy()
        distances = []
        for _ in range(500):
            step(np.zeros(6))
            distances.append(np.linalg.norm(app.client.data.body("tool0").xpos - release_position))
        assert app._active and app.servo.fault is None
        assert max(distances) < .001, f"continued travel after centering: {max(distances)*1000:.3f} mm"
        assert not np.any(app._jog_to_servo().linear_m_s)
        # Another stick deflection works without releasing or re-pressing LB.
        for _ in range(25):
            step(axes)
        assert np.linalg.norm(app.client.data.body("tool0").xpos - release_position) > .002
    finally:
        app.shutdown()


def test_left_center_keeps_right_stick_control_and_force_response():
    from replace_disk_robot.applications.jaka_joy.joy_simulation import make_sim_app
    # This motion assertion needs a known speed, independent of the user's YAML.
    app = make_sim_app(parse_args(["--linear-speed", ".2", "--angular-speed-deg", "4"]))
    app._next_print = float("inf")
    t = 0.
    def step(axes):
        nonlocal t
        app.apply_sample(JoySample(t, tuple(axes), frozenset({"lb"})), t)
        app.tick(t, app.client.dt)
        app.client.step()
        t += app.client.dt
    try:
        app.initialize()
        app.apply_sample(JoySample(t), t)
        for _ in range(100):
            step([1., 0., .5, 0., 0., 0.])
        app.client.data.xfrc_applied[app.client.data.body("tool0").id, :3] = [0., 0., -3.]
        for _ in range(25):
            step([1., 0., .5, 0., 0., 0.])
        before = app.motion.state()
        step([0., 0., .5, 0., 0., 0.])
        # The offset and its velocity were retained and integrated for one step.
        assert np.linalg.norm(before.offset[:3]) > 1e-5
        assert np.linalg.norm(app.motion.state().offset[:3]) > 1e-5
        assert np.linalg.norm(app.motion.state().velocity[:3]) > 1e-5
        assert np.linalg.norm(app._jog_to_servo().angular_rad_s) > 0.
        nominal_rotation = rotation_matrix(app.motion.nominal_pose.quaternion_wxyz)
        for _ in range(100):
            step([0., 0., .5, 0., 0., 0.])
        assert np.linalg.norm(rotation_matrix(app.motion.nominal_pose.quaternion_wxyz)
                              - nominal_rotation) > .01
        assert app._active and app.servo.fault is None
    finally:
        app.shutdown()
