"""Command-trace equivalence to the preserved, hardware-validated source.

The original code is imported only with fake clients/kinematics. These tests
never load the SDK, connect to a controller, or enable physical motion.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from test_jaka_keyboard_admittance import DummyClient, LinearKinematics, _raw_sensor, _write_fit
from replace_disk_robot.applications.jaka_keyboard import config, node
from replace_disk_robot.applications.jaka_keyboard.log import json_value

FIXTURE = Path(__file__).parent / "fixtures/jaka_keyboard_servo_before_refactor.py"
spec = importlib.util.spec_from_file_location("keyboard_before_refactor", FIXTURE)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
# Relocating the fixture must not relocate its original config/calibration paths.
legacy.PROJECT_ROOT = config.PROJECT_ROOT
legacy.DEFAULT_CONFIG_PATH = config.DEFAULT_CONFIG_PATH


def test_original_control_preserved_after_path_and_switch_renames():
    import ast
    original = Path(config.__file__).with_name("jaka_keyboard_servo.py")
    class NormalizeSwitches(ast.NodeTransformer):
        def visit_Call(self, item):
            self.generic_visit(item)
            if (isinstance(item.func, ast.Name) and item.func.id == "getattr"
                    and len(item.args) == 3 and isinstance(item.args[1], ast.Constant)):
                baseline_names = {"gravity_compensation_enable": "no_gravity_compensation",
                                  "tare_enable": "no_tare"}
                name = baseline_names.get(item.args[1].value)
                if name:
                    item.args[1] = ast.Constant(name)
                    item.args[2] = ast.Constant(False)
                    return ast.UnaryOp(op=ast.Not(), operand=item)
            return item
        def visit_Attribute(self, item):
            self.generic_visit(item)
            if item.attr == "plot_wrench_enable":
                item.attr = "plot_wrench"
            return item
        def visit_Constant(self, item):
            if isinstance(item.value, str):
                item.value = item.value.replace("set --gravity-compensation-enable true",
                                                "remove --no-gravity-compensation")
            return item
    def control_definitions(source):
        # Parameter parsing is intentionally changed. Normalize equivalent
        # positive switches before comparing the unchanged controller body.
        nodes = NormalizeSwitches().visit(ast.parse(source)).body
        nodes = [item for item in nodes if not (isinstance(item, ast.FunctionDef) and
                 item.name in {"_build_parser", "_load_config", "_validate_config_args", "_validate_admittance_args"})]
        return [ast.dump(item) for item in nodes if
                isinstance(item, (ast.FunctionDef, ast.ClassDef, ast.AnnAssign)) or
                (isinstance(item, ast.Assign) and
                 not any(isinstance(target, ast.Name) and target.id in
                         {"SRC_DIR", "PROJECT_ROOT", "DEFAULT_CONFIG_PATH"}
                         for target in item.targets))]
    assert control_definitions(original.read_text()) == control_definitions(FIXTURE.read_text())


@pytest.mark.parametrize("filename,has_logging", [
    ("jaka_keyboard_node.py", True), ("jaka_keyboard_servo.py", False),
])
def test_moved_entry_help_without_sdk(tmp_path, filename, has_logging):
    import subprocess
    entry = Path(config.__file__).with_name(filename)
    result = subprocess.run([sys.executable, str(entry), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert ("--log-dir" in result.stdout) is has_logging


def test_moved_entries_share_default_config(monkeypatch):
    from replace_disk_robot.applications.jaka_keyboard import jaka_keyboard_servo
    monkeypatch.setattr(sys, "argv", ["keyboard"])
    old_args, new_args = jaka_keyboard_servo._parse_args(), config._parse_args()
    assert config.DEFAULT_CONFIG_PATH.is_file()
    assert jaka_keyboard_servo.DEFAULT_CONFIG_PATH == config.DEFAULT_CONFIG_PATH
    for name, value in vars(old_args).items():
        assert getattr(new_args, name) == value


def make_pair(monkeypatch, tmp_path, *, command_frame="base", dry_run=False,
              admittance=True, record=False, real_profile=False):
    fit, _ = _write_fit(tmp_path)
    profile = config.DEFAULT_CONFIG_PATH if real_profile else tmp_path / "profile.json"
    if not real_profile:
        profile.write_text("{}", encoding="utf-8")
    # Translate only for the frozen pre-refactor fixture; runtime interfaces
    # accept the positive names exclusively.
    old_config = config._load_config(profile, config._build_parser())
    toggle_mapping = {"gravity_compensation_enable": ("no_gravity_compensation", True),
                      "tare_enable": ("no_tare", True),
                      "plot_wrench_enable": ("plot_wrench", False)}
    for current_name, (old_name, inverted) in toggle_mapping.items():
        if current_name in old_config:
            value = old_config.pop(current_name)
            old_config[old_name] = not value if inverted else value
    baseline_profile = tmp_path / "baseline_profile.json"
    baseline_profile.write_text(json.dumps(old_config, default=str))
    argv = ["jaka_keyboard_servo.py", "--config", str(profile),
            "--gravity-json", str(fit), "--command-frame", command_frame,
            "--max-force-n", "100", "--max-torque-nm", "100"]
    if admittance:
        argv.append("--admittance")
    if dry_run:
        argv.append("--dry-run")
    baseline_argv = argv.copy()
    baseline_argv[2] = str(baseline_profile)
    if not admittance:
        baseline_argv += ["--no-gravity-compensation", "--no-tare"]
        argv += ["--gravity-compensation-enable", "false", "--tare-enable", "false"]
    monkeypatch.setattr(sys, "argv", baseline_argv)
    old_args = legacy._parse_args()
    monkeypatch.setattr(sys, "argv", argv)
    new_args = config._parse_args()
    reverse_mapping = {old: (current, inverse) for current, (old, inverse) in toggle_mapping.items()}
    # Defaults and effective settings are identical after translating names.
    for name, value in vars(old_args).items():
        if name == "config":
            continue
        if name in reverse_mapping:
            current, inverse = reverse_mapping[name]
            assert getattr(new_args, current) == (not value if inverse else value)
        else:
            assert getattr(new_args, name) == value
    old_kin, new_kin = LinearKinematics(35), LinearKinematics(35)
    old_kin.end_effector_frame = new_kin.end_effector_frame = "tool0"
    monkeypatch.setattr(legacy, "JakaKinematics", lambda: old_kin)
    old_client, new_client = DummyClient(), DummyClient()
    seen_old, seen_new, captures = [], [], []
    def pre_callback(destination):
        def collect(t, state, q, force, app):
            destination.append((t, q.position_rad.copy(), force.as_vector().copy(),
                                None if app.motion is None else app.motion.state().offset.copy()))
        return collect
    old = legacy.JakaKeyboardServo(old_client, old_args, sample_callback=pre_callback(seen_old))
    new = node.JakaKeyboardServo(new_client, new_args, kinematics=new_kin,
                                sample_callback=pre_callback(seen_new),
                                tick_callback=captures.append if record else None)
    for app in (old, new):
        app._next_print = float("inf")
        app.initialize()
        if app.wrench_processor is not None:
            app.wrench_processor.reset()
    return old, new, old_client, new_client, seen_old, seen_new, captures


def assert_same(old, new, old_client, new_client):
    assert old.status == new.status
    assert old.servo.fault == new.servo.fault
    assert old.servo_enabled == new.servo_enabled
    assert old.keys.pressed == new.keys.pressed
    np.testing.assert_array_equal(old.last_wrench.as_vector(), new.last_wrench.as_vector())
    np.testing.assert_array_equal(old.servo.target.position_rad, new.servo.target.position_rad)
    assert old_client.servo_calls == new_client.servo_calls
    np.testing.assert_array_equal(old_client.commands, new_client.commands)
    if old.motion is not None:
        np.testing.assert_array_equal(old.motion.state().offset, new.motion.state().offset)
        np.testing.assert_array_equal(old.motion.state().velocity, new.motion.state().velocity)
        np.testing.assert_array_equal(old.motion.nominal_pose.position_m, new.motion.nominal_pose.position_m)
        np.testing.assert_array_equal(old.motion.nominal_pose.quaternion_wxyz, new.motion.nominal_pose.quaternion_wxyz)
        np.testing.assert_array_equal(old.last_corrected_pose.position_m, new.last_corrected_pose.position_m)
        np.testing.assert_array_equal(old.last_corrected_pose.quaternion_wxyz, new.last_corrected_pose.quaternion_wxyz)


@pytest.mark.parametrize("command_frame", ["base", "tool"])
@pytest.mark.parametrize("dry_run,admittance,real_profile", [
    (False, True, False), (True, True, False), (False, False, False),
    (True, False, False), (False, True, True),
])
@pytest.mark.parametrize("record", [False, True])
def test_reference_commands_and_callback_order_match_original(
    monkeypatch, tmp_path, command_frame, dry_run, admittance, real_profile, record,
):
    old, new, oc, nc, before_old, before_new, captures = make_pair(
        monkeypatch, tmp_path, command_frame=command_frame, dry_run=dry_run,
        admittance=admittance, record=record, real_profile=real_profile,
    )
    for k in range(100):
        for app, client in ((old, oc), (new, nc)):
            client.joint_position = app.servo.target.position_rad.copy()
            app.keys.clear()
            if 10 <= k < 30:
                app.keys.press("r")
            if 30 <= k < 50:
                app.keys.press("a")
                app.keys.press("q")
            if 60 <= k < 80:
                app.keys.press("f")
            client.torque_sensor = (
                _raw_sensor(app, external_force_base=np.array([1.2, 0.1, 0.0]),
                            external_torque_base=np.array([0.02, 0.01, 0.08]))
                if admittance else np.array([.2, .1, 0, 0, 0, 0])
            )
            app.tick(k*.008, .008 + .001*(k % 3))
        assert_same(old, new, oc, nc)
    for a, b in zip(before_old, before_new, strict=True):
        assert a[0] == b[0]
        for x, y in zip(a[1:], b[1:], strict=True):
            np.testing.assert_array_equal(x, y)
    if record:
        assert len(captures) == 100
        first = deepcopy(json_value(captures[0]))
        new.keys.press("w")
        assert json_value(captures[0]) == first, "snapshots must not alias controller state"
    old.shutdown()
    new.shutdown()
    assert_same(old, new, oc, nc)


@pytest.mark.parametrize("fault", ["force", "tracking", "timeout", "invalid_dt", "invalid_wrench", "stop", "focus"])
def test_stop_recovery_and_command_count_match_original(monkeypatch, tmp_path, fault):
    old, new, oc, nc, _, _, _ = make_pair(monkeypatch, tmp_path, record=True)
    for app, client in ((old, oc), (new, nc)):
        client.torque_sensor = _raw_sensor(app)
        dt = .008
        if fault == "force":
            client.torque_sensor = _raw_sensor(app, external_force_base=np.array([1000., 0, 0]))
        elif fault == "tracking":
            client.joint_position[0] = .5
        elif fault == "timeout":
            dt = .06
        elif fault == "invalid_dt":
            dt = 0
        elif fault == "invalid_wrench":
            client.torque_sensor[0] = np.nan
        elif fault == "stop":
            app.stop("stopped")
        elif fault == "focus":
            app.handle_focus(False)
        app.tick(.1, dt)
    assert_same(old, new, oc, nc)
    for app, client in ((old, oc), (new, nc)):
        client.torque_sensor = _raw_sensor(app)
        if app.wrench_processor is not None:
            app.wrench_processor.reset()
        app.resume()
        app.tick(.2, .008)
    assert_same(old, new, oc, nc)
    old.shutdown()
    new.shutdown()


def test_canonical_node_does_not_import_glfw_in_headless_mode():
    import subprocess
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; from replace_disk_robot.applications.jaka_keyboard.node import JakaKeyboardServo; "
        "assert 'glfw' not in sys.modules; assert 'jkrc' not in sys.modules",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
