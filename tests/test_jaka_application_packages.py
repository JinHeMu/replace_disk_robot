"""Application package boundaries and executable entrypoints after the split."""
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('module,forbidden', [
    ('common.node', ('jaka_keyboard', 'jaka_joy')),
    ('jaka_keyboard.node', ('jaka_joy',)),
    ('jaka_joy.joy_node', ('jaka_keyboard',)),
])
def test_headless_import_does_not_load_sibling_apps_or_input_runtimes(module, forbidden):
    code = (f'import importlib, sys; importlib.import_module("replace_disk_robot.applications.{module}"); '
            f'forbidden={forbidden!r}; '
            'assert not any(name.startswith("replace_disk_robot.applications."+app) '
            'for name in sys.modules for app in forbidden); '
            'assert not any(name in sys.modules for name in ("pygame", "glfw", "mujoco"))')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('relative', [
    'src/replace_disk_robot/applications/jaka_joy/jaka_joy_node.py',
    'src/replace_disk_robot/applications/jaka_keyboard/jaka_keyboard_node.py',
    'src/replace_disk_robot/applications/jaka_keyboard/jaka_keyboard_servo.py',
    'tool/analyze_jaka_control_log.py',
])
def test_entrypoint_help_outside_repository(relative, tmp_path):
    entry = Path(__file__).resolve().parents[1] / relative
    result = subprocess.run([sys.executable, str(entry), '--help'], cwd=tmp_path,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout
