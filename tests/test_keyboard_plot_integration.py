"""Regression: plotting/focus must not silently disable keyboard servo."""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'examples'))

from replace_disk_robot.control import KeyControl
from replace_disk_robot.core import Wrench
from replace_disk_robot.visual import ProcessTypePlotter
import replace_disk_robot.visual.process_plotter as process_module
from keyboard_servo import handle_focus, control_status, ServoDemo


class FocusApp:
    def __init__(self):
        self.keys = KeyControl()
        self.servo = SimpleNamespace(fault=None, status='moving')
    def stop(self, reason):
        self.keys.clear()
        self.servo.fault = self.servo.status = reason
    def resume(self):
        self.keys.clear()
        self.servo.fault = None
        self.servo.status = 'holding'


def test_plot_focus_pause_recovers_without_replaying_keys():
    app = FocusApp()
    app.keys.press('q')
    handle_focus(app, False)
    assert app.servo.fault == 'focus_lost'
    assert not app.keys.pressed
    assert 'click ROBOT' in control_status(app)
    handle_focus(app, True)
    assert app.servo.fault is None
    assert not app.keys.pressed
    app.keys.press('w')
    assert app.keys.command().linear_m_s[2] > 0


def test_contact_is_allowed_and_force_above_20n_stops():
    app = ServoDemo()
    assert app.servo.collision_checker is None
    assert app.servo.config.max_tracking_error_rad == 10.0
    assert app.guard.force_limit_n == 20.0
    assert np.isinf(app.guard.torque_limit_nm)

    app.ft.read_wrench = lambda: Wrench('wrist_ft_site', [19.9, 0, 0], [0, 0, 0])
    app.tick()
    assert app.servo.fault is None

    app.ft.read_wrench = lambda: Wrench('wrist_ft_site', [20.1, 0, 0], [0, 0, 0])
    app.tick()
    assert app.servo.fault == 'force_limit'



@pytest.mark.parametrize('fault', ['stopped', 'force_limit', 'tracking_error', 'loop_timeout'])
def test_focus_does_not_erase_or_auto_resume_a_real_fault(fault):
    app = FocusApp()
    app.stop(fault)
    handle_focus(app, False)
    assert app.servo.fault == fault
    handle_focus(app, True)
    assert app.servo.fault == fault
    assert 'Enter' in control_status(app)


def _blocked_plot_worker(samples, events, stop, options):
    events.put(('ready', ''))
    stop.wait(10.)  # Simulates a GUI stuck in a slow redraw, consuming no samples.


def _wait_ready(plotter):
    deadline = time.monotonic()+15
    while plotter.poll() == 'starting' and time.monotonic() < deadline:
        time.sleep(.02)
    assert plotter.status == 'ready', plotter.error


def test_stalled_plot_queue_never_blocks_control_producer(monkeypatch):
    monkeypatch.setattr(process_module, '_plot_worker', _blocked_plot_worker)
    plotter = ProcessTypePlotter(show=False, queue_size=1)
    try:
        _wait_ready(plotter)
        start = time.monotonic()
        for i in range(1000):
            plotter.update(Wrench('wrist', [0,0,0], [0,0,0]), i*.01)
        # Old same-thread redraw exceeding this bound latched loop_timeout.
        assert time.monotonic()-start < .15
        assert plotter.dropped_samples > 0
    finally:
        plotter.close()
    assert not plotter._process.is_alive()


def test_real_plot_process_can_render_and_shut_down():
    plotter = ProcessTypePlotter(show=False)
    try:
        assert plotter.update(Wrench('wrist', [1,2,3], [0,0,0]), 0.)
        _wait_ready(plotter)
        assert plotter.update(Wrench('wrist', [2,3,4], [0,0,0]), .1)
    finally:
        plotter.close()
    assert not plotter._process.is_alive()
    assert plotter.update(Wrench('wrist', [0,0,0], [0,0,0]), .2) is False
    plotter.close()


def test_plot_worker_failure_disables_only_the_plot():
    plotter = ProcessTypePlotter(show=False)
    try:
        plotter.update('unsupported sample', 0.)
        deadline = time.monotonic()+15
        while plotter.poll() == 'starting' and time.monotonic() < deadline:
            time.sleep(.02)
        assert plotter.status == 'error'
        assert 'unsupported visual type' in plotter.error
        assert not plotter.update(Wrench('wrist',[0,0,0],[0,0,0]), .1)
    finally:
        plotter.close()


@pytest.mark.parametrize('key', ['q', 'w', 'e', 'r'])
def test_qwer_reaches_real_servo_after_plot_focus_return(key, monkeypatch):
    monkeypatch.setattr(process_module, '_plot_worker', _blocked_plot_worker)
    app = ServoDemo()
    plotter = ProcessTypePlotter(show=False, queue_size=1)
    try:
        _wait_ready(plotter)
        handle_focus(app, False)
        app.tick()
        handle_focus(app, True)
        initial = app.robot.arm_position().copy()
        app.keys.press(key)
        for _ in range(25):
            app.tick()
            plotter.update(app.last_wrench, app.data.time)
        assert app.servo.fault is None
        assert app.servo.status == 'moving'
        assert max(abs(app.robot.arm_position()-initial)) > .0001
        app.keys.release(key)
        app.tick()
        assert app.servo.status == 'holding'
    finally:
        plotter.close()
