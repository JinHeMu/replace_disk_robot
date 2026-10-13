"""Offline tests for the automatic 5 Hz F/T gravity capture state machine.

These tests drive ``GravityDatasetWriter`` with synthetic EDG samples, so they
need neither a robot, a controller nor a display.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tool"))

from collect_ft_gravity_data import GravityDatasetWriter, _parse_args  # noqa: E402
from replace_disk_robot.core import JointState  # noqa: E402
from replace_disk_robot.control import JoyControl, JoySample  # noqa: E402

JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7))


class _Servo:
    def __init__(self) -> None:
        self.fault: str | None = None


class _Pose:
    quaternion_wxyz = np.array([1.0, 0.0, 0.0, 0.0])


class _Kinematics:
    def forward(self, measured):  # noqa: ARG002 - pose is irrelevant here
        return _Pose()


class _App:
    def __init__(self) -> None:
        self.kinematics = _Kinematics()
        self.servo = _Servo()
        self.joy = JoyControl()
        self.joy.update(JoySample(0.0))
        self._active = False
        self.resume_required = False

    @property
    def joy_sample(self):
        return self.joy.sample


class _State:
    def __init__(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.joint_position_rad = position
        self.joint_velocity_rad_s = velocity
        self.tcp_position_m = np.zeros(3)
        self.tcp_rpy_rad = np.zeros(3)
        self.torque_sensor = np.zeros(6)


class _Wrench:
    def as_vector(self) -> np.ndarray:
        return np.zeros(6)


def _args(tmp_path: Path, **overrides) -> Namespace:
    values = dict(
        output=tmp_path / "samples.csv",
        overwrite=True,
        rate_hz=125.0,
        capture_rate_hz=5.0,
        capture_seconds=10.0,
        settle_seconds=0.75,
        start_delay_seconds=2.0,
        stationary_speed_rad_s=0.01,
        identity_transform=False,
    )
    values.update(overrides)
    return Namespace(**values)


def _feed(
    writer: GravityDatasetWriter,
    args: Namespace,
    app: _App,
    seconds: float,
    *,
    speed: float = 0.0,
    start_s: float = 0.0,
    q: np.ndarray | None = None,
) -> float:
    """Run ``seconds`` of 125 Hz control ticks; returns the next elapsed time."""

    position = np.zeros(6) if q is None else np.asarray(q, dtype=float)
    state = _State(position, np.full(6, speed))
    measured = JointState(JOINT_NAMES, position)
    wrench = _Wrench()
    elapsed = start_s
    for _ in range(int(round(seconds * args.rate_hz))):
        app.joy.update(replace(app.joy_sample, timestamp_s=elapsed))
        writer(elapsed, state, measured, wrench, app)
        elapsed += 1.0 / args.rate_hz
    return elapsed


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_still_pose_is_recorded_automatically_at_5hz(tmp_path):
    args = _args(tmp_path)
    app = _App()
    writer = GravityDatasetWriter(args.output, args)
    _feed(writer, args, app, 13.5)
    writer.close()

    rows = _rows(args.output)
    ids = [int(row["capture_id"]) for row in rows]
    assert writer.completed_captures == 1
    assert writer.capture_rows_target == 51
    assert ids.count(0) == 51
    assert {row["capture_phase"] for row in rows if int(row["capture_id"]) == 0} == {"recording"}

    times = np.array([float(row["time_s"]) for row in rows])
    assert np.all(np.diff(times) > 0)
    np.testing.assert_allclose(np.diff(times), 0.2, atol=0.01)
    assert ids[0] == -1, "rows during the startup delay must stay capture_id=-1"


def test_start_delay_suppresses_the_first_window(tmp_path):
    args = _args(tmp_path, capture_seconds=2.0, start_delay_seconds=5.0)
    app = _App()
    writer = GravityDatasetWriter(args.output, args)
    _feed(writer, args, app, 4.0)
    assert writer.phase == "idle" and writer.completed_captures == 0

    _feed(writer, args, app, 4.0, start_s=4.0)
    assert writer.completed_captures == 1
    writer.close()


def test_motion_cancels_a_window_and_rearms_the_same_pose(tmp_path):
    args = _args(tmp_path, capture_seconds=2.0)
    app = _App()
    writer = GravityDatasetWriter(args.output, args)
    elapsed = _feed(writer, args, app, 3.5)
    assert writer.phase == "recording"

    elapsed = _feed(writer, args, app, 0.4, speed=0.5, start_s=elapsed)
    assert writer.rejected_windows == 1
    assert writer.completed_captures == 0

    _feed(writer, args, app, 3.5, start_s=elapsed)
    assert writer.completed_captures == 1
    writer.close()

    rows = _rows(args.output)
    rejected = [row for row in rows if row["capture_phase"] == "rejected"]
    assert rejected and {int(row["capture_id"]) for row in rejected} == {-1}
    assert [int(row["capture_id"]) for row in rows].count(0) == writer.capture_rows_target
    times = np.array([float(row["time_s"]) for row in rows])
    assert np.all(np.diff(times) > 0), "flushed rows must keep time order"


def test_same_still_pose_is_not_recorded_twice(tmp_path):
    args = _args(tmp_path, capture_seconds=2.0)
    app = _App()
    writer = GravityDatasetWriter(args.output, args)
    _feed(writer, args, app, 12.0)
    assert writer.completed_captures == 1, "the arm must move before a new window starts"
    writer.close()


def test_incomplete_window_is_not_committed_on_close(tmp_path):
    args = _args(tmp_path, capture_seconds=30.0)
    app = _App()
    writer = GravityDatasetWriter(args.output, args)
    _feed(writer, args, app, 6.0)
    assert writer.phase == "recording" and writer.recorded_rows > 0
    writer.close()

    rows = _rows(args.output)
    assert {int(row["capture_id"]) for row in rows} == {-1}
    assert "rejected" in {row["capture_phase"] for row in rows}


def test_latched_servo_fault_blocks_new_windows(tmp_path):
    args = _args(tmp_path, capture_seconds=2.0)
    app = _App()
    app.servo.fault = "force_limit"
    writer = GravityDatasetWriter(args.output, args)
    _feed(writer, args, app, 8.0)
    assert writer.completed_captures == 0

    app.servo.fault = None
    _feed(writer, args, app, 5.0, start_s=8.0)
    assert writer.completed_captures == 1
    writer.close()


def test_lb_release_hold_allows_capture_but_lb_held_does_not(tmp_path):
    args = _args(tmp_path, capture_seconds=2.0)
    app = _App()
    app.servo.fault = "stopped"
    app.joy.update(JoySample(0.0, buttons=frozenset({"lb"})))
    with_writer = GravityDatasetWriter(args.output, args)
    elapsed = _feed(with_writer, args, app, 5.0)
    assert with_writer.completed_captures == 0
    app.joy.update(JoySample(elapsed))
    _feed(with_writer, args, app, 4.0, start_s=elapsed)
    assert with_writer.completed_captures == 1
    with_writer.close()


@pytest.mark.parametrize("unavailable", ["disconnect", "recovery", "stick"])
def test_unavailable_or_non_neutral_input_rejects_window(tmp_path, unavailable):
    args = _args(tmp_path, capture_seconds=2.0)
    app = _App()
    writer = GravityDatasetWriter(args.output, args)
    elapsed = _feed(writer, args, app, 3.5)
    assert writer.phase == "recording"
    if unavailable == "disconnect":
        app.joy.update(JoySample(elapsed, connected=False))
    elif unavailable == "recovery":
        app.resume_required = True
    else:
        app.joy.update(JoySample(elapsed, axes=(.5, 0., 0., 0., 0., 0.)))
    elapsed = _feed(writer, args, app, 4.0, start_s=elapsed)
    assert writer.completed_captures == 0 and writer.rejected_windows == 1
    app.resume_required = False
    app.joy.update(JoySample(elapsed))
    _feed(writer, args, app, 4.0, start_s=elapsed)
    assert writer.completed_captures == 1
    writer.close()


def test_precision_motion_rearms_from_actual_joint_displacement(tmp_path):
    args = _args(tmp_path, capture_seconds=2.0)
    app = _App()
    writer = GravityDatasetWriter(args.output, args)
    elapsed = _feed(writer, args, app, 6.0)
    assert writer.completed_captures == 1
    app._active = True
    app.joy.update(JoySample(elapsed, axes=(.5, 0., 0., 0., 0., 0.),
                            buttons=frozenset({"lb", "rb"})))
    elapsed = _feed(writer, args, app, 1.0, start_s=elapsed, speed=.005, q=np.full(6, .005))
    app._active = False
    app.joy.update(JoySample(elapsed))
    _feed(writer, args, app, 4.0, start_s=elapsed, q=np.full(6, .005))
    assert writer.completed_captures == 2
    writer.close()


def test_brief_lb_press_between_csv_ticks_cancels_window(tmp_path):
    args = _args(tmp_path, capture_seconds=2.0)
    app = _App()
    writer = GravityDatasetWriter(args.output, args)
    elapsed = _feed(writer, args, app, 3.5)
    assert writer.phase == "recording"
    app.joy.update(JoySample(elapsed, buttons=frozenset({"lb"})))
    elapsed = _feed(writer, args, app, .008, start_s=elapsed)
    app.joy.update(JoySample(elapsed))
    _feed(writer, args, app, .3, start_s=elapsed)
    assert writer.rejected_windows == 1 and writer.completed_captures == 0
    writer.close()


def test_missing_controller_does_not_create_csv_or_connect_robot(monkeypatch, tmp_path):
    import collect_ft_gravity_data as collector
    args = _parse_args(["--output", str(tmp_path / "new.csv")])
    monkeypatch.setattr(collector, "_parse_args", lambda: args)
    def unavailable(_index):
        raise RuntimeError("controller unavailable")
    monkeypatch.setattr(collector, "XboxController", unavailable)
    monkeypatch.setattr(collector, "make_client", lambda _args: pytest.fail("must not connect robot"))
    with pytest.raises(SystemExit):
        collector.main()
    assert not args.output.exists()


def test_real_joy_servo_collects_in_dry_run_without_commands(tmp_path):
    from collect_ft_gravity_data import JakaJoyServo
    from test_jaka_keyboard_admittance import DummyClient, LinearKinematics
    args = _parse_args(["--dry-run", "--tare-enable", "false", "--output", str(tmp_path / "joy.csv")])
    assert not args.admittance and not args.gravity_compensation_enable and args.gravity_json is None
    client = DummyClient()
    writer = GravityDatasetWriter(args.output, args)
    app = JakaJoyServo(client, args, kinematics=LinearKinematics(), sample_callback=writer)
    app._next_print = float("inf")
    try:
        app.initialize()
        for k in range(1700):
            elapsed = k / args.rate_hz
            app.apply_sample(JoySample(elapsed), elapsed)
            app.tick(elapsed, 1 / args.rate_hz)
        assert writer.completed_captures == 1
        assert client.commands == [] and client.servo_calls == []
    finally:
        app.shutdown()
        writer.close()


def test_joy_servo_move_release_and_capture_next_pose(tmp_path):
    from collect_ft_gravity_data import JakaJoyServo
    from test_jaka_keyboard_admittance import DummyClient, LinearKinematics
    args = _parse_args(["--tare-enable", "false", "--capture-seconds", "2",
                        "--output", str(tmp_path / "joy.csv")])
    client = DummyClient()
    writer = GravityDatasetWriter(args.output, args)
    app = JakaJoyServo(client, args, kinematics=LinearKinematics(), sample_callback=writer)
    app._next_print = float("inf")
    elapsed = 0.0

    def step(seconds, moving=False):
        nonlocal elapsed
        for _ in range(round(seconds * args.rate_hz)):
            sample = JoySample(elapsed, (1., 0., 0., 0., 0., 0.), frozenset({"lb"})) if moving else JoySample(elapsed)
            app.apply_sample(sample, elapsed)
            app.tick(elapsed, 1 / args.rate_hz)
            client.joint_position = app.servo.target.position_rad.copy()
            elapsed += 1 / args.rate_hz

    try:
        app.initialize()
        step(6)
        assert writer.completed_captures == 1
        step(2, moving=True)
        assert app._active and client.joint_position[0] > .005
        step(4)
        assert app.servo.fault == "stopped" and writer.completed_captures == 2
        assert client.servo_calls == [True]
    finally:
        app.shutdown()
        writer.close()
    assert client.servo_calls == [True, False]
    rows = _rows(args.output)
    assert all(row["command_active"] == "0" for row in rows if int(row["capture_id"]) >= 0)
    assert any(row["command_active"] == "1" for row in rows)
