"""Offline tests for the automatic 5 Hz F/T gravity capture state machine.

These tests drive ``GravityDatasetWriter`` with synthetic EDG samples, so they
need neither a robot nor a display; only the GLFW import of the collector module
is required.
"""

from __future__ import annotations

import csv
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("glfw")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tool"))

from collect_ft_gravity_data import GravityDatasetWriter  # noqa: E402
from replace_disk_robot.core import JointState  # noqa: E402

JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7))


class _Keys:
    def __init__(self) -> None:
        self.pressed: set[str] = set()


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
        self.keys = _Keys()
        self.servo = _Servo()


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
    app.servo.fault = "stopped"
    writer = GravityDatasetWriter(args.output, args)
    _feed(writer, args, app, 8.0)
    assert writer.completed_captures == 0

    app.servo.fault = None
    _feed(writer, args, app, 5.0, start_s=8.0)
    assert writer.completed_captures == 1
    writer.close()
