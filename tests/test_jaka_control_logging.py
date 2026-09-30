"""Logging integrity, failure isolation and independent discrete-equation checks."""
from copy import deepcopy
import json
from pathlib import Path
from threading import Event
import time

import numpy as np
import pytest

from test_jaka_keyboard_admittance import DummyClient, LinearKinematics, _raw_sensor, _write_fit
from test_jaka_keyboard_refactor import make_pair, assert_same
from replace_disk_robot.applications.jaka_keyboard import config, node
from replace_disk_robot.applications.jaka_keyboard.analysis import analyze_session
from replace_disk_robot.applications.jaka_keyboard.log import AsyncRecorder, CycleCapture


def logged_app(monkeypatch, tmp_path, *, dry_run=False):
    import sys
    fit, _ = _write_fit(tmp_path)
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["keyboard", "--config", str(profile), "--admittance",
                         "--gravity-json", str(fit), "--log-dir", str(tmp_path/"session"),
                         "--max-force-n", "100", "--max-torque-nm", "100"]
                        + (["--dry-run"] if dry_run else []))
    args = config._parse_args()
    kin = LinearKinematics(35)
    kin.end_effector_frame = "tool0"
    client = DummyClient()
    app = node.JakaKeyboardServo(client, args, kinematics=kin)
    app._next_print = float("inf")
    app.initialize()
    app.wrench_processor.reset()
    return app, client, tmp_path/"session"


@pytest.mark.parametrize("dry_run", [False, True])
def test_log_roundtrip_and_independent_theory(monkeypatch, tmp_path, dry_run):
    app, client, directory = logged_app(monkeypatch, tmp_path, dry_run=dry_run)
    for k in range(30):
        client.joint_position = app.servo.target.position_rad.copy()
        client.torque_sensor = _raw_sensor(app, external_force_base=np.array([5., 0, 0]))
        app.keys.press("a")
        app.tick(k*.008, .008)
    app.stop("stopped")
    app.resume()
    app.shutdown()
    report, _, rows = analyze_session(directory)
    assert len(rows) == 30
    assert report["recording_complete"] is True
    assert report["discrete_equation_matches"] is True
    assert report["max_discrete_residual"] < 1e-12
    assert report["inconsistent_input_cycles"] == 0
    assert all(r["command_sent"] is not dry_run for r in rows)
    assert all(r["admittance_updated"] for r in rows)
    assert all(r["wrench_stages"]["gravity_compensation"] for r in rows)
    metadata = json.loads((directory/"metadata.json").read_text())
    assert metadata["effective_args"]["gravity_compensation_enable"] is True
    assert metadata["effective_args"]["tare_enable"] is True
    assert not any(name.startswith("no_") for name in metadata["effective_args"])
    assert metadata["sensor_to_tool_rotation"] == app.sensor_to_tool_rotation.tolist()
    assert metadata["runtime_calibration"]["gravity_bias_sensor"] == app.gravity_compensator.bias_sensor.tolist()
    assert metadata["sources"]
    events = [json.loads(line)["event"] for line in (directory/"events.jsonl").read_text().splitlines()]
    assert "initialized" in events and "stop" in events and "shutdown" in events
    if not dry_run:
        assert "resumed" in events and "command_accepted" in events
    else:
        assert client.commands == [] and client.servo_calls == []
    # Altering a logged state must be detected by independent arithmetic.
    rows[0]["admittance_integrated"]["offset"][0] += .001
    (directory/"samples.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    corrupt, _, _ = analyze_session(directory)
    assert corrupt["discrete_equation_matches"] is False


class BrokenRecorder:
    error = None
    def publish(self, *args):
        raise OSError("disk unavailable")
    def close(self):
        raise OSError("disk unavailable")


def test_sink_and_callback_failure_do_not_change_commands(monkeypatch, tmp_path):
    old, new, oc, nc, _, _, _ = make_pair(monkeypatch, tmp_path)
    new.recorder = BrokenRecorder()
    def broken_callback(row):
        raise RuntimeError("observer unavailable")
    new.tick_callback = broken_callback
    for k in range(10):
        for app, client in ((old, oc), (new, nc)):
            client.joint_position = app.servo.target.position_rad.copy()
            client.torque_sensor = _raw_sensor(app)
            app.keys.press("r")
            app.tick(k*.008, .008)
        assert_same(old, new, oc, nc)
    old.shutdown()
    new.shutdown()
    assert_same(old, new, oc, nc)


def test_capture_failure_does_not_interrupt_control(monkeypatch, tmp_path):
    old, new, oc, nc, _, _, captures = make_pair(monkeypatch, tmp_path, record=True)
    def fail(*args):
        raise ValueError("bad diagnostic")
    monkeypatch.setattr(CycleCapture, "reference", fail)
    for app, client in ((old, oc), (new, nc)):
        client.torque_sensor = _raw_sensor(app)
        app.tick(.1, .008)
    assert_same(old, new, oc, nc)
    assert captures[0]["observer_errors"]
    old.shutdown()
    new.shutdown()


def test_bounded_queue_drops_without_waiting(monkeypatch, tmp_path):
    entered, release = Event(), Event()
    original_run = AsyncRecorder._run
    def delayed_writer(recorder):
        entered.set()
        release.wait(3)
        original_run(recorder)
    monkeypatch.setattr(AsyncRecorder, "_run", delayed_writer)
    recorder = AsyncRecorder(tmp_path/"log", {"schema_version": 1}, queue_size=1)
    assert entered.wait(1)
    start = time.perf_counter()
    assert recorder.publish("samples", {"sequence": 1})
    assert not recorder.publish("samples", {"sequence": 2})
    assert time.perf_counter()-start < .1
    release.set()
    recorder.close()
    summary = json.loads((tmp_path/"log/summary.json").read_text())
    assert summary["dropped"] == 1 and summary["complete"] is False
    assert summary["written"] == 1


def test_nonfinite_values_and_directory_protection(tmp_path):
    recorder = AsyncRecorder(tmp_path/"log", {"schema_version": 1})
    recorder.publish("samples", {"force": np.array([np.nan, np.inf, 1.])})
    recorder.close()
    assert json.loads((tmp_path/"log/samples.jsonl").read_text())["force"] == [None, None, 1.]
    with pytest.raises(FileExistsError):
        AsyncRecorder(tmp_path/"log", {})
    assert json.loads((tmp_path/"log/samples.jsonl").read_text())["force"][-1] == 1.


def test_invalid_dt_and_force_stop_are_logged(monkeypatch, tmp_path):
    app, client, directory = logged_app(monkeypatch, tmp_path)
    client.torque_sensor = _raw_sensor(app)
    app.tick(0, 0)
    app.resume()
    app.wrench_processor.reset()
    client.torque_sensor = _raw_sensor(app, external_force_base=np.array([1000., 0, 0]))
    app.tick(.1, .008)
    app.shutdown()
    rows = [json.loads(line) for line in (directory/"samples.jsonl").read_text().splitlines()]
    assert rows[0]["fault"] == "invalid_dt"
    assert rows[0]["admittance_updated"] is False
    assert rows[1]["fault"] == "force_limit"
    assert rows[1]["force_gate_tripped"] is True
    assert len(rows[1]["send_attempts"]) == 2, "preserve both original hold sends"
    assert np.linalg.norm(rows[1]["admittance_integrated"]["offset"]) > 0
    assert np.linalg.norm(rows[1]["admittance_after"]["offset"]) == 0


@pytest.mark.parametrize("removed_index", [0, 1, 2])
def test_missing_row_and_unclosed_log_are_not_complete(monkeypatch, tmp_path, removed_index):
    app, client, directory = logged_app(monkeypatch, tmp_path)
    for k in range(3):
        client.torque_sensor = _raw_sensor(app)
        app.tick(k*.008, .008)
    app.shutdown()
    rows = (directory/"samples.jsonl").read_text().splitlines()
    (directory/"samples.jsonl").write_text("\n".join(
        row for index, row in enumerate(rows) if index != removed_index
    )+"\n")
    report, _, _ = analyze_session(directory)
    assert report["missing_sequences_between_rows"] == (1 if removed_index == 1 else 0)
    assert report["sample_count_matches_summary"] is False
    assert report["recording_complete"] is False
    (directory/"summary.json").unlink()
    report, _, _ = analyze_session(directory)
    assert report["recording_complete"] is False


@pytest.mark.parametrize("stage", ["__init__", "finish"])
def test_lost_snapshot_marks_session_incomplete(monkeypatch, tmp_path, stage):
    app, client, directory = logged_app(monkeypatch, tmp_path)
    def failed_snapshot(*args, **kwargs):
        raise RuntimeError("snapshot unavailable")
    monkeypatch.setattr(CycleCapture, stage, failed_snapshot)
    client.torque_sensor = _raw_sensor(app)
    app.tick(0, .008)
    assert len(client.commands) == 1
    assert app.servo.fault is None
    app.shutdown()
    report, _, rows = analyze_session(directory)
    assert rows == []
    assert report["writer_summary"]["observer_errors"] >= 1
    assert report["recording_complete"] is False


def test_background_write_failure_is_reported(tmp_path):
    recorder = AsyncRecorder(tmp_path/"log", {"schema_version": 1})
    original = recorder._files["samples"]
    class FailingFile:
        def write(self, text):
            raise OSError("disk full")
        def flush(self):
            original.flush()
        def close(self):
            original.close()
    recorder._files["samples"] = FailingFile()
    recorder.publish("samples", {"sequence": 1})
    deadline = time.monotonic()+2
    while recorder.error is None and time.monotonic() < deadline:
        time.sleep(.005)
    assert "disk full" in recorder.error
    assert not recorder.publish("samples", {"sequence": 2})
    recorder.close()
    summary = json.loads((tmp_path/"log/summary.json").read_text())
    assert summary["complete"] is False
    assert summary["writer_stopped"] is True
    assert "disk full" in summary["writer_error"]


def test_sdk_send_failure_preserves_exception_and_logs_attempt(monkeypatch, tmp_path):
    app, client, directory = logged_app(monkeypatch, tmp_path)
    client.torque_sensor = _raw_sensor(app)
    def failed_send(*args, **kwargs):
        raise RuntimeError("SDK send failed")
    monkeypatch.setattr(client, "edg_servo_j", failed_send)
    with pytest.raises(RuntimeError, match="SDK send failed"):
        app.tick(0, .008)
    app.shutdown()
    row = json.loads((directory/"samples.jsonl").read_text())
    assert row["command_sent"] is False
    assert row["send_attempts"][0]["accepted"] is False
    assert "SDK send failed" in row["error"]
    events = [json.loads(line)["event"] for line in (directory/"events.jsonl").read_text().splitlines()]
    assert "command_rejected" in events


def test_wrench_observation_failure_does_not_fault_control(monkeypatch, tmp_path):
    from replace_disk_robot.applications.jaka_keyboard import wrench
    old, new, oc, nc, _, _, captures = make_pair(monkeypatch, tmp_path, record=True)
    def failed_diagnostic(*args, **kwargs):
        raise RuntimeError("force diagnostic failed")
    monkeypatch.setattr(wrench, "external_wrench_at_tcp", failed_diagnostic)
    for app, client in ((old, oc), (new, nc)):
        client.torque_sensor = _raw_sensor(app)
        app.tick(0, .008)
    assert_same(old, new, oc, nc)
    assert "force diagnostic failed" in captures[0]["wrench_stages"]["capture_error"]
    old.shutdown()
    new.shutdown()
