"""Detached cycle snapshots and bounded asynchronous JSONL recording.

Only the writer thread serializes and writes samples/events. Metadata and
source snapshots are created before servo enable; close runs after disable.
An SDK send success means acceptance by the adapter, not physical execution.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from queue import Empty, Full, Queue
import subprocess
from threading import Thread
import time

import numpy as np

SCHEMA_VERSION = 1


def json_value(value):
    """Keep invalid sensor values representable as JSON null, never NaN tokens."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_value(item) for item in value]
    return value


def pose_snapshot(pose):
    if pose is None:
        return None
    return {"frame_id": pose.frame_id, "position_m": pose.position_m.copy(),
            "quaternion_wxyz": pose.quaternion_wxyz.copy()}


def admittance_snapshot(app):
    if app.motion is None:
        return None
    state = app.motion.state()
    return {"frame_id": state.frame_id, "offset": state.offset.copy(),
            "velocity": state.velocity.copy(),
            "enabled_axes": app.motion.enabled_axes.copy(),
            "nominal_pose": pose_snapshot(app.motion.nominal_pose)}


def capture_stage(capture, method, app, *args):
    """Observation failures must never interrupt the hardware command path."""
    if capture is None:
        return
    try:
        getattr(capture, method)(*args)
    except Exception as exc:
        capture.data.setdefault("observer_errors", []).append(str(exc))
        app._observer_error(exc)


class CycleCapture:
    """One control tick; stages share a sequence but retain their own timestamps."""
    def __init__(self, app, sequence, elapsed_s, dt_s):
        self.data = {
            "schema_version": SCHEMA_VERSION, "sequence": sequence,
            "elapsed_s": elapsed_s, "dt_s": dt_s,
            "tick_start_monotonic_ns": time.perf_counter_ns(),
            "pressed_before": sorted(app.keys.pressed),
            "admittance_before": admittance_snapshot(app),
            "fault_before": app.servo.fault,
            "admittance_updated": False, "command_sent": False,
        }

    def feedback(self, app, state, measured):
        self.data.update({
            "feedback_received_monotonic_ns": time.perf_counter_ns(),
            "joint_names": list(measured.names),
            "measured_q_rad": measured.position_rad.copy(),
            "measured_qd_rad_s": state.joint_velocity_rad_s.copy(),
            "measured_joint_torque": state.joint_torque.copy(),
            "sdk_cartesian_pose_raw": state.cartesian_pose.copy(),
            "raw_sensor_wrench": state.torque_sensor.copy(),
            "raw_sensor_frame_id": app.sensor_frame_id,
            "measured_tcp_pose": pose_snapshot(app.kinematics.forward(measured)),
        })

    def jog(self, jog):
        self.data["jog"] = {"base_frame": jog.base_frame,
                            "linear_m_s": jog.linear_m_s.copy(),
                            "angular_rad_s": jog.angular_rad_s.copy()}

    def reference(self, app, wrench):
        self.data["admittance_updated"] = True
        self.data["admittance_integrated"] = admittance_snapshot(app)
        self.data["unclamped_pose"] = pose_snapshot(app.motion.corrected_pose)
        self.data["corrected_pose"] = pose_snapshot(app.last_corrected_pose)
        self.data["admittance_input_wrench"] = (
            wrench.as_vector() * app.motion.enabled_axes
        )
        before = self.data["admittance_before"]
        if before is None:
            raise ValueError("missing pre-update admittance snapshot")
        after = self.data["admittance_integrated"]
        cfg = app.admittance.config
        v = before["velocity"]
        x = before["offset"]
        unbounded = v + self.data["dt_s"] * (
            self.data["admittance_input_wrench"] - cfg.damping*v - cfg.stiffness*x
        ) / cfg.mass
        self.data["velocity_clipped_axes"] = np.abs(unbounded) > cfg.max_velocity
        self.data["translation_offset_clamped"] = (
            np.linalg.norm(after["offset"][:3]) > app.args.adm_max_offset_m
        )
        self.data["rotation_offset_clamped"] = (
            np.linalg.norm(after["offset"][3:]) > np.deg2rad(app.args.adm_max_offset_deg)
        )

    def finish(self, app, error=None):
        self.data.update({
            "tick_end_monotonic_ns": time.perf_counter_ns(),
            "status": app.status, "fault": app.servo.fault,
            "dry_run": bool(app.args.dry_run), "error": error,
            "pressed_after": sorted(app.keys.pressed),
            "admittance_after": admittance_snapshot(app),
        })
        return self.data


class AsyncRecorder:
    """Opt-in recorder. Dropped rows and writer errors are explicit in summary."""
    def __init__(self, directory, metadata, *, queue_size=4096):
        if queue_size < 1:
            raise ValueError("log queue size must be positive")
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.queue = Queue(maxsize=queue_size)
        self.accepted = self.written = self.dropped = 0
        self.written_samples = self.observer_errors = 0
        self.error = None
        self.closed = False
        self._files = {}
        try:
            for name in ("samples", "events"):
                self._files[name] = (self.directory / f"{name}.jsonl").open("x", encoding="utf-8")
            (self.directory / "metadata.json").write_text(
                json.dumps(json_value(metadata), ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        except Exception:
            for file in self._files.values():
                file.close()
            raise
        self._thread = Thread(target=self._run, name="jaka-log-writer", daemon=True)
        self._thread.start()

    def publish(self, kind, record):
        # Caller transfers a detached snapshot; never enqueue live controller arrays.
        if self.closed or self.error:
            self.dropped += 1
            return False
        try:
            self.queue.put_nowait((kind, record))
        except Full:
            self.dropped += 1
            return False
        self.accepted += 1
        return True

    def note_observer_error(self):
        # A failed snapshot can lose even the first/last cycle without a visible gap.
        self.observer_errors += 1

    def _run(self):
        last_flush = time.monotonic()
        try:
            while True:
                try:
                    item = self.queue.get(timeout=0.2)
                except Empty:
                    for file in self._files.values():
                        file.flush()
                    continue
                try:
                    if item is None:
                        break
                    kind, record = item
                    self._files[kind].write(json.dumps(
                        json_value(record), ensure_ascii=False,
                        separators=(",", ":"), allow_nan=False,
                    ) + "\n")
                    self.written += 1
                    if kind == "samples":
                        self.written_samples += 1
                    if time.monotonic() - last_flush >= 0.5:
                        for file in self._files.values():
                            file.flush()
                        last_flush = time.monotonic()
                finally:
                    self.queue.task_done()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            for file in self._files.values():
                try:
                    file.close()
                except Exception as exc:
                    self.error = self.error or str(exc)

    def close(self, timeout_s=5.0):
        if self.closed:
            return
        self.closed = True
        deadline = time.monotonic() + timeout_s
        if self._thread.is_alive():
            try:
                self.queue.put(None, timeout=max(0.0, deadline-time.monotonic()))
            except Full:
                self.error = self.error or "writer queue did not drain on close"
            self._thread.join(max(0.0, deadline-time.monotonic()))
        summary = {
            "accepted": self.accepted, "written": self.written,
            "dropped": self.dropped, "pending": max(0, self.accepted-self.written),
            "written_samples": self.written_samples, "observer_errors": self.observer_errors,
            "writer_error": self.error, "writer_stopped": not self._thread.is_alive(),
        }
        summary["complete"] = (
            summary["writer_stopped"] and not summary["writer_error"]
            and not summary["dropped"] and not summary["pending"]
            and not summary["observer_errors"]
        )
        (self.directory / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8",
        )


def session_metadata(args, app, project_root):
    """Snapshot effective parameters and exact model/calibration/code sources."""
    from replace_disk_robot.kinematics.jaka import PROJECT_JAKA_URDF
    from replace_disk_robot.control import admittance, admittance_motion, servo
    from replace_disk_robot.contact import force_processing
    sources = {}
    paths = [Path(args.config), PROJECT_JAKA_URDF, Path(__file__),
             Path(__file__).with_name("node.py"), Path(__file__).with_name("reference.py"),
             Path(__file__).with_name("wrench.py"), Path(__file__).with_name("config.py"),
             Path(__file__).with_name("frontend.py"),
             Path(admittance.__file__), Path(admittance_motion.__file__),
             Path(servo.__file__), Path(force_processing.__file__)]
    if getattr(args, "gravity_json", None):
        paths.append(Path(args.gravity_json))
    for path in paths:
        if path.is_file():
            content = path.read_bytes()
            sources[str(path.resolve())] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "content": content.decode("utf-8"),
            }
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root,
                                  capture_output=True, text=True, timeout=2).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=project_root,
                                   capture_output=True, text=True, timeout=2).stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        revision, dirty = None, None
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "clock": {"monotonic_ns": time.perf_counter_ns(), "unix_ns": time.time_ns(),
                  "sensor_timestamp_available": False},
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "git_revision": revision, "git_dirty": dirty,
        "effective_args": vars(args), "sources": sources,
        "base_frame": app.base_frame, "tool_frame": app.tool_frame,
        "kinematics_class": f"{type(app.kinematics).__module__}.{type(app.kinematics).__qualname__}",
        "joint_names": list(app.kinematics.joint_names),
        "sensor_frame": app.sensor_frame_id,
        "sensor_to_tool_rotation": app.sensor_to_tool_rotation.copy(),
        "tool_to_sensor_m": app.tool_to_sensor_m.copy(),
        "servo_config": app.servo.config,
        "runtime_calibration": {
            "adapter_bias_sensor": app.ft.bias.copy(),
            "gravity_bias_sensor": (None if app.gravity_compensator is None
                                    else app.gravity_compensator.bias_sensor.copy()),
            "gravity_compensation_config": (None if app.gravity_compensator is None
                                            else app.gravity_compensator.config),
        },
        "units": {"position": "m", "angle": "rad", "force": "N", "torque": "N*m",
                  "offset_axes": ["x", "y", "z", "rx", "ry", "rz"],
                  "quaternion": "wxyz", "sdk_cartesian_pose_raw": "mm,rad",
                  "measured_joint_torque": "SDK raw units"},
    }
