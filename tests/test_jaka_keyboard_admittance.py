"""Offline tests for the real-JAKA keyboard admittance path.

No robot, no pinocchio and no MuJoCo: a linear identity-Jacobian kinematics and
a dummy EDG client drive the same ``JakaKeyboardServo`` code path that runs on
hardware.  The synthetic sensor reading is built from the same forward
kinematics and payload model the controller compensates with, so any residual
compliant offset means the gravity/bias/frame chain is wrong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("glfw")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples" / "jaka_driver_tool"))

from replace_disk_robot.applications.jaka_keyboard import jaka_keyboard_servo as keyboard  # noqa: E402
from replace_disk_robot.adapters.jaka import EdgState  # noqa: E402
from replace_disk_robot.core import Pose  # noqa: E402
from replace_disk_robot.core.rotation import (  # noqa: E402
    rotation_matrix,
    rotation_vector_from_matrix,
)

DT_S = 1.0 / 125.0
GRAVITY = 9.80665


class LinearKinematics:
    """Identity-Jacobian kinematics: the six joints are the six Cartesian axes.

    ``tool_rotation_deg`` gives the tool0 orientation about its own X axis; a
    non-identity value catches quaternion composition mistakes that an identity
    rotation would hide.
    """

    def __init__(self, tool_rotation_deg: float = 0.0) -> None:
        self.joint_names = ("x", "y", "z", "rx", "ry", "rz")
        self.dof = 6
        self.base_frame = "world"
        self.end_effector_frame = "world"
        self.joint_limits_rad = (np.full(6, -1.0), np.full(6, 1.0))
        half = 0.5 * np.deg2rad(tool_rotation_deg)
        self.quaternion = np.array([np.cos(half), np.sin(half), 0.0, 0.0])

    def forward(self, state):
        position = np.asarray(state.position_rad, dtype=float)
        return Pose("world", position[:3].copy(), self.quaternion.copy())

    def jacobian(self, state):  # noqa: ARG002 - constant Jacobian
        return np.eye(6)


class DummyClient:
    """Minimal ``JakaEdgClient`` surface used by the keyboard servo."""

    def __init__(self) -> None:
        self.joint_position = np.zeros(6)
        self.torque_sensor = np.zeros(6)
        self.commands: list[np.ndarray] = []
        self.servo_calls: list[bool] = []

    def read_edg_state(self) -> EdgState:
        return EdgState(
            self.joint_position.copy(),
            np.zeros(6),
            np.zeros(6),
            np.zeros(6),
            self.torque_sensor.copy(),
            (),
        )

    def edg_servo_j(self, q, step_num: int = 1) -> None:  # noqa: ARG002
        self.commands.append(np.asarray(q, dtype=float).copy())

    def servo_move_enable(self, enable: bool = True) -> None:
        self.servo_calls.append(bool(enable))


def _write_fit(
    tmp_path: Path,
    *,
    mass: float = 1.5,
    tilt_deg: float = 15.0,
    azimuth_deg: float = 30.0,
) -> tuple[Path, np.ndarray]:
    """Write a payload identification JSON with a tilted gravity vector."""

    direction = np.array([
        np.sin(np.deg2rad(tilt_deg)) * np.cos(np.deg2rad(azimuth_deg)),
        np.sin(np.deg2rad(tilt_deg)) * np.sin(np.deg2rad(azimuth_deg)),
        -np.cos(np.deg2rad(tilt_deg)),
    ])
    path = tmp_path / "payload_identified.json"
    path.write_text(json.dumps({
        "mass_kg": mass,
        "signed_gravity_force_base_n": (mass * GRAVITY * direction).tolist(),
        "center_of_mass_sensor_m": [0.012, -0.021, 0.055],
        "force_bias_sensor_n": [0.31, -0.22, 0.14],
        "torque_bias_sensor_nm": [0.013, -0.007, 0.021],
    }), encoding="utf-8")
    return path, direction


def _make_app(
    monkeypatch,
    tmp_path: Path,
    *argv: str,
    kinematics=None,
) -> tuple[keyboard.JakaKeyboardServo, DummyClient]:
    monkeypatch.setattr(
        keyboard, "JakaKinematics",
        LinearKinematics if kinematics is None else (lambda: kinematics),
    )
    # Behaviour tests use the parser's fixed defaults, not the operator's
    # mutable, hardware-validated YAML profile. Profile compatibility is
    # covered separately against the pre-refactor implementation.
    config = tmp_path / "test_keyboard_config.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "jaka_keyboard_servo.py", "--config", str(config), *argv,
    ])
    args = keyboard._parse_args()
    client = DummyClient()
    app = keyboard.JakaKeyboardServo(client, args)
    app._next_print = float("inf")  # keep test output clean
    app.initialize()
    if app.wrench_processor is not None:
        # initialize() ran against the all-zero dummy EDG state, which is not a
        # valid sensor reading.  The synthetic stream starts with the first
        # tick, so drop the filter history it accumulated.
        app.wrench_processor.reset()
    return app, client


def _push_args(fit_path: Path) -> tuple[str, ...]:
    """Admittance args for a contact test.

    A hand push on the tool tip arrives at the sensor origin and is shifted to
    tool0 over the 0.4 m arm constant, so a few newtons already exceed the
    default 2 N*m guard limit.  These tests exercise the compliance chain, not
    the guard, so they raise both limits.
    """

    return (
        "--admittance", "--gravity-json", str(fit_path),
        "--max-force-n", "100", "--max-torque-nm", "100",
    )


def _raw_sensor(
    app: keyboard.JakaKeyboardServo,
    *,
    external_force_base: np.ndarray | None = None,
    external_torque_base: np.ndarray | None = None,
) -> np.ndarray:
    """Synthetic EDG reading: payload gravity + optional external load + bias."""

    measured = app.arm.read_joint_state()
    sensor_pose = app._sensor_pose_in_base(measured)
    rotation = rotation_matrix(sensor_pose.quaternion_wxyz)
    config = app.gravity_compensator.config
    gravity_sensor = rotation.T @ np.asarray(config.gravity_m_s2, dtype=float)
    payload_force = (
        float(config.load_sign) * float(app.payload_mass_kg) * gravity_sensor
    )
    payload_torque = np.cross(
        np.asarray(app.payload_com_sensor_m, dtype=float), payload_force
    )
    force = payload_force + rotation.T @ np.asarray(
        np.zeros(3) if external_force_base is None else external_force_base, dtype=float
    )
    torque = payload_torque + rotation.T @ np.asarray(
        np.zeros(3) if external_torque_base is None else external_torque_base, dtype=float
    )
    return np.r_[force, torque] + np.asarray(
        app.gravity_compensator.bias_sensor, dtype=float
    )


def _run(
    app: keyboard.JakaKeyboardServo,
    client: DummyClient,
    ticks: int,
    *,
    load=None,
    start_s: float = 0.0,
) -> float:
    """Drive ticks; ``load(app)`` returns the synthetic reading for the current pose.

    A real EDG packet always matches the pose that is read in the same tick, so
    the synthetic sensor reading is rebuilt every tick instead of being frozen
    at the pose where the test started.
    """

    elapsed = start_s
    for _ in range(ticks):
        client.joint_position = (
            app.servo.target.position_rad if app.servo.target is not None else np.zeros(6)
        )
        if load is not None:
            client.torque_sensor = np.asarray(load(app), dtype=float)
        app.tick(elapsed, DT_S)
        elapsed += DT_S
    return elapsed


def _offset(app: keyboard.JakaKeyboardServo) -> np.ndarray:
    assert app.motion is not None
    return app.motion.state().offset


def test_gravity_json_supplies_mass_com_bias_and_tilt(monkeypatch, tmp_path):
    path, direction = _write_fit(tmp_path, mass=1.5, tilt_deg=15.0, azimuth_deg=30.0)
    app, _client = _make_app(
        monkeypatch, tmp_path, "--admittance", "--gravity-json", str(path),
    )

    assert app.gravity_compensator is not None
    assert app.wrench_processor is not None
    config = app.gravity_compensator.config
    assert config.payload_mass_kg == pytest.approx(1.5)
    np.testing.assert_allclose(config.payload_com_sensor_m, [0.012, -0.021, 0.055])
    np.testing.assert_allclose(config.gravity_m_s2, GRAVITY * direction, atol=1e-12)
    np.testing.assert_allclose(
        app.gravity_compensator.bias_sensor, [0.31, -0.22, 0.14, 0.013, -0.007, 0.021]
    )
    assert config.gravity_frame_id == app.base_frame == "world"
    # Without explicit extrinsic fields the module constants are used.
    np.testing.assert_allclose(
        app.sensor_to_tool_rotation, keyboard.DEFAULT_SENSOR_TO_TOOL_ROTATION
    )
    np.testing.assert_allclose(app.tool_to_sensor_m, keyboard.DEFAULT_TOOL_ARM_M)


def test_tilted_payload_gravity_causes_no_compliant_drift(monkeypatch, tmp_path):
    """The whole point of gravity compensation: own weight must not move the offset."""

    path, _direction = _write_fit(tmp_path, mass=1.5, tilt_deg=15.0)
    app, client = _make_app(
        monkeypatch, tmp_path, "--admittance", "--gravity-json", str(path),
    )
    _run(app, client, 100, load=lambda target: _raw_sensor(target))

    assert np.linalg.norm(_offset(app)[:3]) < 1e-6
    assert np.linalg.norm(_offset(app)[3:]) < 1e-6
    assert app.last_corrected_pose is not None
    np.testing.assert_allclose(
        app.last_corrected_pose.position_m, app.motion.nominal_pose.position_m, atol=1e-9
    )
    # The payload weight is real: without compensation it would be ~14.7 N.
    assert float(np.linalg.norm(_raw_sensor(app)[:3])) > 10.0


def test_external_push_moves_the_compliant_offset(monkeypatch, tmp_path):
    path, _direction = _write_fit(tmp_path)
    app, client = _make_app(monkeypatch, tmp_path, *_push_args(path))
    push = np.array([5.0, 0.0, 0.0])
    _run(app, client, 150, load=lambda target: _raw_sensor(target, external_force_base=push))
    assert app.servo.fault is None, app.servo.fault

    offset = _offset(app)
    assert offset[0] > 0.005, offset
    assert abs(offset[1]) < 1e-9 and abs(offset[2]) < 1e-9, offset
    nominal = app.motion.nominal_pose
    assert app.last_corrected_pose.position_m[0] > nominal.position_m[0]
    # The pose servo must actually chase the compliant reference.
    assert app.servo.target is not None
    assert app.servo.target.position_rad[0] > 0.0


def test_compliant_offset_is_clamped(monkeypatch, tmp_path):
    path, _direction = _write_fit(tmp_path)
    app, client = _make_app(
        monkeypatch, tmp_path, *_push_args(path), "--adm-max-offset-m", "0.005",
    )
    push = np.array([5.0, 0.0, 0.0])
    _run(app, client, 200, load=lambda target: _raw_sensor(target, external_force_base=push))
    assert app.servo.fault is None, app.servo.fault

    # The integrator itself is unbounded (5 N / 300 N/m = 16.7 mm steady state);
    # the clamp applies to the pose that is handed to the servo.
    assert float(np.linalg.norm(_offset(app)[:3])) > 0.01
    delta = app.last_corrected_pose.position_m - app.motion.nominal_pose.position_m
    assert 0.0049 <= float(np.linalg.norm(delta)) <= 0.005 + 1e-9


def test_translation_mask_ignores_torque_but_all_axes_do_not(monkeypatch, tmp_path):
    path, _direction = _write_fit(tmp_path)
    torque = np.array([0.0, 0.0, 2.0])
    app, client = _make_app(monkeypatch, tmp_path, *_push_args(path))
    _run(app, client, 100, load=lambda target: _raw_sensor(target, external_torque_base=torque))
    assert np.linalg.norm(_offset(app)) < 1e-9
    assert app.admittance_text().endswith("axes=111000")

    app_all, client_all = _make_app(
        monkeypatch, tmp_path, *_push_args(path), "--admittance-axes", "all",
    )
    _run(app_all, client_all, 100,
         load=lambda target: _raw_sensor(target, external_torque_base=torque))
    assert abs(_offset(app_all)[5]) > 1e-4


def test_stop_and_resume_reseed_the_compliance(monkeypatch, tmp_path):
    path, _direction = _write_fit(tmp_path)
    app, client = _make_app(monkeypatch, tmp_path, *_push_args(path))
    push = np.array([5.0, 0.0, 0.0])
    load = lambda target: _raw_sensor(target, external_force_base=push)  # noqa: E731
    _run(app, client, 100, load=load)
    assert _offset(app)[0] > 0.005

    app.stop("stopped")
    assert app.servo.fault == "stopped"
    np.testing.assert_allclose(_offset(app), np.zeros(6))

    assert app.resume() is True
    assert app.servo.fault is None
    np.testing.assert_allclose(_offset(app), np.zeros(6))

    # After the resume a fresh offset is integrated from the measured pose.
    _run(app, client, 100, load=load, start_s=1.0)
    assert _offset(app)[0] > 0.005


def test_dry_run_never_commands_the_robot(monkeypatch, tmp_path):
    path, _direction = _write_fit(tmp_path)
    app, client = _make_app(
        monkeypatch, tmp_path, *_push_args(path), "--dry-run",
    )
    push = np.array([5.0, 0.0, 0.0])
    _run(app, client, 100, load=lambda target: _raw_sensor(target, external_force_base=push))

    assert client.commands == []
    assert client.servo_calls == []
    assert app.servo_enabled is False
    # The chain still integrates, so the sign of a hand push can be verified.
    assert _offset(app)[0] > 0.005
    assert "Fext" in app.admittance_text()


def test_admittance_requires_gravity_compensation(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "jaka_keyboard_servo.py", "--admittance", "--gravity-compensation-enable", "false",
    ])
    with pytest.raises(SystemExit, match="gravity compensation"):
        keyboard._validate_admittance_args(keyboard._parse_args())

    monkeypatch.setattr(
        sys, "argv", ["jaka_keyboard_servo.py", "--admittance", "--gravity-json", "x.json"],
    )
    keyboard._validate_admittance_args(keyboard._parse_args())  # must not raise

    monkeypatch.setattr(sys, "argv", ["jaka_keyboard_servo.py", "--dry-run"])
    keyboard._validate_admittance_args(keyboard._parse_args())  # not admittance: no-op


def test_orientation_is_preserved_when_no_torque_is_applied(monkeypatch, tmp_path):
    """A non-identity tool pose must survive the clamp unchanged."""

    path, _direction = _write_fit(tmp_path)
    kinematics = LinearKinematics(tool_rotation_deg=35.0)
    app, client = _make_app(
        monkeypatch, tmp_path, *_push_args(path), kinematics=kinematics,
    )
    nominal = app.motion.nominal_pose
    np.testing.assert_allclose(nominal.quaternion_wxyz, kinematics.quaternion, atol=1e-12)

    _run(app, client, 50,
         load=lambda target: _raw_sensor(target, external_force_base=np.array([3.0, 0.0, 0.0])))

    np.testing.assert_allclose(
        app.last_corrected_pose.quaternion_wxyz, nominal.quaternion_wxyz, atol=1e-12
    )
    assert app.servo.target is not None


def test_rotation_offset_is_clamped_at_the_configured_angle(monkeypatch, tmp_path):
    path, _direction = _write_fit(tmp_path)
    kinematics = LinearKinematics(tool_rotation_deg=35.0)
    app, client = _make_app(
        monkeypatch, tmp_path, *_push_args(path),
        "--admittance-axes", "all", "--adm-max-offset-deg", "4",
        "--adm-max-velocity", "1", "1", "1", "1", "1", "1",
        kinematics=kinematics,
    )
    nominal = app.motion.nominal_pose
    _run(app, client, 200,
         load=lambda target: _raw_sensor(target, external_torque_base=np.array([0.0, 0.0, 3.0])))

    relative = (
        rotation_matrix(app.last_corrected_pose.quaternion_wxyz)
        @ rotation_matrix(nominal.quaternion_wxyz).T
    )
    angle_deg = np.rad2deg(np.linalg.norm(rotation_vector_from_matrix(relative)))
    assert 3.9 <= angle_deg <= 4.0 + 1e-6, angle_deg


def test_keyboard_nominal_pose_still_tracks_in_admittance_mode(monkeypatch, tmp_path):
    path, _direction = _write_fit(tmp_path, mass=0.2)
    app, client = _make_app(
        monkeypatch, tmp_path, "--admittance", "--gravity-json", str(path),
    )
    start = app.motion.nominal_pose.position_m.copy()
    app.keys.press("r")  # tool0 +Z insertion axis, no contact
    _run(app, client, 125, load=lambda target: _raw_sensor(target))
    app.keys.release("r")

    assert app.motion.nominal_pose.position_m[2] > start[2] + 0.005
    assert np.linalg.norm(_offset(app)) < 1e-9
    assert app.servo.target is not None
    assert app.servo.target.position_rad[2] > 0.0
    assert app.last_wrench is not None
    assert app.last_wrench.frame_id == app.tool_frame


def test_servo_mode_without_admittance_is_unchanged(monkeypatch, tmp_path):
    """The default (non-compliant) path must keep its original behaviour."""

    app, client = _make_app(
        monkeypatch, tmp_path, "--tare-enable", "false", "--gravity-compensation-enable", "false",
    )
    assert app.admittance_enabled is False
    assert app.motion is None
    assert app.admittance_text() == ""
    assert app.last_external_wrench is None

    client.torque_sensor = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _run(app, client, 20, load=lambda target: np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    # Joint targets are streamed through the original velocity-servo path.
    assert len(client.commands) == 20
    assert app.last_wrench is not None
    assert app.last_wrench.frame_id == app.ft.frame_id


def test_dry_run_without_admittance_never_commands(monkeypatch, tmp_path):
    app, client = _make_app(
        monkeypatch, tmp_path, "--dry-run", "--tare-enable", "false", "--gravity-compensation-enable", "false",
    )
    app.keys.press("w")
    _run(app, client, 20)
    app.keys.release("w")
    assert client.commands == []
    assert client.servo_calls == []
