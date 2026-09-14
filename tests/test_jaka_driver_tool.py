"""No-hardware tests for examples/jaka_driver_tool/jaka_common.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from jaka_driver_tool.jaka_common import (  # noqa: E402
    EdgState,
    JakaError,
    JakaRobotAdapter,
    JakaWristFTAdapter,
    is_arm_port,
    is_force_torque_port,
)
from replace_disk_robot.core import JointState, Wrench  # noqa: E402


RAW_STATE = (
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    [0.01] * 6,
    [0.02] * 6,
    [100.0, 200.0, 300.0, 0.1, 0.2, 0.3],
    [1.0, 2.0, 3.0, 0.01, 0.02, 0.03],
    (
        [0] * 16,
        [1] * 16,
        [0.0, 0.0],
        [1.0, 2.0],
        [0, 0],
        [1, 1],
        [3.3, 4.4],
    ),
)


class DummyClient:
    def __init__(self, state: EdgState) -> None:
        self.state = state
        self.sent_q = None

    def read_edg_state(self) -> EdgState:
        return self.state

    def edg_servo_j(self, q, step_num: int = 1) -> None:
        self.sent_q = np.asarray(q, dtype=float)


def test_edg_state_parses_all_groups_and_mm_to_m() -> None:
    state = EdgState.from_raw(RAW_STATE)

    np.testing.assert_allclose(state.joint_position_rad, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    np.testing.assert_allclose(state.joint_velocity_rad_s, [0.01] * 6)
    np.testing.assert_allclose(state.joint_torque, [0.02] * 6)
    np.testing.assert_allclose(state.tcp_position_m, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(state.tcp_rpy_rad, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(state.torque_sensor, [1.0, 2.0, 3.0, 0.01, 0.02, 0.03])
    assert len(state.io) == 7


def test_jaka_joint_arm_reorders_named_joints() -> None:
    client = DummyClient(EdgState.from_raw(RAW_STATE))
    arm = JakaRobotAdapter(client)
    target = JointState(
        tuple(reversed(arm.joint_names)),
        list(reversed(range(6))),
    )

    arm.command_joint_positions(target)

    np.testing.assert_allclose(client.sent_q, [0, 1, 2, 3, 4, 5])


def test_jaka_joint_arm_rejects_incomplete_target() -> None:
    client = DummyClient(EdgState.from_raw(RAW_STATE))
    arm = JakaRobotAdapter(client)
    with pytest.raises(ValueError):
        arm.command_joint_positions(JointState(("joint1",), [0.0]))


def test_force_torque_identity_is_exact_with_alpha_one() -> None:
    raw = list(RAW_STATE)
    raw[4] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    client = DummyClient(EdgState.from_raw(tuple(raw)))
    ft = JakaWristFTAdapter(
        client,
        frame_id="jaka_tool",
        sensor_to_tool_rotation=np.eye(3),
        tool_arm_m=np.zeros(3),
        filter_alpha=1.0,
        deadband_force_n=0.0,
        deadband_torque_nm=0.0,
    )

    wrench = ft.read_wrench_from(client.state)

    assert isinstance(wrench, Wrench)
    assert wrench.frame_id == "jaka_tool"
    np.testing.assert_allclose(wrench.as_vector(), [1, 2, 3, 4, 5, 6])


def test_force_torque_filter_and_moment_shift() -> None:
    raw = list(RAW_STATE)
    raw[4] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    client = DummyClient(EdgState.from_raw(tuple(raw)))
    ft = JakaWristFTAdapter(
        client,
        sensor_to_tool_rotation=np.eye(3),
        tool_arm_m=[0.0, 0.0, 0.1],
        filter_alpha=1.0,
        deadband_force_n=0.0,
        deadband_torque_nm=0.0,
    )

    wrench = ft.read_wrench_from(client.state)

    # r x F = [0, 0, 0.1] x [1, 0, 0] = [0, 0.1, 0]
    np.testing.assert_allclose(wrench.as_vector(), [1, 0, 0, 0, 0.1, 0])


def test_ports_are_structurally_detected() -> None:
    client = DummyClient(EdgState.from_raw(RAW_STATE))
    assert is_arm_port(JakaRobotAdapter(client))
    assert is_force_torque_port(JakaWristFTAdapter(client))


def test_jaka_error_formats_code_and_operation() -> None:
    error = JakaError(-12, "edg_get_stat")
    assert error.code == -12
    assert "edg_get_stat" in str(error)
