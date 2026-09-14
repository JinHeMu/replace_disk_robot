"""JAKA Python SDK adapters for the backend-neutral core contracts.

* :class:`JakaEdgClient` owns the SDK ``RC`` object and EDG lifecycle.
* :class:`EdgState` converts the tuple returned by ``edg_get_stat`` into
  named NumPy arrays.
* :class:`JakaRobotAdapter` implements ``core.ports.ArmPort``.
* :class:`JakaWristFTAdapter` implements ``core.ports.ForceTorquePort`` and
  mirrors the compensation/filtering logic used by the WBMM hardware
  interface.

All public methods raise :class:`JakaError` when the SDK returns a non-zero
error code.  The scripts under ``examples/jaka_driver_tool`` add read-only and
dry-run modes on top of this module.
"""

from __future__ import annotations

import ctypes
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

JAKA_SDK_DIR = (
    Path(__file__).resolve().parent
    / "jaka_driver"
    / "x86_64-linux-gnu"
)

from ...core import JointState, Wrench
from ...core.ports import ArmPort, ForceTorquePort


# JAKA MoveMode enum values.  The Python wrapper does not export the enum.
ABS = 0
INCR = 1
CONTINUE = 2
STOP = 3

# JAKA CoordType enum values.
COORD_BASE = 0
COORD_JOINT = 1
COORD_TOOL = 2

JAKA_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)

# Default transform from WBMM jaka_hardware_interface.cpp.
DEFAULT_SENSOR_TO_TOOL_ROTATION = np.array(
    [
        [0.707, -0.707, 0.000],
        [0.664, 0.664, 0.342],
        [-0.242, -0.242, 0.940],
    ],
    dtype=float,
)
DEFAULT_TOOL_ARM_M = np.array([0.000, -0.05, -0.4], dtype=float)


class JakaError(RuntimeError):
    """Non-zero return code from the JAKA SDK."""

    def __init__(self, code: int, operation: str) -> None:
        self.code = int(code)
        self.operation = operation
        super().__init__(f"JAKA SDK {operation} failed with code {self.code}")


def _result_code(result: Any) -> int:
    if not isinstance(result, tuple) or not result:
        raise RuntimeError(f"unexpected JAKA SDK return value: {result!r}")
    try:
        return int(result[0])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"unexpected JAKA SDK return code: {result!r}") from exc


def _check_result(result: Any, operation: str) -> Any:
    code = _result_code(result)
    if code != 0:
        raise JakaError(code, operation)
    return result


def _six(value: Sequence[float], name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if array.shape != (6,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 6-vector")
    return array


def load_jkrc() -> Any:
    """Load the vendored ``jkrc`` extension.

    ``jkrc.so`` depends on ``libjakaAPI.so``; preloading it with
    ``RTLD_GLOBAL`` makes the import work even when ``LD_LIBRARY_PATH`` was
    not set before Python started.
    """

    if str(JAKA_SDK_DIR) not in sys.path:
        sys.path.insert(0, str(JAKA_SDK_DIR))
    library = JAKA_SDK_DIR / "libjakaAPI.so"
    if library.is_file():
        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    import jkrc  # type: ignore[import-not-found]

    return jkrc


@dataclass(frozen=True)
class EdgState:
    """Parsed ``EDGState`` from ``RC.edg_get_stat(0)``.

    The JAKA Python wrapper returns::

        (joint_val, joint_vel, joint_torque, cart_pose, torq_sensor, io)

    where the first five items are 6-tuples and ``io`` is a 7-tuple of
    cabinet/tool digital and analog IO tuples.
    """

    joint_position_rad: NDArray[np.float64]
    joint_velocity_rad_s: NDArray[np.float64]
    joint_torque: NDArray[np.float64]
    cartesian_pose: NDArray[np.float64]
    torque_sensor: NDArray[np.float64]
    io: tuple[Any, ...]

    @classmethod
    def from_raw(cls, raw: Sequence[Any]) -> "EdgState":
        if not isinstance(raw, (tuple, list)) or len(raw) != 6:
            raise ValueError("EDG state must be a 6-tuple")
        io = raw[5]
        if not isinstance(io, (tuple, list)) or len(io) != 7:
            raise ValueError("EDG IO state must be a 7-tuple")
        return cls(
            joint_position_rad=_six(raw[0], "joint_position_rad"),
            joint_velocity_rad_s=_six(raw[1], "joint_velocity_rad_s"),
            joint_torque=_six(raw[2], "joint_torque"),
            cartesian_pose=_six(raw[3], "cartesian_pose"),
            torque_sensor=_six(raw[4], "torque_sensor"),
            io=tuple(io),
        )

    @property
    def tcp_position_m(self) -> NDArray[np.float64]:
        return self.cartesian_pose[:3] / 1000.0

    @property
    def tcp_rpy_rad(self) -> NDArray[np.float64]:
        return self.cartesian_pose[3:].copy()


class JakaEdgClient:
    """Small lifecycle wrapper around ``jkrc.RC`` and EDG mode."""

    def __init__(
        self,
        robot_ip: str,
        local_ip: str,
        edg_port: int = 10010,
        edg_mode: int = 0,
    ) -> None:
        if not robot_ip:
            raise ValueError("robot_ip must not be empty")
        if not local_ip:
            raise ValueError("local_ip must not be empty")
        self.robot_ip = robot_ip
        self.local_ip = local_ip
        self.edg_port = int(edg_port)
        self.edg_mode = int(edg_mode)
        self._jkrc = load_jkrc()
        self._robot = self._jkrc.RC(robot_ip)
        self._logged_in = False
        self._edg_enabled = False
        self._servo_enabled = False

    @property
    def robot(self) -> Any:
        """The raw ``jkrc.RC`` object for advanced or one-off calls."""

        return self._robot

    @property
    def logged_in(self) -> bool:
        return self._logged_in

    @property
    def edg_enabled(self) -> bool:
        return self._edg_enabled

    def _call(self, name: str, *args: Any) -> Any:
        function = getattr(self._robot, name)
        result = function(*args)
        _check_result(result, name)
        return result

    def _call_data(self, name: str, *args: Any) -> Any:
        result = getattr(self._robot, name)(*args)
        _check_result(result, name)
        if len(result) < 2:
            raise RuntimeError(f"JAKA SDK {name} returned no data: {result!r}")
        return result[1]

    # ------------------------------------------------------------------
    # Connection and robot power lifecycle
    # ------------------------------------------------------------------
    def login(self) -> None:
        self._call("login")
        self._logged_in = True

    def logout(self) -> None:
        if not self._logged_in:
            return
        try:
            self._call("logout")
        finally:
            self._logged_in = False

    def power_on(self) -> None:
        self._call("power_on")

    def power_off(self) -> None:
        self._call("power_off")

    def enable_robot(self) -> None:
        self._call("enable_robot")

    def disable_robot(self) -> None:
        self._call("disable_robot")

    def shut_down(self) -> None:
        self._call("shut_down")

    def set_joint_lpf(self, cutoff_hz: float) -> None:
        self._call("servo_move_use_joint_LPF", float(cutoff_hz))

    def set_torque_sensor_mode(self, mode: int) -> None:
        self._call("set_torque_sensor_mode", int(mode))

    # ------------------------------------------------------------------
    # EDG
    # ------------------------------------------------------------------
    def edg_enable(self, enable: bool = True) -> None:
        """Enable/disable EDG, using the full 4-argument call.

        C++ uses ``edg_init(true, local_ip, 10010, 0)``; in Python that is
        ``edg_init_extend(...)``.
        """

        result = self._robot.edg_init_extend(
            bool(enable),
            self.local_ip,
            self.edg_port,
            self.edg_mode,
        )
        _check_result(result, "edg_init_extend")
        self._edg_enabled = bool(enable)

    def edg_disable(self, ignore_errors: bool = False) -> None:
        try:
            self.edg_enable(False)
        except JakaError:
            if not ignore_errors:
                raise
        finally:
            self._edg_enabled = False

    def read_edg_state(self) -> EdgState:
        """Read one EDG state tuple and return a parsed :class:`EdgState`."""

        raw = self._call_data("edg_get_stat", 0)
        return EdgState.from_raw(raw)

    def edg_stat_details(self) -> Any:
        return self._call_data("edg_stat_details")

    # ------------------------------------------------------------------
    # Direct state getters (no EDG stream required)
    # ------------------------------------------------------------------
    def get_joint_position(self) -> NDArray[np.float64]:
        return _six(self._call_data("get_joint_position"), "get_joint_position")

    def get_tcp_position(self) -> NDArray[np.float64]:
        return _six(self._call_data("get_tcp_position"), "get_tcp_position")

    def is_in_servomove(self) -> bool:
        data = self._call_data("is_in_servomove")
        if isinstance(data, (tuple, list)):
            return bool(data[0])
        return bool(data)

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def servo_move_enable(self, enable: bool = True) -> None:
        self._call("servo_move_enable", bool(enable))
        self._servo_enabled = bool(enable)

    def edg_servo_j(
        self,
        joint_position_rad: Sequence[float],
        step_num: int = 1,
        robot_index: int = 0,
        move_mode: int = ABS,
    ) -> None:
        q = _six(joint_position_rad, "joint_position_rad")
        self._call(
            "edg_servo_j",
            q.tolist(),
            int(move_mode),
            int(step_num),
            int(robot_index),
        )

    def joint_move(
        self,
        joint_position_rad: Sequence[float],
        speed_rad_s: float,
        *,
        is_block: bool = True,
        acc_rad_s2: float | None = None,
        tol_mm: float | None = None,
    ) -> None:
        q = _six(joint_position_rad, "joint_position_rad")
        args: list[Any] = [q.tolist(), ABS, bool(is_block), float(speed_rad_s)]
        if acc_rad_s2 is not None:
            args.append(float(acc_rad_s2))
        if tol_mm is not None:
            if acc_rad_s2 is None:
                raise ValueError("tol_mm requires acc_rad_s2")
            args.append(float(tol_mm))
        self._call("joint_move", *args)

    # ------------------------------------------------------------------
    # Direct F/T helpers (optional; EDG.torque_sensor is the main path)
    # ------------------------------------------------------------------
    def get_ft_sensor_data(
        self,
        sensor_id: int,
        data_type: int = 3,
    ) -> tuple[int, NDArray[np.float64]]:
        """Return ``(status, [fx,fy,fz,tx,ty,tz])`` from the direct FT API.

        ``data_type`` follows the SDK: 2=original, 3=gravity/bias removed.
        """

        data = self._call_data("get_ft_sensor_data", int(sensor_id), int(data_type))
        status = int(data[0])
        ft = np.asarray(data[1], dtype=float)
        if ft.shape != (6,):
            raise RuntimeError(f"unexpected FT payload: {data!r}")
        return status, ft

    def set_ft_sensor_zero(self, sensor_id: int) -> None:
        self._call("set_ft_sensor_zero", int(sensor_id))


class JakaRobotAdapter(ArmPort):
    """``core.ports.ArmPort`` adapter backed by EDG feedback/servo."""

    def __init__(
        self,
        client: JakaEdgClient,
        joint_names: Sequence[str] = JAKA_JOINT_NAMES,
    ) -> None:
        names = tuple(joint_names)
        if len(names) != 6 or len(set(names)) != 6:
            raise ValueError("joint_names must contain six unique names")
        self.client = client
        self.joint_names = names

    def _ordered(self, target: JointState) -> NDArray[np.float64]:
        if (
            len(target.names) != len(self.joint_names)
            or len(set(target.names)) != len(self.joint_names)
            or set(target.names) != set(self.joint_names)
        ):
            raise ValueError("target must contain each JAKA joint exactly once")
        index = {name: i for i, name in enumerate(target.names)}
        return np.array([target.position_rad[index[name]] for name in self.joint_names])

    def read_joint_state(self) -> JointState:
        state = self.client.read_edg_state()
        return JointState(self.joint_names, state.joint_position_rad)

    def read_joint_velocity(self) -> NDArray[np.float64]:
        return self.client.read_edg_state().joint_velocity_rad_s

    def command_joint_positions(self, target: JointState) -> None:
        self.client.edg_servo_j(self._ordered(target), step_num=1)


class JakaWristFTAdapter(ForceTorquePort):
    """``core.ports.ForceTorquePort`` over the EDG torque sensor.

    The compensation and filtering reproduce the WBMM hardware-interface
    behavior: subtract a static bias, rotate the sensor wrench into the tool
    frame, shift the moment by ``r x F``, low-pass filter, then apply deadband.
    """

    def __init__(
        self,
        client: JakaEdgClient,
        *,
        frame_id: str = "jaka_tool",
        sensor_to_tool_rotation: NDArray[np.float64] | None = None,
        tool_arm_m: Sequence[float] | None = None,
        filter_alpha: float = 0.2,
        deadband_force_n: float = 1.0,
        deadband_torque_nm: float = 0.2,
    ) -> None:
        if not frame_id:
            raise ValueError("frame_id must not be empty")
        if not 0.0 < filter_alpha <= 1.0:
            raise ValueError("filter_alpha must be in (0, 1]")
        self.client = client
        self.frame_id = frame_id
        rotation = (
            DEFAULT_SENSOR_TO_TOOL_ROTATION.copy()
            if sensor_to_tool_rotation is None
            else np.asarray(sensor_to_tool_rotation, dtype=float)
        )
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("sensor_to_tool_rotation must be a finite 3x3 matrix")
        arm = (
            DEFAULT_TOOL_ARM_M.copy()
            if tool_arm_m is None
            else np.asarray(tool_arm_m, dtype=float)
        )
        if arm.shape != (3,) or not np.isfinite(arm).all():
            raise ValueError("tool_arm_m must be a finite 3-vector")
        self.sensor_to_tool_rotation = rotation
        self.tool_arm_m = arm
        self.filter_alpha = float(filter_alpha)
        self.deadband_force_n = float(deadband_force_n)
        self.deadband_torque_nm = float(deadband_torque_nm)
        self.bias = np.zeros(6)
        self._filtered = np.zeros(6)

    def tare(self, samples: int = 50, period_s: float = 0.01) -> NDArray[np.float64]:
        """Estimate the static sensor bias from EDG readings."""

        if samples <= 0:
            raise ValueError("samples must be positive")
        if period_s < 0:
            raise ValueError("period_s must be non-negative")
        total = np.zeros(6)
        valid = 0
        for index in range(samples):
            try:
                total += self.client.read_edg_state().torque_sensor
                valid += 1
            except Exception:
                pass
            if index + 1 < samples and period_s > 0:
                time.sleep(period_s)
        if valid == 0:
            raise RuntimeError("FT tare failed: no valid EDG samples")
        self.bias = total / valid
        self._filtered[:] = 0.0
        return self.bias.copy()

    def reset_filter(self) -> None:
        self._filtered[:] = 0.0

    def read_wrench_from(self, state: EdgState) -> Wrench:
        """Process an already-read EDG state without another UDP read."""

        raw = state.torque_sensor.copy()
        force_sensor = raw[:3] - self.bias[:3]
        torque_sensor = raw[3:] - self.bias[3:]
        force_tool = self.sensor_to_tool_rotation @ force_sensor
        torque_tool = (
            self.sensor_to_tool_rotation @ torque_sensor
            + np.cross(self.tool_arm_m, force_tool)
        )
        compensated = np.r_[force_tool, torque_tool]
        self._filtered = (
            self.filter_alpha * compensated
            + (1.0 - self.filter_alpha) * self._filtered
        )
        output = self._filtered.copy()
        if abs(output[0]) < self.deadband_force_n:
            output[0] = 0.0
        if abs(output[1]) < self.deadband_force_n:
            output[1] = 0.0
        if abs(output[2]) < self.deadband_force_n:
            output[2] = 0.0
        if abs(output[3]) < self.deadband_torque_nm:
            output[3] = 0.0
        if abs(output[4]) < self.deadband_torque_nm:
            output[4] = 0.0
        if abs(output[5]) < self.deadband_torque_nm:
            output[5] = 0.0
        return Wrench(self.frame_id, output[:3], output[3:])

    def read_wrench(self) -> Wrench:
        return self.read_wrench_from(self.client.read_edg_state())


def fmt_array(values: Sequence[float], decimals: int = 4) -> str:
    """Small formatting helper shared by the example scripts."""

    return "[" + ", ".join(f"{float(value):+.{decimals}f}" for value in values) + "]"


def is_arm_port(candidate: Any) -> bool:
    """Return whether *candidate* structurally looks like ``ArmPort``."""

    return all(hasattr(candidate, name) for name in ("read_joint_state", "command_joint_positions"))


def is_force_torque_port(candidate: Any) -> bool:
    """Return whether *candidate* structurally looks like ``ForceTorquePort``."""

    return all(hasattr(candidate, name) for name in ("tare", "read_wrench"))


# Compatibility aliases used by examples and tests.
JakaJointArm = JakaRobotAdapter
JakaForceTorque = JakaWristFTAdapter

__all__ = [
    "ABS",
    "CONTINUE",
    "COORD_BASE",
    "COORD_JOINT",
    "COORD_TOOL",
    "DEFAULT_SENSOR_TO_TOOL_ROTATION",
    "DEFAULT_TOOL_ARM_M",
    "EdgState",
    "INCR",
    "JAKA_JOINT_NAMES",
    "JakaEdgClient",
    "JakaError",
    "JakaForceTorque",
    "JakaJointArm",
    "JakaRobotAdapter",
    "JakaWristFTAdapter",
    "STOP",
    "fmt_array",
    "is_arm_port",
    "is_force_torque_port",
    "load_jkrc",
]
