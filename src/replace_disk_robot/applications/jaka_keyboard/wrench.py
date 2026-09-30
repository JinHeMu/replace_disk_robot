"""Stateful sensor-to-TCP force processing; consumes a supplied EDG sample."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from replace_disk_robot.adapters.jaka import (
    DEFAULT_SENSOR_TO_TOOL_ROTATION, DEFAULT_TOOL_ARM_M, JakaWristFTAdapter, fmt_array,
)
from replace_disk_robot.contact import (
    SensorCompensationConfig, SensorWrenchCompensator, WrenchProcessor, WrenchProcessorConfig,
)
from replace_disk_robot.core import JointState, Pose, Wrench
from replace_disk_robot.core.rotation import quaternion_from_rotation_matrix, rotation_matrix
from replace_disk_robot.contact.force_processing import external_wrench_at_tcp
from .config import PROJECT_ROOT

def _ft_transforms(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.identity_transform:
        return np.eye(3), np.zeros(3)
    return DEFAULT_SENSOR_TO_TOOL_ROTATION, DEFAULT_TOOL_ARM_M


class WrenchPipeline:
    def __init__(self, client, args, kinematics):
        self.args = args
        self.kinematics = kinematics
        self.base_frame = kinematics.base_frame
        self.tool_frame = kinematics.end_effector_frame
        rotation, tool_arm = _ft_transforms(args)
        self.ft = JakaWristFTAdapter(
            client,
            frame_id="tcp_fts_site",
            sensor_to_tool_rotation=rotation,
            tool_arm_m=tool_arm,
            filter_alpha=args.alpha,
            deadband_force_n=args.deadband_force,
            deadband_torque_nm=args.deadband_torque,
        )
        # Online gravity compensation is configured in initialize() when a
        # payload identification JSON is available.
        self.sensor_frame_id = "tcp_fts_sensor"
        self.sensor_to_tool_rotation = np.asarray(
            DEFAULT_SENSOR_TO_TOOL_ROTATION if args.identity_transform else rotation,
            dtype=float,
        ).copy()
        self.tool_to_sensor_m = np.asarray(
            np.zeros(3) if args.identity_transform else tool_arm,
            dtype=float,
        ).copy()
        self.gravity_compensator: SensorWrenchCompensator | None = None
        self.wrench_processor: WrenchProcessor | None = None
        self.payload_mass_kg: float | None = None
        self.payload_com_sensor_m: np.ndarray | None = None
        self.capture_stages = False
        self.stages = {}

    def _configure_gravity_compensation(self) -> None:
        path = Path(self.args.gravity_json).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"gravity identification JSON not found: {path}")
        try:
            fit = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read gravity JSON {path}: {exc}") from exc
        if not isinstance(fit, dict):
            raise ValueError(f"gravity JSON root must be an object: {path}")

        try:
            mass = float(fit["mass_kg"])
            h_base = np.asarray(fit["signed_gravity_force_base_n"], dtype=float)
            com_sensor = np.asarray(fit["center_of_mass_sensor_m"], dtype=float)
            force_bias = np.asarray(fit["force_bias_sensor_n"], dtype=float)
            torque_bias = np.asarray(fit["torque_bias_sensor_nm"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"gravity JSON is missing required fields: {exc}") from exc

        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"gravity JSON mass_kg must be positive, got {mass!r}")
        for name, vector in (
            ("signed_gravity_force_base_n", h_base),
            ("center_of_mass_sensor_m", com_sensor),
            ("force_bias_sensor_n", force_bias),
            ("torque_bias_sensor_nm", torque_bias),
        ):
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError(f"gravity JSON {name} must be a finite 3-vector")

        response = fit.get("sensor_to_tool_rotation")
        if response is None:
            sensor_to_tool = np.asarray(DEFAULT_SENSOR_TO_TOOL_ROTATION, dtype=float)
        else:
            sensor_to_tool = np.asarray(response, dtype=float)
        if sensor_to_tool.shape != (3, 3) or not np.isfinite(sensor_to_tool).all():
            raise ValueError("gravity JSON sensor_to_tool_rotation must be finite 3x3")

        response = fit.get("tool_to_sensor_m")
        if response is None:
            tool_to_sensor = np.asarray(DEFAULT_TOOL_ARM_M, dtype=float)
        else:
            tool_to_sensor = np.asarray(response, dtype=float)
        if tool_to_sensor.shape != (3,) or not np.isfinite(tool_to_sensor).all():
            raise ValueError("gravity JSON tool_to_sensor_m must be a finite 3-vector")

        self.sensor_to_tool_rotation = sensor_to_tool
        self.tool_to_sensor_m = tool_to_sensor
        self.payload_mass_kg = mass
        self.payload_com_sensor_m = com_sensor
        self.gravity_compensator = SensorWrenchCompensator(SensorCompensationConfig(
            frame_id=self.sensor_frame_id,
            load_sign=1.0,
            gravity_frame_id=self.base_frame,
            gravity_m_s2=h_base / mass,
            payload_mass_kg=mass,
            payload_com_sensor_m=com_sensor,
        ))
        # The identification result already contains the static sensor bias.
        self.gravity_compensator.bias_sensor = np.r_[force_bias, torque_bias]
        self.wrench_processor = WrenchProcessor(WrenchProcessorConfig(
            frame_id=self.tool_frame,
            load_sign=1.0,
            filter_alpha=self.args.alpha,
            force_deadband_n=self.args.deadband_force,
            torque_deadband_nm=self.args.deadband_torque,
        ))
        print(
            f"[keyboard] gravity compensation ON: {path} "
            f"mass={mass:.6f} kg, |h|={float(np.linalg.norm(h_base)):.4f} N"
        )

    def _sensor_pose_in_base(self, measured: JointState) -> Pose:
        tool_pose = self.kinematics.forward(measured)
        base_from_tool = rotation_matrix(tool_pose.quaternion_wxyz)
        base_from_sensor = base_from_tool @ self.sensor_to_tool_rotation
        sensor_origin = tool_pose.position_m + base_from_tool @ self.tool_to_sensor_m
        return Pose(
            self.base_frame,
            sensor_origin,
            quaternion_from_rotation_matrix(base_from_sensor),
        )

    def _tare_compensated_wrench(self, state, measured: JointState) -> None:
        """Zero the current gravity-compensated sensor wrench.

        The offset is folded into ``SensorWrenchCompensator.bias_sensor`` so the
        low-pass filter and force guard see the already-tared wrench.
        """

        if self.gravity_compensator is None or self.wrench_processor is None:
            raise RuntimeError("gravity compensation is not configured")
        raw = Wrench(
            self.sensor_frame_id,
            state.torque_sensor[:3].copy(),
            state.torque_sensor[3:].copy(),
        )
        sensor_pose = self._sensor_pose_in_base(measured)
        compensated_sensor = self.gravity_compensator.compensate(
            raw,
            sensor_pose,
            payload_mass_kg=self.payload_mass_kg,
            payload_com_sensor_m=self.payload_com_sensor_m,
        )
        offset = compensated_sensor.as_vector()
        self.gravity_compensator.bias_sensor = (
            self.gravity_compensator.bias_sensor + offset
        )
        self.wrench_processor.reset()
        print(
            "[keyboard] gravity-compensated tare: "
            f"offset={fmt_array(offset, 4)}"
        )

    def _gravity_compensated_wrench(self, state, measured: JointState) -> Wrench:
        if self.gravity_compensator is None or self.wrench_processor is None:
            raise RuntimeError("gravity compensation is not configured")
        raw = Wrench(
            self.sensor_frame_id,
            state.torque_sensor[:3].copy(),
            state.torque_sensor[3:].copy(),
        )
        sensor_pose = self._sensor_pose_in_base(measured)
        compensated_sensor = self.gravity_compensator.compensate(
            raw,
            sensor_pose,
            payload_mass_kg=self.payload_mass_kg,
            payload_com_sensor_m=self.payload_com_sensor_m,
        )
        tool_pose = self.kinematics.forward(measured)
        result = self.wrench_processor.update_tcp(
            compensated_sensor,
            sensor_pose,
            tool_pose,
        )
        if self.capture_stages:
            try:
                external = external_wrench_at_tcp(
                    compensated_sensor, sensor_pose, tool_pose, self.tool_frame,
                    load_sign=self.wrench_processor.config.load_sign,
                )
                self.stages = {
                    "compensated_sensor": compensated_sensor.as_vector().copy(),
                    "compensated_sensor_frame": self.sensor_frame_id,
                    "external_tcp_unfiltered": external.as_vector().copy(),
                    "filtered_before_deadband": self.wrench_processor.state().as_vector(),
                    "processed": result.as_vector().copy(), "frame_id": result.frame_id,
                    "gravity_compensation": True,
                }
            except Exception as exc:
                self.stages = {"capture_error": str(exc)}
        return result

    def _read_wrench(self, state, measured: JointState) -> Wrench:
        self.stages = {}
        if self.gravity_compensator is not None and self.wrench_processor is not None:
            return self._gravity_compensated_wrench(state, measured)
        result = self.ft.read_wrench_from(state)
        if self.capture_stages:
            try:
                sensor = state.torque_sensor - self.ft.bias
                force = self.ft.sensor_to_tool_rotation @ sensor[:3]
                torque = (self.ft.sensor_to_tool_rotation @ sensor[3:]
                          + np.cross(self.ft.tool_arm_m, force))
                self.stages = {
                    "compensated_sensor": sensor.copy(),
                    "compensated_sensor_frame": self.sensor_frame_id,
                    "external_tcp_unfiltered": np.r_[force, torque],
                    "filtered_before_deadband": self.ft._filtered.copy(),
                    "processed": result.as_vector().copy(), "frame_id": result.frame_id,
                    "gravity_compensation": False,
                }
            except Exception as exc:
                self.stages = {"capture_error": str(exc)}
        return result
