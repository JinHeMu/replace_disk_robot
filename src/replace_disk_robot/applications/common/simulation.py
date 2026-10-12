"""Shared MuJoCo feedback backend for JAKA applications.

Only feedback/actuation and the model-known F/T compensation differ from EDG.
Real calibration files are never used to compensate simulated sensor readings.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
import mujoco

from replace_disk_robot.adapters.jaka import EdgState
from replace_disk_robot.adapters.mujoco import load_model, reset_keyframe, MujocoRobotAdapter, MujocoWristFTAdapter
from replace_disk_robot.contact import SensorCompensationConfig, SensorWrenchCompensator, WrenchProcessor, WrenchProcessorConfig
from replace_disk_robot.contact.force_processing import external_wrench_at_tcp
from replace_disk_robot.core import Pose, Wrench
from replace_disk_robot.core.rotation import quaternion_from_rotation_matrix, rotation_matrix
from replace_disk_robot.kinematics.jaka import JAKA_JOINT_NAMES
from .wrench import WrenchPipeline


class MujocoJakaClient:
    def __init__(self, keyframe="low", rate_hz=125.):
        self.model, self.data = load_model("jaka")
        reset_keyframe(self.model, self.data, keyframe)
        self.robot = MujocoRobotAdapter(
            self.model, self.data, compensate_bias=True, joint_names=JAKA_JOINT_NAMES,
            actuator_names=tuple(f"joint_{i}_servo" for i in range(1, 7)),
            gripper_actuator_name=None,
        )
        self.ft = MujocoWristFTAdapter(self.model, self.data, "tcp_fts_force", "tcp_fts_torque", "tcp_fts_site")
        self.enabled = False
        self.target = self.robot.arm_position()
        self.command_count = 0
        self.steps_per_tick = round(1. / rate_hz / self.model.opt.timestep)
        self.dt = self.steps_per_tick * self.model.opt.timestep
        if self.steps_per_tick < 1 or not np.isclose(self.dt, 1. / rate_hz):
            raise ValueError("simulation control period must be a multiple of MuJoCo timestep")
        for _ in range(round(.5 / self.model.opt.timestep)):
            self.robot.command_arm(self.target)
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def read_edg_state(self):
        q = self.robot.arm_position()
        return EdgState(q, self.data.qvel[self.robot._dof_ids].copy(),
                        self.data.qfrc_actuator[self.robot._dof_ids].copy(),
                        np.zeros(6), self.ft.raw(), ((),) * 7)

    def servo_move_enable(self, enabled):
        self.enabled = bool(enabled)

    def edg_servo_j(self, q, step_num=1):
        if not self.enabled:
            raise RuntimeError("simulation servo is disabled")
        self.target = np.asarray(q, dtype=float).copy()
        self.robot.command_arm(self.target)
        self.command_count += 1

    def step(self):
        for _ in range(self.steps_per_tick):
            self.robot.command_arm(self.target)
            mujoco.mj_step(self.model, self.data)
        # Synchronize site poses/sensors with the current qpos before feedback.
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("non-finite simulation state")


class MujocoWrenchPipeline(WrenchPipeline):
    def __init__(self, client, args, kinematics):
        super().__init__(client, args, kinematics)
        self.client = client
        self.sensor_frame_id = "tcp_fts_site"
        self.gravity_compensator = SensorWrenchCompensator(SensorCompensationConfig(
            frame_id=self.sensor_frame_id, load_sign=-1.,
            gravity_m_s2=client.model.opt.gravity.copy(),
        ))
        self.wrench_processor = WrenchProcessor(WrenchProcessorConfig(
            frame_id=self.tool_frame, load_sign=-1., filter_alpha=args.alpha,
            force_deadband_n=args.deadband_force, torque_deadband_nm=args.deadband_torque,
        ))
        pose, mass, com = self._sensor_geometry()
        raw = client.ft.raw()
        self.gravity_compensator.tare(Wrench(self.sensor_frame_id, raw[:3], raw[3:]), pose,
                                     payload_mass_kg=mass, payload_com_sensor_m=com)
        self.payload_mass_kg, self.payload_com_sensor_m = mass, com
        base_from_sensor = rotation_matrix(self._sensor_pose_in_base(None).quaternion_wxyz)
        tool = kinematics.forward(client.robot.read_joint_state())
        base_from_tool = rotation_matrix(tool.quaternion_wxyz)
        self.sensor_to_tool_rotation = base_from_tool.T @ base_from_sensor
        self.tool_to_sensor_m = base_from_tool.T @ (
            self._sensor_pose_in_base(None).position_m - tool.position_m)

    def _sensor_geometry(self):
        site = self.client.data.site("tcp_fts_site")
        body_id = self.client.model.site_bodyid[site.id]
        rot = site.xmat.reshape(3, 3)
        pose = Pose("world", site.xpos.copy(), quaternion_from_rotation_matrix(rot))
        mass = float(self.client.model.body_subtreemass[body_id])
        com = rot.T @ (self.client.data.subtree_com[body_id] - site.xpos)
        return pose, mass, com

    def _configure_gravity_compensation(self):
        print("[jaka/sim] model-known payload gravity compensation; no hardware calibration used")

    def _sensor_pose_in_base(self, measured):
        site = self.client.data.site("tcp_fts_site")
        base = self.client.data.body("jaka_base_link")
        rot = base.xmat.reshape(3, 3).T
        return Pose(self.base_frame, rot @ (site.xpos - base.xpos),
                    quaternion_from_rotation_matrix(rot @ site.xmat.reshape(3, 3)))

    def _tare_compensated_wrench(self, state, measured):
        pose, mass, com = self._sensor_geometry()
        raw = Wrench(self.sensor_frame_id, state.torque_sensor[:3], state.torque_sensor[3:])
        self.gravity_compensator.tare(raw, pose, payload_mass_kg=mass, payload_com_sensor_m=com)
        self.wrench_processor.reset()

    def _read_wrench(self, state, measured):
        pose, mass, com = self._sensor_geometry()
        raw = Wrench(self.sensor_frame_id, state.torque_sensor[:3], state.torque_sensor[3:])
        compensated = self.gravity_compensator.compensate(raw, pose, payload_mass_kg=mass,
                                                         payload_com_sensor_m=com)
        result = self.wrench_processor.update_tcp(compensated, self._sensor_pose_in_base(measured),
                                                  self.kinematics.forward(measured))
        external = external_wrench_at_tcp(compensated, self._sensor_pose_in_base(measured),
                                         self.kinematics.forward(measured), self.tool_frame,
                                         load_sign=-1.)
        self.stages = {"compensated_sensor": compensated.as_vector(),
                       "compensated_sensor_frame": self.sensor_frame_id,
                       "external_tcp_unfiltered": external.as_vector(),
                       "processed": result.as_vector(), "frame_id": result.frame_id,
                       "gravity_compensation": True, "backend": "mujoco"}
        return result
