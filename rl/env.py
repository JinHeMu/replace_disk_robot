"""MuJoCo environment for the residual-RL insertion demo.

The environment owns the 20 Hz policy / 100 Hz baseline and servo cadence, but
it never writes ``data.ctrl`` directly.  All motion is submitted through the
existing ``CartesianServo`` and the final joint target is released only through
:class:`~rl.safety.InsertionGuard`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from replace_disk_robot.adapters.mujoco import (
    MujocoRobotAdapter,
    MujocoWristFTAdapter,
    load_model,
    reset_home,
)
from replace_disk_robot.contact import WrenchProcessor, WrenchProcessorConfig
from replace_disk_robot.control import CartesianJog, CartesianServo, ServoConfig
from replace_disk_robot.core.rotation import (
    quaternion_from_rotation_matrix,
    rotation_matrix,
    rotation_matrix_from_rotation_vector,
    rotation_vector_from_matrix,
)
from replace_disk_robot.core.types import JointState, Pose, Wrench
from replace_disk_robot.kinematics import UR5eKinematics
from replace_disk_robot.kinematics.tool import FixedToolKinematics

from .config import RLConfig
from .features import FeatureSnapshot, build_observation, observation_schema
from .insertion import InsertionController
from .residual import (
    ResidualLimitInfo,
    ResidualLimiter,
    compose_taskspace_command,
)
from .reward import RewardInputs, compute_reward
from .safety import GuardConfig, InsertionGuard


ENTRY_SITE = "socket_entry"
DRIVE_CENTER_SITE = "drive_center"
DRIVE_FRONT_SITE = "drive_front"
SENSOR_SITE = "wrist_ft_site"

DRIVE_TOOL_IN_G_BASE = Pose(
    frame_id="g_base",
    position_m=np.array([0.0, 0.0, 0.2135]),
    quaternion_wxyz=np.array([0.5, -0.5, -0.5, -0.5]),
)


@dataclass(frozen=True)
class EnvInfo:
    """Flat metric dictionary returned by reset/step."""

    values: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)


class ResidualInsertionEnv:
    """Gym-like, NumPy-first environment with explicit task-frame contracts."""

    def __init__(
        self,
        config: RLConfig | None = None,
        *,
        seed: int | None = None,
        randomize: bool | None = None,
    ) -> None:
        self.config = config or RLConfig()
        self.randomize = (
            bool(self.config.randomization.enabled) if randomize is None else bool(randomize)
        )
        self.rng = np.random.default_rng(seed)

        self.model, self.data = load_model("ur5e")
        self.data.qfrc_applied[:] = 0.0
        reset_home(self.model, self.data)

        self.robot = MujocoRobotAdapter(self.model, self.data, compensate_bias=True)
        self.ft = MujocoWristFTAdapter(self.model, self.data)
        parent = UR5eKinematics(end_effector_frame="g_base")
        self.kinematics = FixedToolKinematics(parent, "g_base", DRIVE_TOOL_IN_G_BASE)
        self.base_frame = self.kinematics.forward(
            self.robot.read_joint_state()
        ).frame_id
        self.tool_frame = "drive_center"
        self._joint_limits = parent.joint_limits_rad

        insertion = self.config.insertion
        self.servo = CartesianServo(
            self.kinematics,
            self._joint_limits,
            ServoConfig(
                linear_speed_m_s=0.01,
                angular_speed_rad_s=np.deg2rad(5.0),
                max_tracking_error_rad=10.0,
                command_timeout_s=0.15,
                max_dt_s=0.05,
            ),
        )
        self.wrench_processor = WrenchProcessor(
            WrenchProcessorConfig(
                frame_id=ENTRY_SITE,
                load_sign=-1.0,
                filter_alpha=0.25,
                force_deadband_n=0.15,
                torque_deadband_nm=0.02,
            )
        )
        self.controller = InsertionController(insertion)
        self.limiter = ResidualLimiter(self.config.residual)
        self.guard = InsertionGuard(self._guard_config())

        physics_dt = float(self.model.opt.timestep)
        self.steps_per_servo = max(
            1, int(round((1.0 / insertion.control_hz) / physics_dt))
        )
        self.servo_dt = self.steps_per_servo * physics_dt
        self.rl_dt = 1.0 / insertion.rl_hz
        self.servo_per_rl = max(1, int(round(self.rl_dt / self.servo_dt)))
        self.servo_dt = self.rl_dt / self.servo_per_rl
        self._expected_depth_step = insertion.forward_speed_m_s * self.rl_dt

        self._last_observation = np.zeros(32, dtype=float)
        self._last_residual = np.zeros(5, dtype=float)
        self._last_baseline = np.zeros(5, dtype=float)
        self._last_external_wrench = Wrench(ENTRY_SITE, np.zeros(3), np.zeros(3))
        self._previous_wrench = np.zeros(6, dtype=float)
        self._previous_center_position = np.zeros(3, dtype=float)
        self._previous_center_rotation = np.eye(3)
        self._previous_observation_time_s = 0.0
        self._episode_start_time_s = 0.0
        self._previous_depth_m = 0.0
        self._last_limit_info: ResidualLimitInfo | None = None
        self._last_guard_reason = "ok"
        self._stable_count = 0
        self._fault = False
        self._success = False
        self._terminated = False
        self._truncated = False
        self._reset_metrics()

    def _guard_config(self) -> GuardConfig:
        return GuardConfig(
            hard_force_n=10.0,
            hard_torque_nm=1.0,
            max_depth_m=0.075,
            min_depth_m=-0.025,
            max_lateral_error_m=0.010,
            max_angular_error_rad=np.deg2rad(8.0),
            max_tracking_error_rad=10.0,
        )

    @property
    def observation_size(self) -> int:
        return len(self._last_observation)

    @property
    def action_size(self) -> int:
        return 5

    @property
    def observation_schema(self) -> dict[str, Any]:
        return observation_schema()

    def _reset_metrics(self) -> None:
        self._metrics = {
            "episode_time_s": 0.0,
            "depth_m": 0.0,
            "max_depth_m": -np.inf,
            "lateral_error_m": 0.0,
            "angular_error_rad": 0.0,
            "axial_resistance_n": 0.0,
            "peak_force_n": 0.0,
            "peak_torque_nm": 0.0,
            "mean_force_n": 0.0,
            "force_samples": 0,
            "residual_clip_steps": 0,
            "residual_timeout_steps": 0,
            "fault_reason": "",
            "success": False,
            "guard_reason": "ok",
            "baseline_status": "idle",
            "reward_components": {},
        }

    def reset(self, *, seed: int | None = None) -> tuple[NDArray[np.float64], EnvInfo]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        reset_home(self.model, self.data)
        self.data.qfrc_applied[:] = 0.0
        if self.randomize:
            self._apply_randomization()
        self._settle()
        self.ft.tare()
        self.wrench_processor.reset()
        self.controller.reset()
        self.limiter.reset()
        self.guard.reset()
        measured = self.robot.read_joint_state()
        self.servo.reset(measured)

        center = self._site_pose_task(DRIVE_CENTER_SITE)
        self._previous_center_position = center.position_m.copy()
        self._previous_center_rotation = rotation_matrix(center.quaternion_wxyz)
        self._last_external_wrench = Wrench(ENTRY_SITE, np.zeros(3), np.zeros(3))
        self._previous_wrench = np.zeros(6, dtype=float)
        self._last_residual = np.zeros(5, dtype=float)
        self._last_baseline = np.zeros(5, dtype=float)
        self._last_limit_info = None
        self._stable_count = 0
        self._fault = False
        self._success = False
        self._terminated = False
        self._truncated = False
        self._last_guard_reason = "ok"
        self._episode_start_time_s = float(self.data.time)
        self._previous_observation_time_s = float(self.data.time)
        self._reset_metrics()
        observation = self._build_observation(update_previous=True)
        self._last_observation = observation
        state = self._task_state()
        self._previous_depth_m = state["depth_m"]
        self._update_metrics(state)
        return observation, EnvInfo(self._info())

    def step(
        self, normalized_action: ArrayLike
    ) -> tuple[NDArray[np.float64], float, bool, bool, EnvInfo]:
        if self._terminated or self._truncated:
            raise RuntimeError("step() called after episode termination; call reset() first")
        action = np.asarray(normalized_action, dtype=float)
        if action.shape != (5,) or not np.all(np.isfinite(action)):
            raise ValueError("normalized_action must be a finite vector with shape (5,)")

        now = float(self.data.time)
        timeout = self.limiter.check_timeout(now)
        residual, limit_info = self.limiter.limit(action, self.rl_dt, now)
        if timeout:
            residual, limit_info = self.limiter.zero(now, timed_out=True)
        self._last_limit_info = limit_info
        self._last_residual = residual.copy()
        if limit_info.any_clipped:
            self._metrics["residual_clip_steps"] += 1
        if timeout:
            self._metrics["residual_timeout_steps"] += 1

        for _ in range(self.servo_per_rl):
            if self._fault or self._terminated:
                break
            self._run_servo_tick(residual)

        state = self._task_state()
        self._update_metrics(state)
        observation = self._build_observation(update_previous=True)
        self._last_observation = observation

        # Success requires a stable window with the disk inside the task
        # tolerance while residual is not being used to hide a persistent fault.
        success_now = (
            not self._fault
            and state["depth_m"] >= self.config.insertion.success_depth_m
            and state["lateral_error_m"] <= self.config.insertion.max_lateral_error_m
            and state["angular_error_rad"] <= self.config.insertion.max_pitch_yaw_error_rad
        )
        if success_now:
            self._stable_count += 1
        else:
            self._stable_count = 0
        if self._stable_count >= self.config.insertion.stable_steps:
            self._success = True
            self._terminated = True

        if self._fault:
            self._terminated = True
        elapsed = float(self.data.time) - self._episode_start_time_s
        self._metrics["episode_time_s"] = elapsed
        if not self._terminated and elapsed >= self.config.insertion.max_episode_time_s:
            self._truncated = True

        reward = compute_reward(
            RewardInputs(
                previous_depth_m=self._previous_depth_m,
                depth_m=state["depth_m"],
                lateral_error_m=state["lateral_error_m"],
                angular_error_rad=state["angular_error_rad"],
                wrench=self._last_external_wrench,
                normalized_action=np.asarray(action, dtype=float),
                previous_normalized_action=self._last_normalized_action(),
                dt_s=self.rl_dt,
                success=self._success,
                fault=self._fault,
                axial_soft_limit_n=self.config.insertion.axial_soft_limit_n,
                expected_depth_step_m=self._expected_depth_step,
            ),
            self.config.reward,
        )
        self._metrics["reward"] = reward.total
        self._metrics["reward_components"] = reward.as_dict()
        self._metrics["success"] = self._success
        self._previous_depth_m = state["depth_m"]
        return observation, float(reward.total), bool(self._terminated), bool(self._truncated), EnvInfo(self._info())

    def _last_normalized_action(self) -> NDArray[np.float64]:
        limits = np.asarray(self.config.residual.action_limits, dtype=float)
        return np.clip(self._last_residual / limits, -1.0, 1.0)

    def _settle(self) -> None:
        steps = int(round(self.config.insertion.settle_time_s / self.model.opt.timestep))
        for _ in range(max(0, steps)):
            self.robot.command_joint_positions(self.robot.read_joint_state())
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def _apply_randomization(self) -> None:
        """Apply a small C2 initial-offset randomization through IK."""

        cfg = self.config.randomization
        measured = self.robot.read_joint_state()
        nominal = self.kinematics.forward(measured)
        rng = self.rng
        offset = rng.uniform(-1.0, 1.0, size=3) * cfg.lateral_offset_m
        rotations = rng.uniform(-1.0, 1.0, size=3) * cfg.angle_rad
        # Keep the insertion-axis approach roughly aligned; the scene has no
        # back plate and the guard remains the final authority.
        rotations[0] *= 0.25
        rotation = rotation_matrix_from_rotation_vector(rotations)
        target = Pose(
            nominal.frame_id,
            nominal.position_m + rotation_matrix(nominal.quaternion_wxyz) @ offset,
            quaternion_from_rotation_matrix(
                rotation @ rotation_matrix(nominal.quaternion_wxyz)
            ),
        )
        try:
            solution = self.kinematics.inverse(target, measured)
        except RuntimeError:
            return
        self.data.qpos[:6] = solution.position_rad
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _run_servo_tick(self, residual: NDArray[np.float64]) -> None:
        now = float(self.data.time)
        measured = self.robot.read_joint_state()
        measured_pose_task = self._site_pose_task(DRIVE_CENTER_SITE)
        external = self._process_wrench_once()
        baseline = self.controller.update(measured_pose_task, external, self.servo_dt)
        command = compose_taskspace_command(
            baseline.command,
            residual,
            max_forward_m_s=self.config.insertion.max_command_forward_m_s,
            max_lateral_m_s=self.config.insertion.max_command_lateral_m_s,
            max_angular_rad_s=self.config.insertion.max_command_angular_rad_s,
        )
        self._last_baseline = baseline.command.copy()
        self._metrics["baseline_status"] = baseline.status
        self._metrics["axial_resistance_n"] = baseline.axial_resistance_n

        jog = self._task_command_to_jog(command, measured)
        self.servo.submit(jog, now)
        requested = self.servo.update(measured, self.servo_dt, now)
        if self.servo.fault:
            self._fault = True
            self._metrics["fault_reason"] = self.servo.fault
            return

        state = self._task_state()
        tracking_error = float(
            np.max(np.abs(measured.position_rad - requested.position_rad))
        ) if requested.position_rad.size else None

        q_command = requested
        for _ in range(self.steps_per_servo):
            current = self.robot.read_joint_state()
            guard_result = self.guard.update(
                external.as_vector(),
                current.position_rad,
                q_command.position_rad,
                depth_m=state["depth_m"],
                lateral_error_m=state["lateral_error_m"],
                angular_error_rad=state["angular_error_rad"],
                tracking_error_rad=tracking_error,
            )
            q_command = JointState(requested.names, guard_result.joint_target)
            self._last_guard_reason = guard_result.reason
            if guard_result.fault:
                self._fault = True
                self._metrics["fault_reason"] = guard_result.reason
                q_command = current
                self.robot.command_joint_positions(q_command)
                break
            self.robot.command_joint_positions(q_command)
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(
                np.concatenate((self.data.qpos, self.data.qvel, self.data.sensordata))
            ).all():
                self._fault = True
                self._metrics["fault_reason"] = "non_finite_state"
                return

    def _process_wrench_once(self) -> Wrench:
        raw = self.ft.read_wrench()
        sensor_pose = self._site_pose_task(SENSOR_SITE)
        tcp_pose = self._site_pose_task(DRIVE_CENTER_SITE)
        external = self.wrench_processor.update(raw, sensor_pose, tcp_pose.position_m)
        self._last_external_wrench = external
        return external

    def _task_command_to_jog(
        self, command: NDArray[np.float64], measured: JointState
    ) -> CartesianJog:
        entry_rotation = self.data.site(ENTRY_SITE).xmat.reshape(3, 3)
        measured_pose = self.kinematics.forward(measured)
        tool_rotation = rotation_matrix(measured_pose.quaternion_wxyz)
        linear_world = entry_rotation @ command[:3]
        angular_task = np.array([0.0, command[3], command[4]])
        angular_tcp = tool_rotation.T @ entry_rotation @ angular_task
        return CartesianJog(
            base_frame=self.base_frame,
            linear_m_s=linear_world,
            angular_rad_s=angular_tcp,
        )

    def _site_pose_task(self, name: str) -> Pose:
        entry = self.data.site(ENTRY_SITE)
        entry_rotation = entry.xmat.reshape(3, 3)
        site = self.data.site(name)
        position = entry_rotation.T @ (site.xpos - entry.xpos)
        rotation = entry_rotation.T @ site.xmat.reshape(3, 3)
        return Pose(
            ENTRY_SITE,
            position,
            quaternion_from_rotation_matrix(rotation),
        )

    def _site_position_task(self, name: str) -> NDArray[np.float64]:
        entry = self.data.site(ENTRY_SITE)
        return entry.xmat.reshape(3, 3).T @ (self.data.site(name).xpos - entry.xpos)

    def _task_state(self) -> dict[str, float]:
        center = self._site_pose_task(DRIVE_CENTER_SITE)
        front_position = self._site_position_task(DRIVE_FRONT_SITE)
        center_rotation = rotation_matrix(center.quaternion_wxyz)
        rotation_vector = rotation_vector_from_matrix(center_rotation)
        lateral_error = float(np.linalg.norm(center.position_m[1:3]))
        angular_error = float(np.linalg.norm(rotation_vector[[1, 2]]))
        return {
            "depth_m": float(front_position[0]),
            "center_position_m": center.position_m.copy(),
            "center_rotation": center_rotation.copy(),
            "lateral_offset_m": center.position_m[1:3].copy(),
            "pitch_yaw_rad": rotation_vector[[1, 2]].copy(),
            "lateral_error_m": lateral_error,
            "angular_error_rad": angular_error,
        }

    def _build_observation(self, *, update_previous: bool) -> NDArray[np.float64]:
        now = float(self.data.time)
        state = self._task_state()
        dt = max(now - self._previous_observation_time_s, self.rl_dt)
        center_position = state["center_position_m"]
        center_rotation = state["center_rotation"]
        linear_velocity = (center_position - self._previous_center_position) / dt
        relative_rotation = self._previous_center_rotation.T @ center_rotation
        angular_vector = rotation_vector_from_matrix(relative_rotation) / dt
        wrench_vector = self._last_external_wrench.as_vector()
        wrench_rate = (wrench_vector - self._previous_wrench) / max(
            self.config.observation.wrench_rate_time_s, dt
        )
        snapshot = FeatureSnapshot(
            depth_m=state["depth_m"],
            lateral_offset_m=state["lateral_offset_m"],
            pitch_yaw_rad=state["pitch_yaw_rad"],
            tcp_velocity_m_s=linear_velocity,
            tcp_angular_velocity_rad_s=angular_vector[[1, 2]],
            wrench=self._last_external_wrench,
            wrench_rate=Wrench(
                ENTRY_SITE,
                wrench_rate[:3],
                wrench_rate[3:],
            ),
            baseline_command=self._last_baseline,
            last_residual=self._last_residual,
        )
        observation = build_observation(snapshot, self.config.observation)
        if update_previous:
            self._previous_center_position = center_position.copy()
            self._previous_center_rotation = center_rotation.copy()
            self._previous_wrench = wrench_vector.copy()
            self._previous_observation_time_s = now
        return observation

    def _update_metrics(self, state: dict[str, float]) -> None:
        self._metrics["depth_m"] = state["depth_m"]
        self._metrics["max_depth_m"] = max(
            float(self._metrics["max_depth_m"]), state["depth_m"]
        )
        self._metrics["lateral_error_m"] = state["lateral_error_m"]
        self._metrics["angular_error_rad"] = state["angular_error_rad"]
        force = self._last_external_wrench.as_vector()
        force_norm = float(np.linalg.norm(force[:3]))
        torque_norm = float(np.linalg.norm(force[3:]))
        self._metrics["peak_force_n"] = max(self._metrics["peak_force_n"], force_norm)
        self._metrics["peak_torque_nm"] = max(self._metrics["peak_torque_nm"], torque_norm)
        count = int(self._metrics["force_samples"])
        self._metrics["mean_force_n"] = (
            (self._metrics["mean_force_n"] * count) + force_norm
        ) / (count + 1)
        self._metrics["force_samples"] = count + 1
        self._metrics["success"] = self._success
        self._metrics["guard_reason"] = self._last_guard_reason

    def _info(self) -> dict[str, Any]:
        info = dict(self._metrics)
        info.update(
            {
                "fault": self._fault,
                "terminated": self._terminated,
                "truncated": self._truncated,
                "observation_schema": observation_schema(),
                "baseline_command": self._last_baseline.tolist(),
                "residual_command": self._last_residual.tolist(),
                "external_wrench_task": self._last_external_wrench.as_vector().tolist(),
                "guard_reason": self._last_guard_reason,
            }
        )
        return info


__all__ = ["EnvInfo", "ResidualInsertionEnv"]
