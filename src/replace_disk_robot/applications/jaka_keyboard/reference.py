"""Keyboard/reference composition using the existing control algorithms."""
from __future__ import annotations
import numpy as np
from replace_disk_robot.control import (
    AdmittanceConfig, AdmittanceController, CartesianJog, KeyControl,
    KeyboardAdmittanceController, MotionReferenceConfig,
)
from replace_disk_robot.core import JointState, Pose, Wrench
from replace_disk_robot.core.rotation import (
    multiply as quaternion_multiply, quaternion_from_rotation_vector,
    rotation_matrix, rotation_vector_from_matrix,
)

def _base_translation_from_keys(
    pressed: set[str],
    tool_z_in_base: np.ndarray,
    speed_m_s: float,
) -> np.ndarray:
    """Build the real-robot base-mode translation command.

    The arm mount ``jaka_base_link`` is yawed -90 degrees relative to the
    mobile platform.  Platform left/right is therefore jaka-base -X/+X.
    R/F deliberately remain on the current tool +Z insertion axis.
    """

    tool_z = np.asarray(tool_z_in_base, dtype=float).copy()
    norm = float(np.linalg.norm(tool_z))
    if tool_z.shape != (3,) or not np.isfinite(tool_z).all() or norm < 1e-12:
        raise ValueError("tool_z_in_base must be a finite non-zero 3-vector")
    tool_z /= norm

    linear = np.zeros(3)
    if "w" in pressed:
        linear += np.array([0.0, 0.0, 1.0])
    if "s" in pressed:
        linear -= np.array([0.0, 0.0, 1.0])
    if "a" in pressed:
        linear -= np.array([1.0, 0.0, 0.0])
    if "d" in pressed:
        linear += np.array([1.0, 0.0, 0.0])
    if "r" in pressed:
        linear += tool_z
    if "f" in pressed:
        linear -= tool_z
    linear /= max(1.0, float(np.linalg.norm(linear)))
    return linear * float(speed_m_s)


class ReferencePipeline:
    def __init__(self, args, kinematics, arm, servo):
        self.args, self.kinematics, self.arm, self.servo = args, kinematics, arm, servo
        self.base_frame, self.tool_frame = kinematics.base_frame, kinematics.end_effector_frame
        self.command_frame = self.tool_frame if args.command_frame == "tool" else self.base_frame
        self.keys = KeyControl(
            args.linear_speed,
            np.deg2rad(args.angular_speed_deg),
            base_frame=self.command_frame,
        )
        self.admittance_enabled = bool(args.admittance)
        self.motion: KeyboardAdmittanceController | None = None
        self.last_external_wrench: Wrench | None = None
        self.last_corrected_pose: Pose | None = None
        if self.admittance_enabled:
            self.admittance = AdmittanceController(AdmittanceConfig(
                frame_id=self.tool_frame,
                mass=args.adm_mass,
                damping=args.adm_damping,
                stiffness=args.adm_stiffness,
                max_velocity=args.adm_max_velocity,
                max_dt_s=max(0.01, 4.0 / args.rate_hz),
            ))
            enabled_axes = np.array(
                [True, True, True, False, False, False]
                if args.admittance_axes == "translation" else [True] * 6,
                dtype=bool,
            )
            axis_index = {"x": 0, "y": 1, "z": 2, "rx": 3, "ry": 4, "rz": 5}
            for axis in getattr(args, "adm_disable_axes", ()):
                enabled_axes[axis_index[axis]] = False
            self.motion_config = MotionReferenceConfig(
                enabled_axes=enabled_axes,
                max_dt_s=max(0.01, 4.0 / args.rate_hz),
            )
    def _reset_compliance(self, measured: JointState | None) -> None:
        """Re-seed nominal/admittance state at the measured pose.

        Called on start, resume and every stop so that a fault recovery never
        replays a stale compliant offset.
        """

        if not self.admittance_enabled:
            return False
        if measured is None:
            try:
                measured = self.arm.read_joint_state()
            except Exception:  # noqa: BLE001 - keep the previous reference
                return False
        pose = self.kinematics.forward(measured)
        if self.motion is None:
            self.motion = KeyboardAdmittanceController(
                self.admittance, pose, self.motion_config, tcp_frame_id=self.tool_frame,
            )
        else:
            self.motion.reset(pose)
        self.last_external_wrench = None
        self.last_corrected_pose = pose
        return True

    def _clamp_offset(self, corrected: Pose, nominal: Pose) -> Pose:
        """Limit the compliant displacement relative to the nominal pose.

        ``AdmittanceController`` intentionally has no offset clamp; on hardware
        the caller owns that limit.
        """

        delta = np.asarray(corrected.position_m, dtype=float) - np.asarray(
            nominal.position_m, dtype=float
        )
        distance = float(np.linalg.norm(delta))
        limit_m = float(self.args.adm_max_offset_m)
        position = (
            np.asarray(nominal.position_m, dtype=float) + delta * (limit_m / distance)
            if distance > limit_m else np.asarray(corrected.position_m, dtype=float)
        )

        relative = (
            rotation_matrix(corrected.quaternion_wxyz)
            @ rotation_matrix(nominal.quaternion_wxyz).T
        )
        rotation_vector = rotation_vector_from_matrix(relative)
        angle = float(np.linalg.norm(rotation_vector))
        limit_rad = float(np.deg2rad(self.args.adm_max_offset_deg))
        if angle > limit_rad:
            # Scale the *relative* rotation and apply it to the nominal pose;
            # the corrected quaternion is already nominal-composed.
            delta_quaternion = quaternion_from_rotation_vector(
                rotation_vector * (limit_rad / angle)
            )
            quaternion = quaternion_multiply(delta_quaternion, np.asarray(
                nominal.quaternion_wxyz, dtype=float
            ))
            quaternion = quaternion / np.linalg.norm(quaternion)
        else:
            quaternion = np.asarray(corrected.quaternion_wxyz, dtype=float)
        return Pose(nominal.frame_id, position, quaternion)

    def _update_compliance(self, jog, wrench, tcp_pose, dt_s) -> Pose:
        """One keyboard-nominal + admittance step; returns the clamped pose."""

        if self.motion is None:
            raise RuntimeError("admittance is enabled but the motion reference is not seeded")
        corrected = self.motion.update(jog, wrench, tcp_pose, dt_s)
        self.last_corrected_pose = self._clamp_offset(corrected, self.motion.nominal_pose)
        return self.last_corrected_pose

    def admittance_text(self) -> str:
        """Compact admittance state for the status line and window title."""

        if not self.admittance_enabled:
            return ""
        force = 0.0
        if self.last_external_wrench is not None:
            force = float(np.linalg.norm(self.last_external_wrench.force_n))
        offset = self.motion.state().offset if self.motion is not None else np.zeros(6)
        axes = self.motion.enabled_axes if self.motion is not None else np.zeros(6, dtype=bool)
        return (
            f"|Fext|={force:5.2f}N "
            f"dx={np.linalg.norm(offset[:3]) * 1000.0:4.1f}mm "
            f"dth={np.rad2deg(np.linalg.norm(offset[3:])):4.1f}deg "
            f"axes={''.join('1' if value else '0' for value in axes)}"
        )

    def _jog_to_servo(self, reference: JointState | Pose | None = None) -> CartesianJog:
        """Map keyboard keys to Servo's base-linear/intrinsic-TCP convention.

        * tool mode: W/S/A/D and rotations use the current tool0 frame.  R/F
          still follow the current tool0 +Z/-Z insertion axis.
        * base mode: W/S use base +/-Z; A/D use base -X/+X, which corresponds
          to platform left/right for the -90-degree arm mounting; R/F move
          along current tool0 +Z/-Z.  Base-axis rotations are converted to
          intrinsic TCP angular velocity.

        ``reference`` overrides the pose the keys are interpreted in; the
        admittance path passes the *nominal* pose so that compliant offsets do
        not feed back into the keyboard frame.
        """
        if reference is None:
            target = self.servo.target
            if target is None:
                target = self.arm.read_joint_state()
            pose = self.kinematics.forward(target)
        elif isinstance(reference, Pose):
            pose = reference
        else:
            pose = self.kinematics.forward(reference)
        base_from_tool = rotation_matrix(pose.quaternion_wxyz)

        if self.command_frame == self.tool_frame:
            command = self.keys.command(forward_axis=np.array([0.0, 0.0, 1.0]))
            return CartesianJog(
                self.base_frame,
                base_from_tool @ command.linear_m_s,
                command.angular_rad_s,
            )

        command = self.keys.command()
        linear = _base_translation_from_keys(
            self.keys.pressed,
            base_from_tool[:, 2],
            self.args.linear_speed,
        )
        return CartesianJog(
            self.base_frame,
            linear,
            base_from_tool.T @ command.angular_rad_s,
        )
