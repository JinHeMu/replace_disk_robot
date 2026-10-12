"""Keyboard state and JAKA key-to-Cartesian frame mapping."""
from __future__ import annotations
import numpy as np
from replace_disk_robot.control import CartesianJog, KeyControl
from replace_disk_robot.core import JointState, Pose
from replace_disk_robot.core.rotation import rotation_matrix
from ..common.reference import ReferencePipeline as CommonReferencePipeline

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


class ReferencePipeline(CommonReferencePipeline):
    def __init__(self, args, kinematics, arm, servo):
        super().__init__(args, kinematics, arm, servo)
        self.keys = KeyControl(
            args.linear_speed,
            np.deg2rad(args.angular_speed_deg),
            base_frame=self.command_frame,
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
