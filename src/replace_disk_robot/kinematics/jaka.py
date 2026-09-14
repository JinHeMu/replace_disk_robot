"""Fixed-base JAKA ZU5 arm kinematics implemented with Pinocchio.

The source URDF describes the complete Tracer + JAKA assembly, but only
``joint_1`` through ``joint_6`` are movable in Pinocchio.  The mobile base and
all fixed accessories therefore contribute no degrees of freedom.  Poses and
Jacobians default to ``jaka_base_link`` so base motion is deliberately outside
this arm-only contract.
"""

from __future__ import annotations

from pathlib import Path

from .ur5e import UR5eKinematics


JAKA_JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7))

PROJECT_JAKA_URDF = (
    Path(__file__).resolve().parents[3]
    / "simulation"
    / "mujoco"
    / "models"
    / "jaka"
    / "urdf"
    / "tracer_jaka_zu5.urdf"
)


class JakaKinematics(UR5eKinematics):
    """FK, geometric Jacobian and DLS IK for the fixed-base JAKA ZU5 arm."""

    robot_name = "JAKA ZU5"

    def __init__(
        self,
        urdf_path: str | Path = PROJECT_JAKA_URDF,
        *,
        end_effector_frame: str = "tool0",
        base_frame: str = "jaka_base_link",
        ik_max_iterations: int = 200,
        ik_position_tolerance_m: float = 1e-5,
        ik_orientation_tolerance_rad: float = 1e-5,
        ik_damping: float = 1e-4,
        ik_max_step_rad: float = 0.2,
    ) -> None:
        super().__init__(
            urdf_path,
            end_effector_frame=end_effector_frame,
            base_frame=base_frame,
            ik_max_iterations=ik_max_iterations,
            ik_position_tolerance_m=ik_position_tolerance_m,
            ik_orientation_tolerance_rad=ik_orientation_tolerance_rad,
            ik_damping=ik_damping,
            ik_max_step_rad=ik_max_step_rad,
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        return JAKA_JOINT_NAMES
