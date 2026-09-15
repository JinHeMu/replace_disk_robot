"""Deterministic insertion baseline for residual RL.

The baseline owns axial progress and the intended lateral/angular compliance.
It converts a filtered external wrench, already expressed in the task frame, to
a bounded five-dimensional Cartesian velocity command.  It never writes joint
commands, never accesses MuJoCo, and never bypasses the final safety guard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from replace_disk_robot.core.types import Pose, Wrench

from .config import InsertionConfig


@dataclass(frozen=True)
class InsertionCommand:
    """Task-frame command and state emitted by the classical baseline."""

    command: NDArray[np.float64]  # [vx, vy, vz, wy, wz] in socket_entry
    status: str
    axial_resistance_n: float
    lateral_force_n: NDArray[np.float64]
    lateral_moment_nm: NDArray[np.float64]

    def __post_init__(self) -> None:
        command = np.asarray(self.command, dtype=float)
        if command.shape != (5,) or not np.all(np.isfinite(command)):
            raise ValueError("command must be a finite vector with shape (5,)")
        object.__setattr__(self, "command", command.copy())
        object.__setattr__(
            self,
            "lateral_force_n",
            np.asarray(self.lateral_force_n, dtype=float).copy(),
        )
        object.__setattr__(
            self,
            "lateral_moment_nm",
            np.asarray(self.lateral_moment_nm, dtype=float).copy(),
        )


class InsertionController:
    """Axial force governor plus lateral/angular force-velocity compliance.

    Conventions
    -----------
    ``external_wrench`` is the load acting on the tool from the environment,
    expressed in the task frame.  During insertion, axial wall friction on the
    disk is therefore approximately negative along task X.  The lateral terms
    command motion in the direction of the lateral external load, which is the
    direction that relieves the contact.
    """

    def __init__(self, config: InsertionConfig | None = None) -> None:
        self.config = config or InsertionConfig()
        self._last_command = np.zeros(5, dtype=float)
        self._status = "idle"

    @property
    def last_command(self) -> NDArray[np.float64]:
        return self._last_command.copy()

    @property
    def status(self) -> str:
        return self._status

    def reset(self, measured_pose: Pose | None = None) -> None:
        if measured_pose is not None and not isinstance(measured_pose, Pose):
            raise TypeError("measured_pose must be a Pose")
        self._last_command = np.zeros(5, dtype=float)
        self._status = "idle"

    def update(
        self,
        measured_pose: Pose,
        external_wrench: Wrench,
        dt_s: float,
    ) -> InsertionCommand:
        if not isinstance(measured_pose, Pose):
            raise TypeError("measured_pose must be a Pose")
        if not isinstance(external_wrench, Wrench):
            raise TypeError("external_wrench must be a Wrench")
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be finite and positive")

        cfg = self.config
        force = np.asarray(external_wrench.force_n, dtype=float)
        torque = np.asarray(external_wrench.torque_nm, dtype=float)
        if force.shape != (3,) or torque.shape != (3,):
            raise ValueError("external wrench must contain 3+3 finite values")

        # Axial resistance is positive while the environment opposes +X motion.
        axial_resistance = max(0.0, float(-force[0]))
        axial_effort = cfg.forward_speed_m_s + cfg.axial_force_gain_m_s_n * (
            cfg.target_axial_force_n - axial_resistance
        )
        vx = float(
            np.clip(axial_effort, -cfg.max_retract_speed_m_s, cfg.max_forward_speed_m_s)
        )

        lateral_force = force[1:3].copy()
        lateral_moment = torque[1:3].copy()
        vy, vz = np.clip(
            cfg.lateral_force_gain_m_s_n * lateral_force,
            -cfg.lateral_speed_limit_m_s,
            cfg.lateral_speed_limit_m_s,
        )
        wy, wz = np.clip(
            cfg.angular_torque_gain_rad_s_nm * lateral_moment,
            -cfg.angular_speed_limit_rad_s,
            cfg.angular_speed_limit_rad_s,
        )

        if axial_resistance > cfg.axial_soft_limit_n:
            status = "high_axial_load"
            vx = min(vx, 0.0)
        elif axial_resistance > cfg.target_axial_force_n:
            status = "load_governed"
        elif abs(force[0]) < 1e-9:
            status = "free_advance"
        else:
            status = "advancing"

        command = np.array([vx, vy, vz, wy, wz], dtype=float)
        self._last_command = command
        self._status = status
        return InsertionCommand(
            command=command,
            status=status,
            axial_resistance_n=float(axial_resistance),
            lateral_force_n=lateral_force,
            lateral_moment_nm=lateral_moment,
        )


__all__ = ["InsertionCommand", "InsertionController"]
