"""MuJoCo implementation of :class:`CollisionCheckerPort`.

The checker treats contacts between the robot arm/gripper and the non-robot
environment as collisions.  Contacts between two environment bodies are ignored
(e.g. the replacement drive resting on the staging bench).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

from ...core import CollisionCheckerPort, JointState


DEFAULT_ROBOT_BODIES: tuple[str, ...] = (
    "ur_base",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "wrist_ft_link",
    "gripper_base_mount",
    "g_base",
    "replacement_drive_held",
    "right_driver",
    "right_coupler",
    "right_spring_link",
    "right_follower",
    "right_pad",
    "right_silicone_pad",
    "left_driver",
    "left_coupler",
    "left_spring_link",
    "left_follower",
    "left_pad",
    "left_silicone_pad",
)

ARM_JOINTS: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@dataclass
class MujocoCollisionChecker(CollisionCheckerPort):
    """Robot-vs-environment collision checker backed by a MuJoCo model."""

    model: mujoco.MjModel
    data: mujoco.MjData
    robot_bodies: tuple[str, ...] = DEFAULT_ROBOT_BODIES
    safety_margin_m: float = 0.0
    dist_max_m: float = 10.0

    def __post_init__(self) -> None:
        if self.safety_margin_m < 0.0:
            raise ValueError("safety_margin_m must be non-negative")
        if self.dist_max_m <= 0.0:
            raise ValueError("dist_max_m must be positive")

        robot_body_id_set: set[int] = set()
        for name in self.robot_bodies:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            if body_id < 0:
                raise KeyError(f"MuJoCo model is missing robot body: {name}")
            robot_body_id_set.add(body_id)

        self._robot_body_set = robot_body_id_set
        self._robot_geom_ids: list[int] = []
        self._environment_geom_ids: list[int] = []

        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            collidable = (
                int(self.model.geom_contype[geom_id]) > 0
                or int(self.model.geom_conaffinity[geom_id]) > 0
            )
            if not collidable:
                continue
            if body_id in robot_body_id_set:
                self._robot_geom_ids.append(geom_id)
            else:
                self._environment_geom_ids.append(geom_id)

        self._joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ARM_JOINTS
            ],
            dtype=int,
        )
        if np.any(self._joint_ids < 0):
            missing = [
                name for name, idx in zip(ARM_JOINTS, self._joint_ids) if idx < 0
            ]
            raise KeyError(f"MuJoCo model is missing arm joints: {missing}")
        self._qpos_ids = self.model.jnt_qposadr[self._joint_ids]

        self._robot_geom_set = set(self._robot_geom_ids)
        self._environment_geom_set = set(self._environment_geom_ids)

    def _apply_joints(self, joints: JointState) -> None:
        if set(joints.names) != set(ARM_JOINTS) or len(joints.names) != len(ARM_JOINTS):
            raise ValueError("joint state must contain each UR5e arm joint exactly once")
        index = {name: i for i, name in enumerate(joints.names)}
        self.data.qpos[self._qpos_ids] = np.array(
            [joints.position_rad[index[name]] for name in ARM_JOINTS]
        )
        mujoco.mj_forward(self.model, self.data)

    def is_collision_free(self, joints: JointState) -> bool:
        """Return True if no robot-environment contact is active."""
        self._apply_joints(joints)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            robot_count = int(geom1 in self._robot_geom_set) + int(
                geom2 in self._robot_geom_set
            )
            env_count = int(geom1 in self._environment_geom_set) + int(
                geom2 in self._environment_geom_set
            )
            if robot_count == 1 and env_count == 1:
                return False
        return True

    def minimum_distance(self, joints: JointState) -> float:
        """Return the smallest signed robot-environment geom distance."""
        self._apply_joints(joints)

        if not self._robot_geom_ids or not self._environment_geom_ids:
            return float("inf")

        min_distance = float("inf")
        for robot_geom in self._robot_geom_ids:
            for env_geom in self._environment_geom_ids:
                distance = float(
                    mujoco.mj_geomDistance(
                        self.model,
                        self.data,
                        robot_geom,
                        env_geom,
                        self.dist_max_m,
                        None,
                    )
                )
                if distance < min_distance:
                    min_distance = distance
        return min_distance
