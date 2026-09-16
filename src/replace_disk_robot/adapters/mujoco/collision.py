"""MuJoCo implementation of :class:`CollisionCheckerPort`.

The checker treats contacts between the robot arm/gripper and the non-robot
environment as collisions.  Contacts between two environment bodies are ignored
The rigidly held drive belongs to the robot collision set.
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
    arm_joints: tuple[str, ...] = ARM_JOINTS
    use_separate_planning_data: bool = True

    def __post_init__(self) -> None:
        if self.safety_margin_m < 0.0:
            raise ValueError("safety_margin_m must be non-negative")
        if self.dist_max_m <= 0.0:
            raise ValueError("dist_max_m must be positive")
        self.arm_joints = tuple(self.arm_joints)
        if not self.arm_joints or len(set(self.arm_joints)) != len(self.arm_joints):
            raise ValueError("arm_joints must contain unique joint names")

        if self.use_separate_planning_data and hasattr(mujoco, "mj_copyData"):
            self._planning_data = mujoco.MjData(self.model)
        else:
            self._planning_data = self.data

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
                for name in self.arm_joints
            ],
            dtype=int,
        )
        if np.any(self._joint_ids < 0):
            missing = [
                name for name, idx in zip(self.arm_joints, self._joint_ids) if idx < 0
            ]
            raise KeyError(f"MuJoCo model is missing arm joints: {missing}")
        self._qpos_ids = self.model.jnt_qposadr[self._joint_ids]

        self._robot_geom_set = set(self._robot_geom_ids)
        self._environment_geom_set = set(self._environment_geom_ids)

    def _query_data(self) -> mujoco.MjData:
        if self._planning_data is self.data:
            return self.data
        mujoco.mj_copyData(self._planning_data, self.model, self.data)
        return self._planning_data

    def _apply_joints(self, joints: JointState) -> mujoco.MjData:
        if (
            set(joints.names) != set(self.arm_joints)
            or len(joints.names) != len(self.arm_joints)
        ):
            raise ValueError(
                "joint state must contain each configured arm joint exactly once"
            )
        data = self._query_data()
        index = {name: i for i, name in enumerate(joints.names)}
        data.qpos[self._qpos_ids] = np.array(
            [joints.position_rad[index[name]] for name in self.arm_joints]
        )
        mujoco.mj_forward(self.model, data)
        return data

    def is_collision_free(self, joints: JointState) -> bool:
        """Return True if no contact or safety-margin violation is active."""
        data = self._apply_joints(joints)
        for i in range(data.ncon):
            contact = data.contact[i]
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
        if self.safety_margin_m <= 0.0:
            return True
        return self._minimum_distance_loaded(data) > self.safety_margin_m

    def minimum_distance(self, joints: JointState) -> float:
        """Return the smallest signed robot-environment geom distance."""
        data = self._apply_joints(joints)
        return self._minimum_distance_loaded(data)

    def _minimum_distance_loaded(self, data: mujoco.MjData) -> float:
        if not self._robot_geom_ids or not self._environment_geom_ids:
            return float("inf")

        min_distance = float("inf")
        for robot_geom in self._robot_geom_ids:
            for env_geom in self._environment_geom_ids:
                distance = float(
                    mujoco.mj_geomDistance(
                        self.model,
                        data,
                        robot_geom,
                        env_geom,
                        self.dist_max_m,
                        None,
                    )
                )
                if distance < min_distance:
                    min_distance = distance
        return min_distance
