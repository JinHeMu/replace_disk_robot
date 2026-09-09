"""MuJoCo implementations of the stable hardware-facing ports."""

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ...core.types import JointState, Wrench


ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
ARM_ACTUATORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)


def _ids(model: mujoco.MjModel, obj_type: mujoco.mjtObj, names: tuple[str, ...]) -> NDArray[np.int_]:
    result = np.array([mujoco.mj_name2id(model, obj_type, name) for name in names], dtype=int)
    if np.any(result < 0):
        missing = [name for name, idx in zip(names, result) if idx < 0]
        raise KeyError(f"MuJoCo model is missing required names: {missing}")
    return result


@dataclass
class MujocoRobotAdapter:
    """Joint-position and gripper adapter; it contains no planner."""

    model: mujoco.MjModel
    data: mujoco.MjData

    def __post_init__(self) -> None:
        joint_ids = _ids(self.model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS)
        self._qpos_ids = self.model.jnt_qposadr[joint_ids]
        self._actuator_ids = _ids(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, ARM_ACTUATORS)
        self._gripper_id = int(_ids(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, ("fingers_actuator",)
        )[0])

    def arm_position(self) -> NDArray[np.float64]:
        return self.data.qpos[self._qpos_ids].copy()

    def read_joint_state(self) -> JointState:
        return JointState(ARM_JOINTS, self.arm_position())

    def command_arm(self, q_target: ArrayLike) -> None:
        target = np.asarray(q_target, dtype=float)
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            raise ValueError("q_target must be a finite 6-vector in radians")
        self.data.ctrl[self._actuator_ids] = target

    def command_joint_positions(self, target: JointState) -> None:
        if set(target.names) != set(ARM_JOINTS) or len(target.names) != len(ARM_JOINTS):
            raise ValueError("joint target must contain each UR5e arm joint exactly once")
        index = {name: i for i, name in enumerate(target.names)}
        ordered = np.array([target.position_rad[index[name]] for name in ARM_JOINTS])
        self.command_arm(ordered)

    def command_gripper(self, command: float) -> None:
        if not np.isfinite(command):
            raise ValueError("gripper command must be finite")
        self.data.ctrl[self._gripper_id] = float(np.clip(command, 0.0, 255.0))

    def command_gripper_opening(self, opening_m: float) -> None:
        """Command physical opening in metres; 0.085 m is fully open."""
        if not np.isfinite(opening_m):
            raise ValueError("gripper opening must be finite")
        clipped = float(np.clip(opening_m, 0.0, 0.085))
        self.command_gripper(255.0 * (1.0 - clipped / 0.085))


@dataclass
class MujocoWristFTAdapter:
    """Read and tare the raw wrist wrench in ``wrist_ft_site`` coordinates."""

    model: mujoco.MjModel
    data: mujoco.MjData

    def __post_init__(self) -> None:
        self._force_id = int(_ids(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, ("wrist_force",)
        )[0])
        self._torque_id = int(_ids(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, ("wrist_torque",)
        )[0])
        self.bias = np.zeros(6)

    def raw(self) -> NDArray[np.float64]:
        force_adr = self.model.sensor_adr[self._force_id]
        torque_adr = self.model.sensor_adr[self._torque_id]
        return np.concatenate((
            self.data.sensordata[force_adr:force_adr + 3],
            self.data.sensordata[torque_adr:torque_adr + 3],
        )).copy()

    def tare(self) -> NDArray[np.float64]:
        self.bias = self.raw()
        return self.bias.copy()

    def wrench(self) -> NDArray[np.float64]:
        return self.raw() - self.bias

    def read_wrench(self) -> Wrench:
        vector = self.wrench()
        return Wrench("wrist_ft_site", vector[:3], vector[3:])
