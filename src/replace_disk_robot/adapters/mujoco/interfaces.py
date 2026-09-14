"""MuJoCo implementations of the stable hardware-facing ports."""

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ...core.types import JointState, Wrench
from ...core.ports import ArmPort, ForceTorquePort


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

# return id
def _ids(model: mujoco.MjModel, obj_type: mujoco.mjtObj, names: tuple[str, ...]) -> NDArray[np.int_]:
    result = np.array([mujoco.mj_name2id(model, obj_type, name) for name in names], dtype=int)
    if np.any(result < 0):
        missing = [name for name, idx in zip(names, result) if idx < 0]
        raise KeyError(f"MuJoCo model is missing required names: {missing}")
    return result


@dataclass
class MujocoRobotAdapter(ArmPort):
    """Named joint-position adapter with an optional gripper actuator."""

    model: mujoco.MjModel
    data: mujoco.MjData
    compensate_bias: bool = False
    joint_names: tuple[str, ...] = ARM_JOINTS
    actuator_names: tuple[str, ...] = ARM_ACTUATORS
    gripper_actuator_name: str | None = "fingers_actuator"

    def __post_init__(self) -> None:
        self.joint_names = tuple(self.joint_names)
        self.actuator_names = tuple(self.actuator_names)
        if not self.joint_names or len(self.joint_names) != len(self.actuator_names):
            raise ValueError("joint_names and actuator_names must have equal non-zero length")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be unique")
        if len(set(self.actuator_names)) != len(self.actuator_names):
            raise ValueError("actuator_names must be unique")
        joint_ids = _ids(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_names)
        self._qpos_ids = self.model.jnt_qposadr[joint_ids]
        self._dof_ids = self.model.jnt_dofadr[joint_ids]
        self._actuator_ids = _ids(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.actuator_names
        )
        self._gripper_id = None
        if self.gripper_actuator_name is not None:
            self._gripper_id = int(_ids(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                (self.gripper_actuator_name,),
            )[0])

    def arm_position(self) -> NDArray[np.float64]:
        return self.data.qpos[self._qpos_ids].copy()

    def read_joint_state(self) -> JointState:
        return JointState(self.joint_names, self.arm_position())

    def command_arm(self, q_target: ArrayLike) -> None:
        target = np.asarray(q_target, dtype=float)
        expected_shape = (len(self.joint_names),)
        if target.shape != expected_shape or not np.all(np.isfinite(target)):
            raise ValueError(
                f"q_target must be a finite vector with shape {expected_shape} in radians"
            )
        control = target.copy()
        if self.compensate_bias:
            # Explicit opt-in equilibrium feedforward for the position servos.
            gains = self.model.actuator_gainprm[self._actuator_ids, 0]
            if np.any(gains <= 0):
                raise ValueError('Positive position-servo gains required for bias compensation')
            control += self.data.qfrc_bias[self._dof_ids] / gains
        self.data.ctrl[self._actuator_ids] = control

    def command_joint_positions(self, target: JointState) -> None:
        if (
            set(target.names) != set(self.joint_names)
            or len(target.names) != len(self.joint_names)
        ):
            raise ValueError("joint target must contain each configured arm joint exactly once")
        index = {name: i for i, name in enumerate(target.names)}
        ordered = np.array([
            target.position_rad[index[name]] for name in self.joint_names
        ])
        self.command_arm(ordered)

    def command_gripper(self, command: float) -> None:
        if self._gripper_id is None:
            raise RuntimeError("this robot adapter has no gripper actuator")
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
class MujocoWristFTAdapter(ForceTorquePort):
    """Read and tare a named MuJoCo force/torque sensor pair."""

    model: mujoco.MjModel
    data: mujoco.MjData
    force_sensor_name: str = "wrist_force"
    torque_sensor_name: str = "wrist_torque"
    frame_id: str = "wrist_ft_site"

    def __post_init__(self) -> None:
        if not self.force_sensor_name or not self.torque_sensor_name or not self.frame_id:
            raise ValueError("force sensor, torque sensor and frame names must not be empty")
        self._force_id = int(_ids(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, (self.force_sensor_name,)
        )[0])
        self._torque_id = int(_ids(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, (self.torque_sensor_name,)
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
        return Wrench(self.frame_id, vector[:3], vector[3:])
