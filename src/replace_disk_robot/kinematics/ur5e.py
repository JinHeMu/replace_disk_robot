"""Fixed-base UR5e kinematics implemented with Pinocchio.

The public methods deliberately mirror :class:`Planar2LinkKinematics`: named
joint states go in, while ``Pose`` or a 6-by-6 geometric Jacobian comes out.
Pinocchio is imported lazily so the rest of the project remains usable without
the optional kinematics dependency.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from replace_disk_robot.core.types import JointState, Pose


UR5E_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

PROJECT_UR5E_URDF = (
    Path(__file__).resolve().parents[3]
    / "simulation"
    / "mujoco"
    / "models"
    / "ur5e"
    / "ur5e_from_mujoco.urdf"
)


def _load_pinocchio() -> Any:
    """Load Pinocchio with a useful error for the known ROS/NumPy ABI clash."""
    spec = find_spec("pinocchio")
    if spec is None:
        raise ImportError(
            "Pinocchio is required for arm kinematics; install the optional "
            "dependency with `pip install -e '.[kinematics]'`"
        )

    # Ubuntu 22.04 / ROS Humble's binary is built against NumPy 1.x. Importing
    # it with a user-installed NumPy 2.x can terminate Python before an
    # exception can be caught, so fail safely and explain the environment fix.
    origin = str(spec.origin or "")
    if int(np.__version__.split(".", maxsplit=1)[0]) >= 2 and origin.startswith(
        "/opt/ros/humble/"
    ):
        raise RuntimeError(
            "the ROS Humble Pinocchio binary is incompatible with the active "
            f"NumPy {np.__version__}; use an isolated environment with a "
            "compatible Pinocchio build, or run the ROS package with NumPy 1.x"
        )

    import pinocchio as pin

    return pin


class UR5eKinematics:
    """Pinocchio FK, geometric Jacobian and damped-least-squares IK for UR5e.

    The model is always reduced to the six named UR5e arm joints. Any gripper
    or accessory joints present in the URDF are locked at their neutral values.
    ``base_frame`` must be fixed with respect to Pinocchio's universe frame.
    """

    robot_name = "UR5e"

    def __init__(
        self,
        urdf_path: str | Path = PROJECT_UR5E_URDF,
        *,
        end_effector_frame: str = "wrist_3_link",
        base_frame: str = "world",
        ik_max_iterations: int = 200,
        ik_position_tolerance_m: float = 1e-5,
        ik_orientation_tolerance_rad: float = 1e-5,
        ik_damping: float = 1e-4,
        ik_max_step_rad: float = 0.2,
    ) -> None:
        path = Path(urdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{self.robot_name} URDF does not exist: {path}")
        if not end_effector_frame or not base_frame:
            raise ValueError("end_effector_frame and base_frame must not be empty")
        if ik_max_iterations <= 0:
            raise ValueError("ik_max_iterations must be positive")
        if ik_position_tolerance_m <= 0 or ik_orientation_tolerance_rad <= 0:
            raise ValueError("IK tolerances must be positive")
        if ik_damping <= 0 or ik_max_step_rad <= 0:
            raise ValueError("ik_damping and ik_max_step_rad must be positive")

        self._pin = _load_pinocchio()
        full_model = self._pin.buildModelFromUrdf(str(path))
        self._validate_arm_joints(full_model)

        arm_names = set(self.joint_names)
        locked_joint_ids = [
            joint_id
            for joint_id in range(1, full_model.njoints)
            if full_model.names[joint_id] not in arm_names
        ]
        self._model = self._pin.buildReducedModel(
            full_model,
            locked_joint_ids,
            self._pin.neutral(full_model),
        )
        self._data = self._model.createData()

        self._end_effector_frame = self._frame_id(end_effector_frame)
        self._base_frame = self._frame_id(base_frame)
        if self._model.frames[self._base_frame].parentJoint != 0:
            raise ValueError(
                f"base frame {base_frame!r} is not fixed with respect to the model root"
            )

        self.urdf_path = path
        self.end_effector_frame = end_effector_frame
        self.base_frame = base_frame
        self.ik_max_iterations = int(ik_max_iterations)
        self.ik_position_tolerance_m = float(ik_position_tolerance_m)
        self.ik_orientation_tolerance_rad = float(ik_orientation_tolerance_rad)
        self.ik_damping = float(ik_damping)
        self.ik_max_step_rad = float(ik_max_step_rad)

        self._q_indices = np.array(
            [
                self._model.joints[self._model.getJointId(name)].idx_q
                for name in self.joint_names
            ],
            dtype=int,
        )
        self._v_indices = np.array(
            [
                self._model.joints[self._model.getJointId(name)].idx_v
                for name in self.joint_names
            ],
            dtype=int,
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        return UR5E_JOINT_NAMES

    @property
    def dof(self) -> int:
        return 6

    @property
    def joint_limits_rad(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return lower and upper limits ordered exactly like ``joint_names``."""
        lower = np.asarray(self._model.lowerPositionLimit)[self._q_indices].copy()
        upper = np.asarray(self._model.upperPositionLimit)[self._q_indices].copy()
        return lower, upper

    def forward(self, joints: JointState) -> Pose:
        """Return the end-effector pose expressed in ``base_frame``."""
        q = self._configuration(joints)
        base_to_end = self._base_to_end(q)
        quaternion = self._pin.Quaternion(base_to_end.rotation)
        return Pose(
            frame_id=self.base_frame,
            position_m=np.asarray(base_to_end.translation).copy(),
            quaternion_wxyz=np.array(
                [quaternion.w, quaternion.x, quaternion.y, quaternion.z],
                dtype=float,
            ),
        )

    def jacobian(self, joints: JointState) -> NDArray[np.float64]:
        """Return ``[linear; angular]`` geometric Jacobian in ``base_frame``."""
        q = self._configuration(joints)
        self._update_placements(q)
        jacobian_world = self._pin.computeFrameJacobian(
            self._model,
            self._data,
            q,
            self._end_effector_frame,
            self._pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        world_rotation_base = self._data.oMf[self._base_frame].rotation
        base_rotation_world = world_rotation_base.T
        rotate_twist = np.zeros((6, 6), dtype=float)
        rotate_twist[:3, :3] = base_rotation_world
        rotate_twist[3:, 3:] = base_rotation_world
        return np.asarray(rotate_twist @ jacobian_world[:, self._v_indices]).copy()

    def inverse(self, target: Pose, seed: JointState) -> JointState:
        """Solve full-pose IK from ``seed`` using damped least squares.

        A six-axis arm can have multiple IK branches; the seed selects the local branch. A
        ``RuntimeError`` is raised if the requested tolerances are not reached.
        """
        if target.frame_id != self.base_frame:
            raise ValueError(
                f"expected target in frame {self.base_frame!r}, got {target.frame_id!r}"
            )
        q = self._configuration(seed)
        target_rotation = self._pin.Quaternion(*target.quaternion_wxyz).matrix()

        self._update_placements(q)
        world_to_base = self._data.oMf[self._base_frame].copy()
        base_to_target = self._pin.SE3(target_rotation, target.position_m)
        world_to_target = world_to_base * base_to_target

        position_error = np.inf
        orientation_error = np.inf
        for _ in range(self.ik_max_iterations):
            self._update_placements(q)
            world_to_end = self._data.oMf[self._end_effector_frame]
            end_to_target = world_to_end.actInv(world_to_target)

            base_to_end = world_to_base.actInv(world_to_end)
            position_error = float(
                np.linalg.norm(target.position_m - base_to_end.translation)
            )
            orientation_error = float(
                np.linalg.norm(
                    self._pin.log3(base_to_end.rotation.T @ target_rotation)
                )
            )
            if (
                position_error <= self.ik_position_tolerance_m
                and orientation_error <= self.ik_orientation_tolerance_rad
            ):
                return JointState(self.joint_names, q[self._q_indices])

            error = self._pin.log6(end_to_target).vector
            frame_jacobian = self._pin.computeFrameJacobian(
                self._model,
                self._data,
                q,
                self._end_effector_frame,
                self._pin.ReferenceFrame.LOCAL,
            )
            error_jacobian = (
                -self._pin.Jlog6(end_to_target.inverse()) @ frame_jacobian
            )
            normal = error_jacobian @ error_jacobian.T
            normal += (self.ik_damping**2) * np.eye(6)
            velocity = -error_jacobian.T @ np.linalg.solve(normal, error)

            step_norm = float(np.linalg.norm(velocity[self._v_indices]))
            if step_norm > self.ik_max_step_rad:
                velocity *= self.ik_max_step_rad / step_norm
            q = self._pin.integrate(self._model, q, velocity)
            q = np.clip(
                q,
                self._model.lowerPositionLimit,
                self._model.upperPositionLimit,
            )

        raise RuntimeError(
            f"{self.robot_name} IK did not converge after "
            f"{self.ik_max_iterations} iterations "
            f"(position error={position_error:.3e} m, "
            f"orientation error={orientation_error:.3e} rad)"
        )

    def _validate_arm_joints(self, model: Any) -> None:
        missing = [name for name in self.joint_names if not model.existJointName(name)]
        if missing:
            raise ValueError(
                f"URDF is missing required {self.robot_name} joints: {missing}"
            )
        invalid = [
            name
            for name in self.joint_names
            if model.joints[model.getJointId(name)].nq != 1
            or model.joints[model.getJointId(name)].nv != 1
        ]
        if invalid:
            raise ValueError(
                f"{self.robot_name} joints must each have one position and velocity: "
                f"{invalid}"
            )

    def _frame_id(self, name: str) -> int:
        if not self._model.existFrame(name):
            raise ValueError(f"URDF does not contain frame {name!r}")
        return int(self._model.getFrameId(name))

    def _configuration(self, joints: JointState) -> NDArray[np.float64]:
        if joints.names != self.joint_names:
            raise ValueError(
                f"expected joint names {self.joint_names}, got {joints.names}"
            )
        q = np.asarray(self._pin.neutral(self._model)).copy()
        q[self._q_indices] = joints.position_rad
        return q

    def _update_placements(self, q: NDArray[np.float64]) -> None:
        self._pin.forwardKinematics(self._model, self._data, q)
        self._pin.updateFramePlacements(self._model, self._data)

    def _base_to_end(self, q: NDArray[np.float64]) -> Any:
        self._update_placements(q)
        return self._data.oMf[self._base_frame].actInv(
            self._data.oMf[self._end_effector_frame]
        )
