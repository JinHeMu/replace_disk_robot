"""Dependency-inversion ports implemented by MuJoCo now and ROS 2 later."""

from typing import Protocol, Sequence

from numpy.typing import NDArray
import numpy as np

from .types import JointState, Pose, TrajectoryPoint, Wrench


class ArmPort(Protocol):
    def read_joint_state(self) -> JointState: ...
    def command_joint_positions(self, target: JointState) -> None: ...


class GripperPort(Protocol):
    def command_gripper_opening(self, opening_m: float) -> None: ...


class ForceTorquePort(Protocol):
    def tare(self) -> NDArray[np.float64]: ...
    def read_wrench(self) -> Wrench: ...


class MechanismPort(Protocol):
    def positions(self) -> NDArray[np.float64]: ...
    def command(self, carrier_extraction: float, latch_press: float) -> None: ...


class KinematicsPort(Protocol):

    @property
    def joint_names(self) -> tuple[str, ...]:...

    @property
    def dof(self) -> int:...

    def forward(
        self,joints: JointState,) -> Pose:...

    def jacobian(
        self,joints: JointState,) -> NDArray[np.float64]:...

    def inverse(
        self,target: Pose,seed: JointState,) -> JointState:...

class CollisionCheckerPort(Protocol):

    def is_collision_free(self, joints: JointState) -> bool: ...

    def minimum_distance(self, joints: JointState) -> float: ...


class TrajectoryOptimizerPort(Protocol):
    """Backend-neutral discrete trajectory optimizer.

    Implementations take an initial joint-space trajectory and return an
    optimized trajectory with the same length, timing and joint order.
    """

    def optimize(
        self, initial: Sequence[TrajectoryPoint]
    ) -> Sequence[TrajectoryPoint]: ...


class TrajectoryPlannerPort(Protocol):
    def plan(self, start: JointState, goal: JointState) -> Sequence[TrajectoryPoint]: ...


class ContactControllerPort(Protocol):
    def update(self, nominal: Pose, wrench: Wrench, dt_s: float) -> Pose: ...
