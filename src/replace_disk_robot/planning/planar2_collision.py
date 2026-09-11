"""2-D planar arm collision detection demo on top of core ports.

This is a simple, dependency-free 2D example.  It implements
:class:`CollisionCheckerPort`-compatible methods for a
:class:`~replace_disk_robot.kinematics.planar2.Planar2LinkKinematics` arm.

Obstacles are represented as circles in the XY plane.  The robot is modelled
as two finite line segments with a small radius.  ``minimum_distance`` returns
the signed surface clearance:

* positive -> no collision
* negative -> penetration depth
* zero -> touching
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..core import JointState
from ..kinematics.planar2 import Planar2LinkKinematics


@dataclass
class CircleObstacle:
    """A 2D disk obstacle in the robot's XY plane."""

    x: float
    y: float
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("obstacle radius must be positive")


def _point_segment_distance(
    point: NDArray[np.float64],
    segment_start: NDArray[np.float64],
    segment_end: NDArray[np.float64],
) -> float:
    """2D point-to-line-segment Euclidean distance."""
    ab = segment_end - segment_start
    length_sq = float(ab @ ab)
    if length_sq < 1e-12:
        return float(np.linalg.norm(point - segment_start))
    t = float(np.clip((point - segment_start) @ ab / length_sq, 0.0, 1.0))
    closest = segment_start + t * ab
    return float(np.linalg.norm(point - closest))


class Planar2LinkCollisionChecker:
    """Collision checker for a 2-link planar arm against circular obstacles.

    Examples
    --------
    >>> from replace_disk_robot.kinematics.planar2 import Planar2LinkKinematics
    >>> from replace_disk_robot.planning.planar2_collision import (
    ...     Planar2LinkCollisionChecker, CircleObstacle)
    >>> kinematics = Planar2LinkKinematics(link1_m=1.0, link2_m=0.8)
    >>> checker = Planar2LinkCollisionChecker(
    ...     kinematics,
    ...     obstacles=[CircleObstacle(1.2, 0.0, 0.1)],
    ...     link_radius_m=0.02,
    ... )
    >>> checker.is_collision_free(
    ...     JointState(kinematics.joint_names, [0.0, 0.0])
    ... )
    False
    """

    def __init__(
        self,
        kinematics: Planar2LinkKinematics,
        obstacles: list[CircleObstacle] | None = None,
        *,
        link_radius_m: float = 0.02,
        safety_margin_m: float = 0.0,
    ) -> None:
        if link_radius_m < 0.0:
            raise ValueError("link_radius_m must be non-negative")
        if safety_margin_m < 0.0:
            raise ValueError("safety_margin_m must be non-negative")

        self.kinematics = kinematics
        self.obstacles: list[CircleObstacle] = list(obstacles or [])
        self.link_radius_m = float(link_radius_m)
        self.safety_margin_m = float(safety_margin_m)

    def add_circle_obstacle(self, x: float, y: float, radius: float) -> None:
        """Add a disk obstacle to the current scene."""
        self.obstacles.append(CircleObstacle(x=x, y=y, radius=radius))

    def _check_joints(self, joints: JointState) -> None:
        if joints.names != self.kinematics.joint_names:
            raise ValueError(
                f"expected joint names {self.kinematics.joint_names}, "
                f"got {joints.names}"
            )

    def _link_segments(
        self, joints: JointState
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the two robot link segments as 2D points."""
        self._check_joints(joints)
        q1 = float(joints.position_rad[0])
        q2 = float(joints.position_rad[1])
        l1 = self.kinematics.link1_m
        l2 = self.kinematics.link2_m

        base = np.array([0.0, 0.0], dtype=float)
        elbow = np.array([l1 * np.cos(q1), l1 * np.sin(q1)], dtype=float)
        tip = np.array(
            [
                l1 * np.cos(q1) + l2 * np.cos(q1 + q2),
                l1 * np.sin(q1) + l2 * np.sin(q1 + q2),
            ],
            dtype=float,
        )
        return (base, elbow), (elbow, tip)

    def minimum_distance(self, joints: JointState) -> float:
        """Signed minimum clearance between the two links and all obstacles."""
        if not self.obstacles:
            return float("inf")

        segments = self._link_segments(joints)
        distances: list[float] = []
        for obstacle in self.obstacles:
            center = np.array([obstacle.x, obstacle.y], dtype=float)
            for start, end in segments:
                center_to_segment = _point_segment_distance(center, start, end)
                surface_distance = (
                    center_to_segment
                    - obstacle.radius
                    - self.link_radius_m
                )
                distances.append(surface_distance)

        if not distances:
            return float("inf")
        return min(distances)

    def is_collision_free(self, joints: JointState) -> bool:
        """Return True when the signed clearance is strictly above the margin."""
        return self.minimum_distance(joints) > self.safety_margin_m
