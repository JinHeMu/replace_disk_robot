from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core import CollisionCheckerPort, JointState


def _ordered_positions(start: JointState, goal: JointState) -> NDArray[np.float64]:
    """Return both joint vectors in ``start.names`` order after validating names."""
    if start.names != goal.names:
        if set(start.names) != set(goal.names) or len(start.names) != len(goal.names):
            raise ValueError("start and goal must describe the same joint set")
        index = {name: i for i, name in enumerate(goal.names)}
        goal_ordered = np.array([goal.position_rad[index[name]] for name in start.names])
    else:
        goal_ordered = goal.position_rad
    return np.asarray(start.position_rad, dtype=float), np.asarray(goal_ordered, dtype=float)


def interpolate(start: JointState, goal: JointState, alpha: float) -> JointState:
    """Linear joint-space interpolation between two named joint states."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    start_q, goal_q = _ordered_positions(start, goal)
    return JointState(start.names, (1.0 - alpha) * start_q + alpha * goal_q)


def is_path_collision_free(
    checker: CollisionCheckerPort,
    start: JointState,
    goal: JointState,
    samples: int = 32,
) -> bool:
    """Return True if every sampled state along the linear path is collision free.

    ``samples`` is the number of interior samples; endpoints are always checked.
    """
    if samples < 1:
        raise ValueError("samples must be at least 1")
    alpha_values = np.linspace(0.0, 1.0, samples + 2)
    for alpha in alpha_values:
        if not checker.is_collision_free(interpolate(start, goal, float(alpha))):
            return False
    return True


def minimum_clearance_along_path(
    checker: CollisionCheckerPort,
    start: JointState,
    goal: JointState,
    samples: int = 32,
) -> float:
    """Return the minimum clearance along a collision-free linear path."""
    if samples < 1:
        raise ValueError("samples must be at least 1")
    alpha_values = np.linspace(0.0, 1.0, samples + 2)
    return min(
        checker.minimum_distance(interpolate(start, goal, float(alpha)))
        for alpha in alpha_values
    )


def clearance_cost(checker: CollisionCheckerPort, joints: JointState, eps: float = 1e-3) -> float:
    """Small distance -> large cost; useful for trajectory optimization."""
    distance = checker.minimum_distance(joints)
    return 1.0 / max(float(distance), eps)
