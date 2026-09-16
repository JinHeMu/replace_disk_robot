import numpy as np
import pytest

from replace_disk_robot.core import TrajectoryPoint
from replace_disk_robot.kinematics.planar2 import Planar2LinkKinematics
from replace_disk_robot.planning.optimize import TrajectoryOptimizer


def _zigzag_trajectory():
    q = np.array([
        [0.0, 0.0],
        [0.3, -0.2],
        [0.7, 0.3],
        [1.0, -0.1],
        [1.4, 0.2],
        [2.0, 0.0],
    ])
    return [
        TrajectoryPoint(i * 0.1, q[i])
        for i in range(len(q))
    ]


def test_optimizer_reduces_smoothness_cost_and_keeps_endpoints():
    kinematics = Planar2LinkKinematics(link1_m=1.0, link2_m=1.0)
    optimizer = TrajectoryOptimizer(
        kinematics,
        velocity_weight=10.0,
        acceleration_weight=5.0,
        task_weight=1.0,
        method="gradient",
        max_iterations_gradient=30,
    )
    initial = _zigzag_trajectory()
    optimized = optimizer.optimize(initial)

    assert len(optimized) == len(initial)
    np.testing.assert_allclose(optimized[0].position_rad, initial[0].position_rad)
    np.testing.assert_allclose(optimized[-1].position_rad, initial[-1].position_rad)

    before = optimizer.evaluate(initial)
    after = optimizer.evaluate(optimized)
    assert after["total"] < before["total"]
