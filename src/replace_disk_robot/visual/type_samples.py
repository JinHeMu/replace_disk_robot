"""Convert core data structures into unit-aware plotting samples."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core import CartesianJog, JointState, Pose, TrajectoryPoint, Wrench


@dataclass(frozen=True)
class PlotGroup:
    """Signals that share one axis and one physical unit."""

    title: str
    ylabel: str
    labels: tuple[str, ...]
    values: NDArray[np.float64]
    minimum_half_range: float


@dataclass(frozen=True)
class PlotSample:
    """A title plus one or more unit-compatible signal groups."""

    title: str
    groups: tuple[PlotGroup, ...]


def _group(
    title: str,
    ylabel: str,
    labels: tuple[str, ...],
    values,
    minimum_half_range: float,
) -> PlotGroup:
    array = np.asarray(values, dtype=float)
    if array.shape != (len(labels),) or not np.all(np.isfinite(array)):
        raise ValueError("plot values must be a finite vector matching the labels")
    if not np.isfinite(minimum_half_range) or minimum_half_range <= 0:
        raise ValueError("minimum_half_range must be finite and positive")
    return PlotGroup(title, ylabel, labels, array.copy(), float(minimum_half_range))


def to_plot_sample(value) -> PlotSample:
    """Adapt any supported core type to a plot description.

    This function contains no Matplotlib dependency and is suitable for tests,
    logging frontends, or a future ROS/web visualizer.
    """
    xyz = ("x", "y", "z")
    if isinstance(value, Wrench):
        return PlotSample(
            f"Wrench [{value.frame_id}]",
            (
                _group("Force", "force [N]", ("Fx", "Fy", "Fz"), value.force_n, 1.0),
                _group(
                    "Torque", "torque [N·m]", ("Tx", "Ty", "Tz"),
                    value.torque_nm, 0.1,
                ),
            ),
        )
    if isinstance(value, Pose):
        return PlotSample(
            f"Pose [{value.frame_id}]",
            (
                _group("Position", "position [m]", xyz, value.position_m, 0.01),
                _group(
                    "Quaternion", "quaternion", ("qw", "qx", "qy", "qz"),
                    value.quaternion_wxyz, 1.0,
                ),
            ),
        )
    if isinstance(value, JointState):
        return PlotSample(
            "JointState",
            (_group(
                "Joint position", "position [rad]", tuple(value.names),
                value.position_rad, 0.1,
            ),),
        )
    if isinstance(value, TrajectoryPoint):
        labels = tuple(f"q{i}" for i in range(value.position_rad.size))
        return PlotSample(
            "TrajectoryPoint",
            (_group("Joint trajectory", "position [rad]", labels, value.position_rad, 0.1),),
        )
    if isinstance(value, CartesianJog):
        return PlotSample(
            f"CartesianJog [{value.base_frame}]",
            (
                _group(
                    "Linear velocity", "linear [m/s]", ("vx", "vy", "vz"),
                    value.linear_m_s, 0.01,
                ),
                _group(
                    "TCP angular velocity", "angular [rad/s]",
                    ("wx", "wy", "wz"), value.angular_rad_s, 0.1,
                ),
            ),
        )
    raise TypeError(f"unsupported visual type: {type(value).__name__}")
