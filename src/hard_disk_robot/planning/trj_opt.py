"""Trajectory optimization for backend-neutral planning.

The optimizer works on a sequence of :class:`TrajectoryPoint`.  The cost can
weight:

* task tracking (end-effector reference path),
* joint velocity,
* joint acceleration,
* optionally collision penalty via :class:`CollisionCheckerPort`.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from ..core import CollisionCheckerPort, JointState, TrajectoryPoint
from ..kinematics.planar2 import Planar2LinkKinematics


def _import_scipy_minimize() -> Callable | None:
    """Best-effort SciPy import; fallback to an internal gradient method."""
    try:
        from scipy.optimize import minimize  # type: ignore
        return minimize
    except Exception:
        return None


class TrajectoryOptimizer:
    """Optimize a discrete joint trajectory with smoothness and task costs.

    Parameters
    ----------
    kinematics:
        Used for task tracking in Cartesian workspace.  If ``None``, task
        tracking is ignored.
    collision_checker:
        Optional collision checker; collision penalty is added when provided.
    target_tip_trajectory:
        Optional ``(N, 2)`` desired planar end-effector positions.  If not
        given, a straight line from the first tip to the last tip is used.
    task_weight:
        Weight of task tracking error.
    velocity_weight:
        Weight of squared joint velocity.
    acceleration_weight:
        Weight of squared joint acceleration.
    dt_s:
        Sample period used for velocity/acceleration finite differences.
    optimize_endpoint:
        If True, the final trajectory point is free to move (task can pull it).
        The first point is always fixed.
    method:
        ``"auto"`` tries SciPy L-BFGS-B when available, otherwise internal
        gradient descent.  ``"gradient"`` always uses the internal method.
        ``"lbfgs"``/``"newton"`` use SciPy if available.
    """

    def __init__(
        self,
        kinematics: Planar2LinkKinematics | None = None,
        *,
        collision_checker: CollisionCheckerPort | None = None,
        target_tip_trajectory: NDArray[np.float64] | None = None,
        task_weight: float = 1.0,
        velocity_weight: float = 0.1,
        acceleration_weight: float = 0.01,
        dt_s: float = 0.1,
        optimize_endpoint: bool = False,
        method: str = "auto",
        max_iterations_gradient: int = 200,
        gradient_step_start: float = 0.1,
    ) -> None:
        if task_weight < 0.0 or velocity_weight < 0.0 or acceleration_weight < 0.0:
            raise ValueError("cost weights must be non-negative")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if max_iterations_gradient <= 0:
            raise ValueError("max_iterations_gradient must be positive")
        if method not in {"auto", "gradient", "lbfgs", "newton"}:
            raise ValueError(f"unknown optimization method: {method}")

        self.kinematics = kinematics
        self.collision_checker = collision_checker
        self.task_weight = float(task_weight)
        self.velocity_weight = float(velocity_weight)
        self.acceleration_weight = float(acceleration_weight)
        self.dt_s = float(dt_s)
        self.optimize_endpoint = bool(optimize_endpoint)
        self.method = method
        self.max_iterations_gradient = int(max_iterations_gradient)
        self.gradient_step_start = float(gradient_step_start)
        self._scipy_minimize = _import_scipy_minimize()
        self._target_tip_trajectory = target_tip_trajectory

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def optimize(
        self,
        initial: Sequence[TrajectoryPoint],
    ) -> Sequence[TrajectoryPoint]:
        """Return an optimized :class:`TrajectoryPoint` sequence."""
        if len(initial) < 2:
            raise ValueError("need at least two trajectory points to optimize")
        q = np.array([p.position_rad for p in initial], dtype=float)
        if q.ndim != 2:
            raise ValueError("trajectory points must all be 2D arrays")

        names = self._names(initial, q.shape[1])
        n_joints = q.shape[1]
        free_start = 1
        free_end = q.shape[0] if self.optimize_endpoint else q.shape[0] - 1
        x = q[free_start:free_end].reshape(-1).copy()

        method = self.method
        if method == "auto":
            method = "lbfgs" if self._scipy_minimize is not None else "gradient"

        if method in {"lbfgs", "newton"}:
            if self._scipy_minimize is None:
                raise RuntimeError(
                    f"method={method} requires SciPy; unavailable in this environment"
                )
            x = self._optimize_scipy(x, q, free_start, free_end, method)
        else:
            x = self._optimize_gradient(x, q, free_start, free_end)

        q_opt = q.copy()
        q_opt[free_start:free_end] = x.reshape(-1, n_joints)

        # Enforce optional bounds by clipping if user provided a checker/bounds
        # (currently no explicit bounds API; clip to original convex hull is not
        # desired, so no clipping here).
        return [
            TrajectoryPoint(
                time_from_start_s=float(p.time_from_start_s),
                position_rad=np.asarray(q_opt[i], dtype=float).copy(),
            )
            for i, p in enumerate(initial)
        ]

    def evaluate(self, trajectory: Sequence[TrajectoryPoint]) -> dict[str, float]:
        """Return individual cost terms for a trajectory (useful for comparison)."""
        q = np.array([p.position_rad for p in trajectory], dtype=float)
        target = self._target_path(q)
        return {
            "task": self._task_cost(q, target),
            "velocity": self._velocity_cost(q),
            "acceleration": self._acceleration_cost(q),
            "collision": self._collision_cost(q),
            "total": (
                self._task_cost(q, target)
                + self._velocity_cost(q)
                + self._acceleration_cost(q)
                + self._collision_cost(q)
            ),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _names(
        self,
        initial: Sequence[TrajectoryPoint],
        n_joints: int,
    ) -> tuple[str, ...]:
        # Prefer the kinematics object's named joint order when available.
        if self.kinematics is not None and hasattr(self.kinematics, "joint_names"):
            names = tuple(self.kinematics.joint_names)
            if len(names) == n_joints:
                return names
        return tuple(f"q{i}" for i in range(n_joints))

    def _target_path(
        self,
        q: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self._target_tip_trajectory is not None:
            target = np.asarray(self._target_tip_trajectory, dtype=float)
            if target.shape == (q.shape[0], 2):
                return target
            if target.shape == (2,):
                return np.tile(target, (q.shape[0], 1))
            raise ValueError(
                "target_tip_trajectory must have shape (N, 2) or (2,)"
            )

        # Straight line in Cartesian workspace between first and last tip.
        names = self._names(q, q.shape[1])
        first = self.kinematics.forward(JointState(names, q[0]))
        last = self.kinematics.forward(JointState(names, q[-1]))
        p0 = np.asarray(first.position_m[:2], dtype=float)
        p1 = np.asarray(last.position_m[:2], dtype=float)
        return np.outer(np.linspace(0.0, 1.0, q.shape[0]), p1 - p0) + p0

    def _decode(
        self,
        x: NDArray[np.float64],
        q: NDArray[np.float64],
        free_start: int,
        free_end: int,
    ) -> NDArray[np.float64]:
        q_work = q.copy()
        q_work[free_start:free_end] = x.reshape(-1, q.shape[1])
        return q_work

    def _task_cost(
        self,
        q_work: NDArray[np.float64],
        target: NDArray[np.float64],
    ) -> float:
        if self.kinematics is None or self.task_weight == 0.0:
            return 0.0
        names = self._names(q_work, q_work.shape[1])
        error = 0.0
        for i in range(q_work.shape[0]):
            pose = self.kinematics.forward(JointState(names, q_work[i]))
            error += float(np.sum((pose.position_m[:2] - target[i]) ** 2))
        return self.task_weight * error

    def _velocity_cost(self, q_work: NDArray[np.float64]) -> float:
        if self.velocity_weight == 0.0:
            return 0.0
        diff = np.diff(q_work, axis=0) / self.dt_s
        return float(self.velocity_weight * np.sum(diff * diff))

    def _acceleration_cost(self, q_work: NDArray[np.float64]) -> float:
        if self.acceleration_weight == 0.0:
            return 0.0
        acc = np.diff(q_work, n=2, axis=0) / (self.dt_s * self.dt_s)
        return float(self.acceleration_weight * np.sum(acc * acc))

    def _collision_cost(self, q_work: NDArray[np.float64]) -> float:
        if self.collision_checker is None:
            return 0.0
        # Only a soft penalty: positive if clearance is below 0.05 m.
        names = self._names(q_work, q_work.shape[1])
        error = 0.0
        for i in range(q_work.shape[0]):
            clearance = self.collision_checker.minimum_distance(
                JointState(names, q_work[i])
            )
            if clearance < 0.05:
                error += (0.05 - clearance) ** 2
        return error

    def _cost(
        self,
        x: NDArray[np.float64],
        q: NDArray[np.float64],
        free_start: int,
        free_end: int,
    ) -> float:
        q_work = self._decode(x, q, free_start, free_end)
        target = self._target_path(q_work)
        return (
            self._task_cost(q_work, target)
            + self._velocity_cost(q_work)
            + self._acceleration_cost(q_work)
            + self._collision_cost(q_work)
        )

    def _numerical_gradient(
        self,
        x: NDArray[np.float64],
        q: NDArray[np.float64],
        free_start: int,
        free_end: int,
        eps: float = 1e-6,
    ) -> NDArray[np.float64]:
        grad = np.zeros_like(x)
        for i in range(x.size):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (
                self._cost(x_plus, q, free_start, free_end)
                - self._cost(x_minus, q, free_start, free_end)
            ) / (2.0 * eps)
        return grad

    def _optimize_scipy(
        self,
        x0: NDArray[np.float64],
        q: NDArray[np.float64],
        free_start: int,
        free_end: int,
        method: str,
    ) -> NDArray[np.float64]:
        scipy_method = "L-BFGS-B" if method == "lbfgs" else "Newton-CG"
        maxiter = max(500, self.max_iterations_gradient * 5)
        result = self._scipy_minimize(
            fun=lambda x: self._cost(x, q, free_start, free_end),
            x0=x0,
            method=scipy_method,
            jac=lambda x: self._numerical_gradient(x, q, free_start, free_end),
            options={"maxiter": maxiter},
        )
        x_opt = np.asarray(result.x, dtype=float)
        # If SciPy stops early (e.g. maxiter), still return the best iterate it
        # found.  This keeps the demo usable even when the optimizer is not fully
        # converged.
        if np.all(np.isfinite(x_opt)):
            return x_opt
        raise RuntimeError(
            f"trajectory optimization failed: {result.message}"
        )

    def _optimize_gradient(
        self,
        x: NDArray[np.float64],
        q: NDArray[np.float64],
        free_start: int,
        free_end: int,
    ) -> NDArray[np.float64]:
        x_cur = x.copy()
        step = self.gradient_step_start

        for _ in range(self.max_iterations_gradient):
            grad = self._numerical_gradient(x_cur, q, free_start, free_end)
            norm = float(np.linalg.norm(grad))
            if norm < 1e-10:
                break

            direction = -grad / max(norm, 1e-12)
            cost_cur = self._cost(x_cur, q, free_start, free_end)

            # backtracking line search
            step_local = step
            accepted = False
            for _ in range(20):
                x_next = x_cur + step_local * direction
                cost_next = self._cost(x_next, q, free_start, free_end)
                if cost_next < cost_cur:
                    x_cur = x_next
                    step = min(step * 1.2, 1.0)
                    accepted = True
                    break
                step_local *= 0.5
            if not accepted:
                step *= 0.5
                if step < 1e-6:
                    break

        return x_cur
