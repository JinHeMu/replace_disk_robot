"""RRT-Connect joint-space planner.

This module implements :class:`TrajectoryPlannerPort` using the backend-neutral
:class:`CollisionCheckerPort`.  It performs random sampling and bidirectional
tree expansion in joint space.  The raw tree path is optionally shortened with
a greedy line-of-sight pruning pass, which removes unnecessary zig-zags while
keeping every shortcut collision-free.  No MuJoCo/Pinocchio/FCL dependency is
imported here; the concrete collision behaviour is supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..core import CollisionCheckerPort, JointState, TrajectoryPoint


@dataclass
class _Node:
    q: NDArray[np.float64]
    parent: int | None


def _validate_joints(start: JointState, goal: JointState) -> None:
    if start.names != goal.names:
        raise ValueError("start and goal must use the same joint-name order")
    if not start.names:
        raise ValueError("joint state must not be empty")
    if start.position_rad.shape != goal.position_rad.shape:
        raise ValueError("start and goal must have the same number of joints")


def _distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _joint_state_from_array(
    names: tuple[str, ...], q: NDArray[np.float64]
) -> JointState:
    return JointState(names, q)


class RRTConnectPlanner:
    """Bidirectional RRT planner in joint configuration space.

    Parameters
    ----------
    collision_checker:
        A :class:`CollisionCheckerPort` implementation.
    sample_bounds:
        Joint limits used for random sampling, ordered exactly like the
        planner's joint names.  If not given, a conservative box around
        ``start`` and ``goal`` is used during ``plan``.
    step_size_rad:
        Maximum extension length per RRT step.
    goal_sample_rate:
        Probability of sampling the goal directly.
    max_iterations:
        Maximum number of RRT iterations before giving up.
    max_connection_steps:
        Maximum consecutive extension steps when trying to connect one tree
        to a node from the other tree.
    random_seed:
        If given, makes sampling deterministic.
    path_time_step_s:
        Nominal time between two adjacent RRT waypoints.
    path_pruning:
        If True, apply a greedy shortcut pass to the raw RRT path.  For each
        anchor waypoint, the planner finds the farthest later waypoint that can
        be connected by a collision-free straight segment.  This removes
        unnecessary detours and leaves a simpler path.  The original time stamp
        of every retained waypoint is preserved, so pruning does not increase
        the implied joint speed.  Defaults to True.
    path_prune_step_rad:
        Maximum joint-space interpolation step used when validating pruning
        shortcuts.  Smaller values make the collision check more conservative
        but also more expensive.  If not given, ``step_size_rad`` is used.
    """

    def __init__(
        self,
        collision_checker: CollisionCheckerPort,
        *,
        sample_bounds: Sequence[tuple[float, float]] | None = None,
        step_size_rad: float = 0.1,
        goal_sample_rate: float = 0.05,
        max_iterations: int = 2000,
        max_connection_steps: int = 100,
        random_seed: int | None = None,
        path_time_step_s: float = 0.1,
        path_pruning: bool = True,
        path_prune_step_rad: float | None = None,
    ) -> None:
        if step_size_rad <= 0.0:
            raise ValueError("step_size_rad must be positive")
        if not 0.0 <= goal_sample_rate <= 1.0:
            raise ValueError("goal_sample_rate must be in [0, 1]")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if max_connection_steps <= 0:
            raise ValueError("max_connection_steps must be positive")
        if path_time_step_s <= 0.0:
            raise ValueError("path_time_step_s must be positive")
        if path_prune_step_rad is not None and (
            not np.isfinite(path_prune_step_rad) or path_prune_step_rad <= 0.0
        ):
            raise ValueError("path_prune_step_rad must be finite and positive when provided")

        self.collision_checker = collision_checker
        self.sample_bounds = sample_bounds
        self.step_size_rad = float(step_size_rad)
        self.goal_sample_rate = float(goal_sample_rate)
        self.max_iterations = int(max_iterations)
        self.max_connection_steps = int(max_connection_steps)
        self.random_seed = random_seed
        self.path_time_step_s = float(path_time_step_s)
        self.path_pruning = bool(path_pruning)
        self.path_prune_step_rad = (
            self.step_size_rad
            if path_prune_step_rad is None
            else float(path_prune_step_rad)
        )
        self._rng = np.random.default_rng(random_seed)
        self._joint_names: tuple[str, ...] | None = None

    def plan(
        self, start: JointState, goal: JointState
    ) -> Sequence[TrajectoryPoint]:
        """Plan a collision-free joint-space path from ``start`` to ``goal``."""
        _validate_joints(start, goal)
        self._joint_names = start.names
        if not self.collision_checker.is_collision_free(start):
            raise RuntimeError("start configuration is in collision")
        if not self.collision_checker.is_collision_free(goal):
            raise RuntimeError("goal configuration is in collision")

        start_q = np.asarray(start.position_rad, dtype=float).copy()
        goal_q = np.asarray(goal.position_rad, dtype=float).copy()
        bounds = self._resolve_bounds(start_q, goal_q)

        tree_start: list[_Node] = [_Node(q=start_q, parent=None)]
        tree_goal: list[_Node] = [_Node(q=goal_q, parent=None)]

        for _ in range(self.max_iterations):
            random_q = self._sample(bounds, goal_q)

            if _distance(start_q, random_q) < _distance(goal_q, random_q):
                new_start = self._extend(tree_start, random_q)
                if new_start is not None:
                    connected_goal = self._connect(tree_goal, tree_start[new_start].q)
                    if connected_goal is not None:
                        return self._build_path(
                            start.names,
                            tree_start,
                            tree_goal,
                            new_start,
                            connected_goal,
                        )
            else:
                new_goal = self._extend(tree_goal, random_q)
                if new_goal is not None:
                    connected_start = self._connect(tree_start, tree_goal[new_goal].q)
                    if connected_start is not None:
                        return self._build_path(
                            start.names,
                            tree_start,
                            tree_goal,
                            connected_start,
                            new_goal,
                        )

        raise RuntimeError(
            "RRT-Connect failed to find a collision-free path within "
            f"{self.max_iterations} iterations"
        )

    def _resolve_bounds(
        self,
        start_q: NDArray[np.float64],
        goal_q: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.sample_bounds is not None:
            bounds = np.asarray(self.sample_bounds, dtype=float)
            if bounds.shape != (len(start_q), 2):
                raise ValueError(
                    f"sample_bounds must have shape ({len(start_q)}, 2), got {bounds.shape}"
                )
            if np.any(bounds[:, 0] > bounds[:, 1]):
                raise ValueError("sample_bounds lower limits must not exceed upper limits")
            return bounds

        # Simple fallback: a box that covers both endpoints plus one radian.
        lower = np.minimum(start_q, goal_q) - 1.0
        upper = np.maximum(start_q, goal_q) + 1.0
        return np.column_stack((lower, upper))

    def _sample(
        self,
        bounds: NDArray[np.float64],
        goal_q: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self._rng.random() < self.goal_sample_rate:
            return goal_q.copy()
        return self._rng.uniform(bounds[:, 0], bounds[:, 1])

    def _nearest(
        self,
        tree: list[_Node],
        q: NDArray[np.float64],
    ) -> int:
        best_index = 0
        best_distance = float("inf")
        for i, node in enumerate(tree):
            d = _distance(node.q, q)
            if d < best_distance:
                best_distance = d
                best_index = i
        return best_index

    def _extend(
        self,
        tree: list[_Node],
        target: NDArray[np.float64],
    ) -> int | None:
        """Grow ``tree`` by one step toward ``target``.

        Returns the new node index, or ``None`` if the step is invalid.
        """
        if self._joint_names is None:
            raise RuntimeError("plan() must be called before tree extension")
        nearest_index = self._nearest(tree, target)
        nearest_q = tree[nearest_index].q
        delta = target - nearest_q
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-9:
            return nearest_index

        if norm <= self.step_size_rad:
            new_q = target.copy()
        else:
            new_q = nearest_q + delta / norm * self.step_size_rad

        new_state = _joint_state_from_array(self._joint_names, new_q)

        # The endpoint must be valid and the small segment must not pass through
        # an obstacle.  Segment sampling is intentionally conservative.
        if not self.collision_checker.is_collision_free(new_state):
            return None
        old_state = _joint_state_from_array(self._joint_names, nearest_q)
        if not self._segment_is_free(old_state, new_state):
            return None

        tree.append(_Node(q=new_q, parent=nearest_index))
        return len(tree) - 1

    def _connect(
        self,
        other_tree: list[_Node],
        target: NDArray[np.float64],
    ) -> int | None:
        """Try to connect ``other_tree`` all the way to ``target``."""
        for _ in range(self.max_connection_steps):
            new_index = self._extend(other_tree, target)
            if new_index is None:
                return None
            if _distance(other_tree[new_index].q, target) <= self.step_size_rad * 1e-3:
                return new_index
        return None

    def _segment_is_free(
        self,
        start: JointState,
        goal: JointState,
        samples: int = 8,
    ) -> bool:
        from .collision import is_path_collision_free  # local import avoids cycle

        return is_path_collision_free(
            self.collision_checker,
            start,
            goal,
            samples=samples,
        )

    def _build_path(
        self,
        names: tuple[str, ...],
        tree_start: list[_Node],
        tree_goal: list[_Node],
        start_index: int,
        goal_index: int,
    ) -> Sequence[TrajectoryPoint]:
        forward = self._path_from_tree(names, tree_start, start_index)
        backward_path = self._path_from_tree(names, tree_goal, goal_index)
        # backward_path is from goal root up to connection node; reverse it.
        backward = list(reversed(backward_path))
        raw_path = forward + backward[1:]  # avoid duplicating connection node

        selected = self._prune_path_indices(names, raw_path)
        result: list[TrajectoryPoint] = []
        for index in selected:
            # Preserve the raw waypoint time; long cuts then never imply a
            # higher joint speed than the original RRT segments.
            result.append(
                TrajectoryPoint(
                    time_from_start_s=index * self.path_time_step_s,
                    position_rad=np.asarray(raw_path[index], dtype=float),
                )
            )
        return result

    def _prune_path_indices(
        self,
        names: tuple[str, ...],
        q_path: Sequence[NDArray[np.float64]],
    ) -> list[int]:
        """Return indices of ``q_path`` after greedy line-of-sight pruning.

        The first and last indices are always kept.  For every retained anchor,
        this method searches backwards from the end of the path and keeps the
        farthest waypoint reachable by a collision-free straight segment.
        """
        if not self.path_pruning or len(q_path) <= 2:
            return list(range(len(q_path)))

        selected = [0]
        anchor = 0
        last = len(q_path) - 1
        while anchor < last:
            # Adjacent RRT waypoints are known to be collision-free, so this is
            # a safe fallback when no longer shortcut can be validated.
            next_index = anchor + 1
            for candidate in range(last, anchor + 1, -1):
                if candidate == next_index:
                    break
                if self._path_segment_is_free(
                    names,
                    q_path[anchor],
                    q_path[candidate],
                ):
                    next_index = candidate
                    break
            selected.append(next_index)
            anchor = next_index
        return selected

    def _path_segment_is_free(
        self,
        names: tuple[str, ...],
        q_start: NDArray[np.float64],
        q_goal: NDArray[np.float64],
    ) -> bool:
        """Check a candidate shortcut between two raw joint-space waypoints."""
        start = _joint_state_from_array(names, q_start)
        goal = _joint_state_from_array(names, q_goal)
        distance = _distance(q_start, q_goal)
        # ``is_path_collision_free`` accepts the number of interior samples, so
        # one more sample than ``distance / step`` keeps the validation grid at
        # least as fine as the RRT extension step.
        samples = max(1, int(np.ceil(distance / self.path_prune_step_rad)))
        return self._segment_is_free(start, goal, samples=samples)

    def _path_from_tree(
        self,
        names: tuple[str, ...],
        tree: list[_Node],
        index: int,
    ) -> list[NDArray[np.float64]]:
        path: list[NDArray[np.float64]] = []
        node = tree[index]
        while node is not None:
            path.append(node.q)
            if node.parent is None:
                break
            node = tree[node.parent]
        path.reverse()
        return path

