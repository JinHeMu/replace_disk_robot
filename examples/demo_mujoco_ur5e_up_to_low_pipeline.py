#!/usr/bin/env python3
"""UR5e empty-scene MuJoCo pipeline example: search -> optimize -> execute.

The example loads a bare UR5e + gripper model, removes the held replacement
drive body, and therefore does not load the server/insertion scene.

It starts from an explicit ``up`` joint posture and plans a large-amplitude
``low head`` posture computed by MuJoCo-site IK: the tool center is driven to
``(0.35, 0.0, 0.10) m`` with the gripper approach axis pointing downward.

Pipeline:
    UP joints
      -> RRT-Connect search
      -> joint-space trajectory optimization
      -> quintic time allocation
      -> P3 MotionPipeline execution on the MuJoCo position actuators

Run with the MuJoCo viewer:
    python3 examples/demo_mujoco_ur5e_up_to_low_pipeline.py --viewer

Run headless:
    python3 examples/demo_mujoco_ur5e_up_to_low_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replace_disk_robot.adapters.mujoco import (
    MujocoCollisionChecker,
    MujocoRobotAdapter,
)
from replace_disk_robot.adapters.mujoco.collision import DEFAULT_ROBOT_BODIES
from replace_disk_robot.core import JointState
from replace_disk_robot.planning.optimize import TrajectoryOptimizer
from replace_disk_robot.planning.search import RRTConnectPlanner
from replace_disk_robot.task import MotionPipeline, MotionPipelineState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UR5E_EMPTY_XML = (
    PROJECT_ROOT
    / "simulation"
    / "mujoco"
    / "models"
    / "ur5e"
    / "ur5e_with_gripper.xml"
)
ROBOT_BODIES_WITHOUT_DRIVE = tuple(
    name for name in DEFAULT_ROBOT_BODIES if name != "replacement_drive_held"
)

# Up posture: the original scene home joint values.
UP_Q = np.array(
    [
        2.45232094662072,
        -1.99294546015135,
        2.53041379518691,
        2.60412431855423,
        -0.88152461982582,
        -1.5707963267949,
    ],
    dtype=float,
)

# Low-head posture computed with MuJoCo-site IK:
#   pinch position = (0.35, 0.0, 0.10) m
#   gripper approach axis = world -Z
LOW_HEAD_Q = np.array(
    [
        2.7487055416859194,
        -1.8389576461494255,
        2.3803533337366063,
        4.170993292797509,
        -1.5707963267948966,
        -1.9636834386987707,
    ],
    dtype=float,
)


def _load_empty_ur5e_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load UR5e + gripper only, then remove the held drive body."""
    spec = mujoco.MjSpec.from_file(str(UR5E_EMPTY_XML))
    try:
        drive_body = spec.body("replacement_drive_held")
    except Exception as exc:
        raise RuntimeError(
            "UR5e model no longer contains the expected replacement drive body"
        ) from exc
    spec.delete(drive_body)
    model = spec.compile()
    return model, mujoco.MjData(model)


def _set_arm_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: tuple[str, ...],
    q: np.ndarray,
) -> None:
    for name, value in zip(joint_names, q):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = float(value)
    mujoco.mj_forward(model, data)


class _JointNameKinematics:
    """Minimal handle because the example uses a joint goal and task_weight=0."""

    def __init__(self, joint_names: tuple[str, ...]) -> None:
        self.joint_names = tuple(joint_names)

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def forward(self, joints: JointState):
        raise NotImplementedError("this example uses a joint-space goal")

    def inverse(self, target, seed):
        raise NotImplementedError("this example uses a joint-space goal")


def _joint_limits(
    model: mujoco.MjModel,
    joint_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in joint_names
    ]
    if any(joint_id < 0 for joint_id in joint_ids):
        raise KeyError("UR5e model is missing one or more arm joints")
    ranges = model.jnt_range[np.asarray(joint_ids, dtype=int)]
    return ranges[:, 0].copy(), ranges[:, 1].copy()


def _build_pipeline(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot: MujocoRobotAdapter,
    checker: MujocoCollisionChecker,
    lower: np.ndarray,
    upper: np.ndarray,
) -> MotionPipeline:
    planner = RRTConnectPlanner(
        checker,
        sample_bounds=list(zip(lower.tolist(), upper.tolist())),
        step_size_rad=0.15,
        max_iterations=10000,
        max_connection_steps=300,
        random_seed=0,
        path_pruning=False,
    )
    # Smoothness-only optimization from the raw joint-space RRT path.
    # The Cartesian task is intentionally disabled in P1.
    optimizer = TrajectoryOptimizer(
        kinematics=None,
        collision_checker=None,
        task_weight=0.0,
        velocity_weight=10.0,
        acceleration_weight=2.0,
        dt_s=0.05,
        method="gradient",
        max_iterations_gradient=80,
    )
    return MotionPipeline(
        _JointNameKinematics(tuple(robot.joint_names)),
        checker,
        planner,
        optimizer,
        optimize=True,
        optimization_samples=18,
        optimization_task_weight=0.0,
        joint_limits_rad=(lower, upper),
        arm=robot,
        velocity_limits_rad_s=np.full(len(robot.joint_names), 0.5),
        acceleration_limits_rad_s2=np.full(len(robot.joint_names), 1.0),
        min_segment_time_s=0.08,
        start_tolerance_rad=0.05,
        tracking_tolerance_rad=0.6,
        goal_tolerance_rad=0.02,
        goal_hold_s=0.1,
        timeout_s=5.0,
    )


def _viewer_preflight() -> tuple[bool, str | None]:
    """Check whether a GLFW/OpenGL context can be created headlessly."""
    try:
        import glfw
    except Exception as exc:  # noqa: BLE001 - optional GUI dependency
        return False, str(exc)
    try:
        if not glfw.init():
            return False, "glfw.init() failed"
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(64, 64, "mujoco-viewer-preflight", None, None)
        if not window:
            glfw.terminate()
            return False, "could not create a GLFW/OpenGL window"
        glfw.make_context_current(window)
        glfw.destroy_window(window)
        glfw.terminate()
        return True, None
    except Exception as exc:  # noqa: BLE001 - GUI backend error types vary
        try:
            glfw.terminate()
        except Exception:
            pass
        return False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "simulation"
        / "mujoco"
        / "reports"
        / "ur5e_up_to_low_pipeline_report.json",
        help="JSON report output path",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help=(
            "open the MuJoCo passive viewer while executing; "
            "requires a working OpenGL/GLFW display"
        ),
    )
    parser.add_argument("--sim-steps-per-update", type=int, default=10)
    args = parser.parse_args()

    model, data = _load_empty_ur5e_model()
    model.opt.timestep = 0.001
    robot = MujocoRobotAdapter(model, data)
    joint_names = tuple(robot.joint_names)
    _set_arm_qpos(model, data, joint_names, UP_Q)
    checker = MujocoCollisionChecker(
        model,
        data,
        robot_bodies=ROBOT_BODIES_WITHOUT_DRIVE,
        use_separate_planning_data=True,
    )

    viewer = None
    if args.viewer:
        viewer_ok, viewer_error = _viewer_preflight()
        if not viewer_ok:
            print(
                f"MuJoCo viewer unavailable ({viewer_error}); "
                "continuing headless."
            )
        else:
            try:
                from mujoco import viewer as mujoco_viewer

                viewer = mujoco_viewer.launch_passive(model, data)
                viewer.sync()
                print("MuJoCo viewer opened. Close the viewer window to stop.")
            except Exception as exc:  # noqa: BLE001 - GUI errors vary
                print(
                    f"Could not open MuJoCo viewer ({exc}); "
                    "continuing headless."
                )

    up_state = robot.read_joint_state()
    low_state = JointState(joint_names, LOW_HEAD_Q)
    lower, upper = _joint_limits(model, joint_names)

    if not checker.is_collision_free(up_state):
        raise RuntimeError("UR5e up state is in collision")
    if not checker.is_collision_free(low_state):
        raise RuntimeError("UR5e low head state is in collision")

    print("UP pinch position [m]: ", data.site("pinch").xpos.tolist())
    _set_arm_qpos(model, data, joint_names, LOW_HEAD_Q)
    print("LOW HEAD pinch position [m]: ", data.site("pinch").xpos.tolist())
    _set_arm_qpos(model, data, joint_names, UP_Q)
    pipeline = _build_pipeline(model, data, robot, checker, lower, upper)
    plan = pipeline.submit_goal(low_state, now_s=float(data.time))

    print(f"Search raw waypoints: {len(plan.raw_trajectory)}")
    print(f"Optimized waypoints:  {len(plan.trajectory)}")
    print(f"Timed duration [s]:   {pipeline.status.trajectory_duration_s}")

    while pipeline.state == MotionPipelineState.EXECUTING and data.time < 30.0:
        pipeline.update(float(data.time))
        for _ in range(args.sim_steps_per_update):
            mujoco.mj_step(model, data)
        if viewer is not None:
            if not viewer.is_running():
                if pipeline.state == MotionPipelineState.EXECUTING:
                    pipeline.cancel(float(data.time))
                break
            viewer.sync()

    final_state = robot.read_joint_state()
    final_error = float(np.max(np.abs(final_state.position_rad - low_state.position_rad)))
    report = {
        "state": pipeline.state.value,
        "raw_waypoints": len(plan.raw_trajectory),
        "optimized_waypoints": len(plan.trajectory),
        "trajectory_duration_s": pipeline.status.trajectory_duration_s,
        "tracking_error_rad": pipeline.status.tracking_error_rad,
        "final_error_rad": final_error,
        "failure_reason": pipeline.status.last_error,
        "up_position_rad": up_state.position_rad.tolist(),
        "low_target_rad": low_state.position_rad.tolist(),
        "final_position_rad": final_state.position_rad.tolist(),
        "sim_time_s": float(data.time),
    }

    print(json.dumps(report, indent=2))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to {args.report}")

    if viewer is not None:
        print("Execution finished; close the viewer window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)
        viewer.close()

    if pipeline.state != MotionPipelineState.SUCCEEDED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
