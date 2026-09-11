#!/usr/bin/env python3
"""Fail if the project imports Python/native packages from outside Conda."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ISOLATED_PATH_VARIABLES = (
    "PYTHONPATH",
    "AMENT_PREFIX_PATH",
    "COLCON_PREFIX_PATH",
    "CMAKE_PREFIX_PATH",
    "LD_LIBRARY_PATH",
    "PKG_CONFIG_PATH",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "LIBRARY_PATH",
)
FORBIDDEN_PATH_PARTS = ("/opt/ros/", "/.local/", "/.mujoco/")


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _module_path(module: ModuleType) -> Path:
    location = getattr(module, "__file__", None)
    if not location:
        raise RuntimeError(f"module {module.__name__!r} has no filesystem location")
    return Path(location).resolve()


def _check_environment_variables(errors: list[str]) -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        errors.append("PYTHONNOUSERSITE must be 1")
    for name in ISOLATED_PATH_VARIABLES:
        value = os.environ.get(name, "")
        if value:
            errors.append(f"{name} must be empty, got {value!r}")
    for name in ("ROS_DISTRO", "ROS_VERSION", "ROS_PYTHON_VERSION"):
        if os.environ.get(name, ""):
            errors.append(f"{name} must be empty")


def main() -> None:
    errors: list[str] = []
    prefix = Path(sys.prefix).resolve()
    executable = Path(sys.executable).resolve()
    active_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if Path(active_env).name != "replace_disk_robot" or prefix.name != "replace_disk_robot":
        errors.append("the active Conda environment must be replace_disk_robot")
    if not _inside(executable, prefix):
        errors.append(f"Python executable is outside CONDA_PREFIX: {executable}")
    if site.ENABLE_USER_SITE:
        errors.append("Python user site-packages is enabled")

    _check_environment_variables(errors)
    for entry in sys.path:
        if any(part in entry for part in FORBIDDEN_PATH_PARTS):
            errors.append(f"forbidden sys.path entry: {entry}")

    import mujoco
    import numpy
    import pinocchio

    modules = (numpy, mujoco, pinocchio)
    for module in modules:
        location = _module_path(module)
        if not _inside(location, prefix):
            errors.append(f"{module.__name__} loaded outside Conda: {location}")

    from replace_disk_robot.adapters.mujoco import load_model
    from replace_disk_robot.core import JointState
    from replace_disk_robot.kinematics import UR5eKinematics

    model, _ = load_model()
    kinematics = UR5eKinematics()
    home = JointState(kinematics.joint_names, [0.0] * kinematics.dof)
    pose = kinematics.forward(home)

    print(f"python={sys.version.split()[0]} ({executable})")
    for module in modules:
        print(
            f"{module.__name__}={module.__version__} "
            f"({_module_path(module)})"
        )
    print(f"MuJoCo model: nq={model.nq}, nbody={model.nbody}")
    print(
        "Pinocchio UR5e: "
        f"dof={kinematics.dof}, zero_position={pose.position_m.round(6).tolist()}"
    )

    if errors:
        print("\nISOLATION CHECK: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print("ISOLATION CHECK: PASS")


if __name__ == "__main__":
    main()
