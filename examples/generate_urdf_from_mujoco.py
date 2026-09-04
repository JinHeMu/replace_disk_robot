#!/usr/bin/env python3
"""Generate a minimal URDF directly from the MuJoCo body/joint tree.

The output preserves the MuJoCo body frame tree (including joint anchors via
extra fixed joint-frame links when a MuJoCo joint has a non-zero local `pos`).
Collision/visual geometry is intentionally not converted: this is a kinematic
reference URDF for Pinocchio and similar libraries.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_XML = PROJECT_ROOT / "simulation" / "mujoco" / "models" / "ur5e" / "ur5e_with_gripper.xml"
DEFAULT_URDF = PROJECT_ROOT / "simulation" / "mujoco" / "models" / "ur5e" / "ur5e_from_mujoco.urdf"


def quat_to_mat(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert unit quaternion wxyz to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_rpy(r: NDArray[np.float64]) -> tuple[float, float, float]:
    """Extract URDF rpy (roll, pitch, yaw) from a rotation matrix."""
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    sy = math.hypot(r[0, 0], r[1, 0])
    if sy > 1e-12:
        roll = math.atan2(r[2, 1], r[2, 2])
        pitch = math.atan2(-r[2, 0], sy)
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:
        roll = math.atan2(-r[1, 2], r[1, 1])
        pitch = math.atan2(-r[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def fmt(v: float) -> str:
    return f"{float(v):.10g}"


def transform_body_quat_to_rpy(quat: NDArray[np.float64]) -> tuple[float, float, float]:
    return mat_to_rpy(quat_to_mat(quat))


def joint_origin_from_mujoco(
    body_pos: NDArray[np.float64],
    body_quat: NDArray[np.float64],
    joint_pos: NDArray[np.float64] | None,
) -> tuple[list[float], list[float]]:
    """Compute URDF joint origin (xyz, rpy) from a MuJoCo body and joint anchor."""
    r = quat_to_mat(body_quat)
    xyz = body_pos.copy()
    if joint_pos is not None and np.any(joint_pos != 0.0):
        xyz = xyz + r @ joint_pos
    roll, pitch, yaw = transform_body_quat_to_rpy(body_quat)
    return xyz.tolist(), [roll, pitch, yaw]


def inertial_to_urdf(model: mujoco.MjModel, body_id: int) -> dict | None:
    """Extract MuJoCo inertial as URDF inertial data (full inertia matrix)."""
    mass = float(model.body_mass[body_id])
    if mass <= 0.0:
        return None
    pos = model.body_ipos[body_id].copy()
    quat = model.body_iquat[body_id].copy()
    r = quat_to_mat(quat)
    diag = np.diag(model.body_inertia[body_id])
    inertia_matrix = r @ diag @ r.T
    # URDF stores symmetric inertia entries in a fixed order.
    ixx, ixy, ixz = inertia_matrix[0, 0], inertia_matrix[0, 1], inertia_matrix[0, 2]
    iyy, iyz = inertia_matrix[1, 1], inertia_matrix[1, 2]
    izz = inertia_matrix[2, 2]
    return {
        "mass": mass,
        "xyz": pos.tolist(),
        "rpy": list(transform_body_quat_to_rpy(quat)),
        "inertia": (ixx, ixy, ixz, iyy, iyz, izz),
    }


def generate() -> str:
    model = mujoco.MjModel.from_xml_path(str(MUJOCO_XML))
    lines: list[str] = []
    lines.append('<?xml version="1.0" ?>')
    lines.append('<!-- Auto-generated from simulation/mujoco/models/ur5e/ur5e_with_gripper.xml -->')
    lines.append('<robot name="ur5e_2f85_from_mujoco">')
    lines.append('  <link name="world"/>')

    def add_link(name: str, inertial: dict | None = None) -> None:
        lines.append(f'  <link name="{name}">')
        if inertial is not None:
            xyz = " ".join(fmt(x) for x in inertial["xyz"])
            rpy = " ".join(fmt(x) for x in inertial["rpy"])
            ixx, ixy, ixz, iyy, iyz, izz = inertial["inertia"]
            lines.append(f'    <inertial>')
            lines.append(f'      <origin xyz="{xyz}" rpy="{rpy}"/>')
            lines.append(f'      <mass value="{fmt(inertial["mass"])}"/>')
            lines.append(
                f'      <inertia ixx="{fmt(ixx)}" ixy="{fmt(ixy)}" ixz="{fmt(ixz)}" '
                f'iyy="{fmt(iyy)}" iyz="{fmt(iyz)}" izz="{fmt(izz)}"/>'
            )
            lines.append(f'    </inertial>')
        lines.append('  </link>')

    def add_joint(
        name: str,
        joint_type: str,
        parent: str,
        child: str,
        xyz: list[float],
        rpy: list[float],
        axis: list[float] | None = None,
        limits: tuple[float, float] | None = None,
    ) -> None:
        lines.append(f'  <joint name="{name}" type="{joint_type}">')
        lines.append(f'    <origin xyz="{" ".join(fmt(x) for x in xyz)}" rpy="{" ".join(fmt(x) for x in rpy)}"/>')
        lines.append(f'    <parent link="{parent}"/>')
        lines.append(f'    <child link="{child}"/>')
        if joint_type != "fixed":
            if axis is not None:
                lines.append(f'    <axis xyz="{" ".join(fmt(x) for x in axis)}"/>')
            if limits is not None:
                lower, upper = limits
                lines.append(f'    <limit lower="{fmt(lower)}" upper="{fmt(upper)}" effort="1000" velocity="10"/>')
        lines.append('  </joint>')

    # Pre-create all links so URDF link ordering is stable (not strictly required).
    body_names: list[str] = []
    inertial_data: dict[int, dict | None] = {}
    for body_id in range(1, model.nbody):
        name = str(model.body(body_id).name)
        body_names.append(name)
        inertial_data[body_id] = inertial_to_urdf(model, body_id)
        add_link(name, inertial_data[body_id])

    # Add body/joint connections.
    for body_id in range(1, model.nbody):
        name = str(model.body(body_id).name)
        parent_id = model.body_parentid[body_id]
        parent_name = str(model.body(parent_id).name)
        body_pos = model.body_pos[body_id].copy()
        body_quat = model.body_quat[body_id].copy()
        jnt_adr = model.body_jntadr[body_id]
        jnt_num = model.body_jntnum[body_id]

        if jnt_num == 0:
            roll, pitch, yaw = transform_body_quat_to_rpy(body_quat)
            add_joint(f"{parent_name}_to_{name}", "fixed", parent_name, name,
                      body_pos.tolist(), [roll, pitch, yaw])
            continue
        if jnt_num != 1:
            raise RuntimeError(f"Body '{name}' has {jnt_num} joints; converter only supports 0/1.")

        joint_id = jnt_adr
        jnt_name = str(model.joint(joint_id).name)
        jnt_type_code = int(model.jnt_type[joint_id])
        jnt_axis = model.jnt_axis[joint_id].copy().tolist()
        jnt_pos = model.jnt_pos[joint_id].copy()
        lower, upper = map(float, model.jnt_range[joint_id])

        if jnt_type_code == int(mujoco.mjtJoint.mjJNT_FREE):
            joint_type = "floating"
            xyz, rpy = body_pos.tolist(), list(transform_body_quat_to_rpy(body_quat))
            child = name
            add_joint(jnt_name, joint_type, parent_name, child, xyz, rpy)
        elif jnt_type_code == int(mujoco.mjtJoint.mjJNT_SLIDE):
            joint_type = "prismatic"
            xyz, rpy = joint_origin_from_mujoco(body_pos, body_quat, jnt_pos)
            if np.any(jnt_pos != 0.0):
                joint_frame = f"{name}_joint_frame"
                add_link(joint_frame)
                add_joint(jnt_name, joint_type, parent_name, joint_frame, xyz, rpy,
                          axis=jnt_axis, limits=(lower, upper))
                add_joint(f"{joint_frame}_to_{name}", "fixed", joint_frame, name,
                          (-jnt_pos).tolist(), [0.0, 0.0, 0.0])
            else:
                add_joint(jnt_name, joint_type, parent_name, name, xyz, rpy,
                          axis=jnt_axis, limits=(lower, upper))
        elif jnt_type_code == int(mujoco.mjtJoint.mjJNT_HINGE):
            joint_type = "revolute"
            xyz, rpy = joint_origin_from_mujoco(body_pos, body_quat, jnt_pos)
            if np.any(jnt_pos != 0.0):
                joint_frame = f"{name}_joint_frame"
                add_link(joint_frame)
                add_joint(jnt_name, joint_type, parent_name, joint_frame, xyz, rpy,
                          axis=jnt_axis, limits=(lower, upper))
                add_joint(f"{joint_frame}_to_{name}", "fixed", joint_frame, name,
                          (-jnt_pos).tolist(), [0.0, 0.0, 0.0])
            else:
                add_joint(jnt_name, joint_type, parent_name, name, xyz, rpy,
                          axis=jnt_axis, limits=(lower, upper))
        else:
            raise RuntimeError(f"Unsupported MuJoCo joint type {jnt_type_code} on {name}")

    lines.append('</robot>')
    output = "\n".join(lines) + "\n"
    DEFAULT_URDF.write_text(output, encoding="utf-8")
    print(f"Generated {DEFAULT_URDF}")
    return output


if __name__ == "__main__":
    generate()
