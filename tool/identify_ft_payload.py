#!/usr/bin/env python3
"""Identify payload mass, center of mass and F/T bias from static CSV data.

The input is produced by collect_ft_gravity_data.py.  The model uses the raw
sensor-frame wrench at static poses:

    f_s = R_bs.T @ h_b + b_f
    tau_s = r_sc x (R_bs.T @ h_b) + b_tau

Here h_b is the signed payload gravity vector in the robot base frame,
r_sc is the sensor-origin-to-payload-CoM vector, and b_f/b_tau are constant
sensor offsets.  The signed gravity vector makes the fit independent of the
sensor's force sign convention; mass is norm(h_b) / g.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FORCE_COLUMNS = ("raw_fx_n", "raw_fy_n", "raw_fz_n")
TORQUE_COLUMNS = ("raw_tx_nm", "raw_ty_nm", "raw_tz_nm")
ROTATION_COLUMNS = tuple(
    f"r_base_sensor_{row}{col}" for row in range(3) for col in range(3)
)


@dataclass(frozen=True)
class StaticPose:
    capture_id: int
    samples: int
    rotation_base_sensor: np.ndarray
    force_sensor_n: np.ndarray
    torque_sensor_nm: np.ndarray


def _float(row: dict[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid CSV field {name!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"CSV field {name!r} is not finite")
    return value


def _project_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _singular, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def load_static_poses(path: Path, min_samples: int) -> tuple[list[StaticPose], dict[str, np.ndarray]]:
    groups: dict[int, list[dict[str, str]]] = {}
    transform_rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"capture_id", *FORCE_COLUMNS, *TORQUE_COLUMNS, *ROTATION_COLUMNS}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            capture_id = int(row["capture_id"])
            if capture_id < 0:
                continue
            groups.setdefault(capture_id, []).append(row)
            transform_rows.append(row)

    poses: list[StaticPose] = []
    for capture_id in sorted(groups):
        rows = groups[capture_id]
        if len(rows) < min_samples:
            print(
                f"[identify] skip capture {capture_id}: {len(rows)} samples "
                f"(< {min_samples})"
            )
            continue
        rotations = np.array([
            [_float(row, name) for name in ROTATION_COLUMNS] for row in rows
        ]).reshape(-1, 3, 3)
        forces = np.array([
            [_float(row, name) for name in FORCE_COLUMNS] for row in rows
        ])
        torques = np.array([
            [_float(row, name) for name in TORQUE_COLUMNS] for row in rows
        ])
        poses.append(
            StaticPose(
                capture_id=capture_id,
                samples=len(rows),
                rotation_base_sensor=_project_rotation(np.mean(rotations, axis=0)),
                force_sensor_n=np.median(forces, axis=0),
                torque_sensor_nm=np.median(torques, axis=0),
            )
        )

    transforms: dict[str, np.ndarray] = {}
    if transform_rows and all(
        f"r_sensor_tool_{row}{col}" in transform_rows[0]
        for row in range(3) for col in range(3)
    ):
        first = transform_rows[0]
        transforms["rotation_sensor_to_tool"] = _project_rotation(np.array([
            _float(first, f"r_sensor_tool_{row}{col}")
            for row in range(3) for col in range(3)
        ]).reshape(3, 3))
        arm_names = ("tool_to_sensor_x_m", "tool_to_sensor_y_m", "tool_to_sensor_z_m")
        if all(name in first for name in arm_names):
            transforms["tool_to_sensor_m"] = np.array([_float(first, name) for name in arm_names])
    return poses, transforms


def _robust_lstsq(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    blocks: int,
    huber_delta: float,
    iterations: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively reweighted least squares with one weight per 3D pose."""

    weights = np.ones(blocks)
    solution = np.linalg.lstsq(matrix, target, rcond=None)[0]
    for _ in range(iterations):
        row_weights = np.repeat(np.sqrt(weights), 3)
        weighted_matrix = matrix * row_weights[:, None]
        weighted_target = target * row_weights
        updated = np.linalg.lstsq(weighted_matrix, weighted_target, rcond=None)[0]
        residual = (matrix @ updated - target).reshape(blocks, 3)
        norms = np.linalg.norm(residual, axis=1)
        median = float(np.median(norms))
        scale = max(1e-12, 1.4826 * float(np.median(np.abs(norms - median))))
        cutoff = max(1e-12, median + huber_delta * scale)
        new_weights = np.ones_like(norms)
        mask = norms > cutoff
        new_weights[mask] = cutoff / norms[mask]
        solution = updated
        if np.max(np.abs(new_weights - weights)) < 1e-6:
            weights = new_weights
            break
        weights = new_weights
    return solution, weights


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def identify_payload(
    poses: list[StaticPose],
    *,
    gravity_m_s2: float = 9.80665,
    huber_delta: float = 2.5,
) -> dict[str, object]:
    if len(poses) < 6:
        raise ValueError("at least 6 accepted static poses are required; >=12 is recommended")
    rotations = [pose.rotation_base_sensor for pose in poses]
    forces = np.array([pose.force_sensor_n for pose in poses])
    torques = np.array([pose.torque_sensor_nm for pose in poses])

    force_matrix = np.vstack([
        np.hstack((rotation.T, np.eye(3))) for rotation in rotations
    ])
    if np.linalg.matrix_rank(force_matrix) < 6:
        raise ValueError("force fit is rank deficient; collect more diverse tool orientations")
    force_solution, force_weights = _robust_lstsq(
        force_matrix, forces.reshape(-1), blocks=len(poses), huber_delta=huber_delta
    )
    gravity_vector_base = force_solution[:3]
    force_bias = force_solution[3:]
    gravity_norm = float(np.linalg.norm(gravity_vector_base))
    if gravity_norm < 1e-6:
        raise ValueError(
            "identified gravity load is nearly zero; verify that CSV contains raw, "
            "uncompensated sensor data and a non-zero payload"
        )
    gravity_forces_sensor = np.array([
        rotation.T @ gravity_vector_base for rotation in rotations
    ])

    torque_matrix = np.vstack([
        np.hstack((-_skew(force), np.eye(3))) for force in gravity_forces_sensor
    ])
    if np.linalg.matrix_rank(torque_matrix) < 6:
        raise ValueError("torque fit is rank deficient; collect more diverse tool orientations")
    torque_solution, torque_weights = _robust_lstsq(
        torque_matrix, torques.reshape(-1), blocks=len(poses), huber_delta=huber_delta
    )
    center_of_mass_sensor = torque_solution[:3]
    torque_bias = torque_solution[3:]

    predicted_force = (force_matrix @ force_solution).reshape(-1, 3)
    predicted_torque = (torque_matrix @ torque_solution).reshape(-1, 3)
    force_residual = forces - predicted_force
    torque_residual = torques - predicted_torque
    vertical_alignment_deg = math.degrees(math.acos(float(np.clip(
        abs(gravity_vector_base[2]) / gravity_norm, 0.0, 1.0
    ))))
    force_condition = float(np.linalg.cond(force_matrix))
    torque_condition = float(np.linalg.cond(torque_matrix))

    return {
        "model": "static_raw_sensor_wrench",
        "pose_count": len(poses),
        "sample_count": sum(pose.samples for pose in poses),
        "gravity_m_s2": gravity_m_s2,
        "mass_kg": gravity_norm / gravity_m_s2,
        "signed_gravity_force_base_n": gravity_vector_base.tolist(),
        "gravity_direction_base_unit": (gravity_vector_base / gravity_norm).tolist(),
        "vertical_alignment_error_deg": vertical_alignment_deg,
        "center_of_mass_sensor_m": center_of_mass_sensor.tolist(),
        "center_of_mass_sensor_mm": (1000.0 * center_of_mass_sensor).tolist(),
        "force_bias_sensor_n": force_bias.tolist(),
        "torque_bias_sensor_nm": torque_bias.tolist(),
        "fit": {
            "force_rms_n": float(np.sqrt(np.mean(force_residual ** 2))),
            "force_peak_n": float(np.max(np.linalg.norm(force_residual, axis=1))),
            "torque_rms_nm": float(np.sqrt(np.mean(torque_residual ** 2))),
            "torque_peak_nm": float(np.max(np.linalg.norm(torque_residual, axis=1))),
            "force_matrix_condition": force_condition,
            "torque_matrix_condition": torque_condition,
            "force_pose_weights": force_weights.tolist(),
            "torque_pose_weights": torque_weights.tolist(),
        },
        "capture_ids": [pose.capture_id for pose in poses],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV from collect_ft_gravity_data.py")
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON result path (default: CSV name + _identified.json)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing result JSON")
    parser.add_argument("--min-samples", type=int, default=50,
                        help="minimum rows required for each capture")
    parser.add_argument("--gravity", type=float, default=9.80665)
    parser.add_argument("--huber-delta", type=float, default=2.5)
    parser.add_argument("--max-com-m", type=float, default=0.5,
                        help="warn if identified sensor-to-CoM distance exceeds this")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.min_samples <= 0 or args.gravity <= 0 or args.huber_delta <= 0:
        raise SystemExit("--min-samples, --gravity and --huber-delta must be positive")
    csv_path = args.csv.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else csv_path.with_name(f"{csv_path.stem}_identified.json")
    )
    try:
        poses, transforms = load_static_poses(csv_path, args.min_samples)
        result = identify_payload(
            poses, gravity_m_s2=args.gravity, huber_delta=args.huber_delta
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"[identify] FAILED: {exc}") from exc

    if "rotation_sensor_to_tool" in transforms and "tool_to_sensor_m" in transforms:
        r_sensor_tool = transforms["rotation_sensor_to_tool"]
        tool_to_sensor = transforms["tool_to_sensor_m"]
        com_sensor = np.asarray(result["center_of_mass_sensor_m"], dtype=float)
        com_tool = tool_to_sensor + r_sensor_tool @ com_sensor
        result["center_of_mass_tool_m"] = com_tool.tolist()
        result["center_of_mass_tool_mm"] = (1000.0 * com_tool).tolist()
        result["sensor_to_tool_rotation"] = r_sensor_tool.tolist()
        result["tool_to_sensor_m"] = tool_to_sensor.tolist()

    warnings: list[str] = []
    if result["vertical_alignment_error_deg"] > 15.0:
        warnings.append(
            "fitted gravity is more than 15 deg from base vertical; check FK, "
            "sensor-to-tool rotation, force sign/frame, and compensation mode"
        )
    if np.linalg.norm(result["center_of_mass_sensor_m"]) > args.max_com_m:
        warnings.append("identified sensor-to-CoM distance is physically implausible")
    fit = result["fit"]
    if fit["force_matrix_condition"] > 100.0 or fit["torque_matrix_condition"] > 100.0:
        warnings.append("identification is poorly conditioned; collect more diverse orientations")
    result["warnings"] = warnings

    if output.exists() and not args.overwrite:
        raise SystemExit(
            f"[identify] output JSON already exists: {output} "
            "(choose another --output or pass --overwrite)"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[identify] poses: {result['pose_count']} ({result['sample_count']} samples)")
    print(f"[identify] mass: {result['mass_kg']:.6f} kg")
    print(
        "[identify] CoM from sensor: "
        + np.array2string(np.asarray(result["center_of_mass_sensor_mm"]), precision=3)
        + " mm"
    )
    if "center_of_mass_tool_mm" in result:
        print(
            "[identify] CoM from tool:   "
            + np.array2string(np.asarray(result["center_of_mass_tool_mm"]), precision=3)
            + " mm"
        )
    print(
        f"[identify] residual RMS: force={fit['force_rms_n']:.5f} N, "
        f"torque={fit['torque_rms_nm']:.6f} Nm"
    )
    for warning in warnings:
        print(f"[identify] WARNING: {warning}")
    print(f"[identify] JSON: {output}")


if __name__ == "__main__":
    main()
