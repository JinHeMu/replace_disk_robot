#!/usr/bin/env python3
"""Identify payload mass, center of mass and F/T bias from static CSV data.

The input is produced by collect_ft_gravity_data.py.  By default every CSV row
is treated as one quasi-static sample, which is convenient for slow motion
where the collector's still-window detection did not mark accepted captures.
Pass ``--static-poses`` to restore the capture_id window segmentation.  The
model uses the raw sensor-frame wrench at static poses:

    f_s = R_bs.T @ h_b + b_f
    tau_s = r_sc x (R_bs.T @ h_b) + b_tau

Here h_b is the signed payload gravity vector in the robot base frame,
r_sc is the sensor-origin-to-payload-CoM vector, and b_f/b_tau are constant
sensor offsets.  The signed gravity vector makes the fit independent of the
sensor's force sign convention; mass is norm(h_b) / g.

By default the physical down direction is restricted to a 4-degree cone about
base -Z. Both sensor load signs are allowed. This fits mass and bias under the
direction constraint, rather than clipping a free fit afterwards. A zero tilt
limit fixes gravity to base vertical; --gravity-mode free retains the original
unrestricted fit for diagnostics or a verified tilted installation. Constrained
tilt is a model assumption, not an independent inclinometer measurement.
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


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], dict[str, np.ndarray]]:
    """Read all rows and the optional sensor-to-tool metadata."""

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"capture_id", *FORCE_COLUMNS, *TORQUE_COLUMNS, *ROTATION_COLUMNS}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    transforms: dict[str, np.ndarray] = {}
    if rows and all(
        f"r_sensor_tool_{row}{col}" in rows[0]
        for row in range(3) for col in range(3)
    ):
        first = rows[0]
        transforms["rotation_sensor_to_tool"] = _project_rotation(np.array([
            _float(first, f"r_sensor_tool_{row}{col}")
            for row in range(3) for col in range(3)
        ]).reshape(3, 3))
        arm_names = ("tool_to_sensor_x_m", "tool_to_sensor_y_m", "tool_to_sensor_z_m")
        if all(name in first for name in arm_names):
            transforms["tool_to_sensor_m"] = np.array([_float(first, name) for name in arm_names])
    return rows, transforms


def load_static_poses(path: Path, min_samples: int) -> tuple[list[StaticPose], dict[str, np.ndarray]]:
    """Use the collector's capture_id segmentation (legacy behavior)."""

    rows, transforms = _read_csv_rows(path)
    groups: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        capture_id = int(row["capture_id"])
        if capture_id < 0:
            continue
        groups.setdefault(capture_id, []).append(row)

    poses: list[StaticPose] = []
    for capture_id in sorted(groups):
        group = groups[capture_id]
        if len(group) < min_samples:
            print(
                f"[identify] skip capture {capture_id}: {len(group)} samples "
                f"(< {min_samples})"
            )
            continue
        rotations = np.array([
            [_float(row, name) for name in ROTATION_COLUMNS] for row in group
        ]).reshape(-1, 3, 3)
        forces = np.array([
            [_float(row, name) for name in FORCE_COLUMNS] for row in group
        ])
        torques = np.array([
            [_float(row, name) for name in TORQUE_COLUMNS] for row in group
        ])
        poses.append(
            StaticPose(
                capture_id=capture_id,
                samples=len(group),
                rotation_base_sensor=_project_rotation(np.mean(rotations, axis=0)),
                force_sensor_n=np.median(forces, axis=0),
                torque_sensor_nm=np.median(torques, axis=0),
            )
        )
    return poses, transforms


def load_all_rows(path: Path, min_samples: int) -> tuple[list[StaticPose], dict[str, np.ndarray]]:
    """Treat every CSV row as one static pose.

    The collector normally marks accepted still windows with a non-negative
    ``capture_id``.  For slow, quasi-static motions the operator may prefer to
    ignore that segmentation entirely; every row then contributes directly to
    the gravity fit instead of being collapsed into one median pose per window.
    """

    rows, transforms = _read_csv_rows(path)
    if len(rows) < min_samples:
        raise ValueError(
            f"only {len(rows)} CSV rows available; at least {min_samples} are required"
        )
    poses = [
        StaticPose(
            capture_id=int(row["capture_id"]),
            samples=1,
            rotation_base_sensor=_project_rotation(np.array([
                _float(row, name) for name in ROTATION_COLUMNS
            ]).reshape(3, 3)),
            force_sensor_n=np.array([
                _float(row, name) for name in FORCE_COLUMNS
            ]),
            torque_sensor_nm=np.array([
                _float(row, name) for name in TORQUE_COLUMNS
            ]),
        )
        for row in rows
    ]
    return poses, transforms


def repair_stale_tool0_extrinsics(poses, transforms):
    """Repair captures made AFTER tool0 Z-180 change with BEFORE-change extrinsics.

    R_bs_logged = R_bt_new R_ts_old. Rebuild the physical sensor orientation as
    R_bt_new D R_ts_old; also express the lever in new tool axes. This is not
    needed for a pre-change capture, whose physical sensor orientation is valid.
    """
    required = {"rotation_sensor_to_tool", "tool_to_sensor_m"}
    if not required.issubset(transforms):
        raise ValueError("stale tool0 repair requires recorded sensor rotation and lever")
    old = transforms["rotation_sensor_to_tool"]
    flip = np.diag([-1., -1., 1.])
    repaired = [StaticPose(p.capture_id, p.samples,
                          p.rotation_base_sensor @ old.T @ flip @ old,
                          p.force_sensor_n, p.torque_sensor_nm) for p in poses]
    return repaired, {**transforms, "rotation_sensor_to_tool": flip @ old,
                      "tool_to_sensor_m": flip @ transforms["tool_to_sensor_m"]}


def _robust_lstsq(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    blocks: int,
    huber_delta: float,
    iterations: int = 20,
    solver=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively reweighted least squares with one weight per 3D pose."""

    weights = np.ones(blocks)
    solve = solver or (lambda a, b: np.linalg.lstsq(a, b, rcond=None)[0])
    solution = solve(matrix, target)
    for _ in range(iterations):
        row_weights = np.repeat(np.sqrt(weights), 3)
        weighted_matrix = matrix * row_weights[:, None]
        weighted_target = target * row_weights
        updated = solve(weighted_matrix, weighted_target)
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


def _gravity_tilt(vector: np.ndarray) -> float:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return 0.0
    return math.degrees(math.acos(float(np.clip(abs(vector[2]) / norm, 0.0, 1.0))))


def _bounded_force_solution(matrix, target, tilt_limit_deg):
    """Solve weighted force/bias LS over the two signed vertical cones.

    Eliminate bias first. If the unconstrained optimum is outside the cones,
    the constrained optimum lies on their boundary (or at zero load). Profile
    out signed load magnitude and minimize the remaining periodic azimuth.
    This uses only NumPy and refits bias for the selected gravity vector.
    """
    a, b = matrix[:, :3], matrix[:, 3:]
    centered_a = a - b @ np.linalg.lstsq(b, a, rcond=None)[0]
    centered_y = target - b @ np.linalg.lstsq(b, target, rcond=None)[0]
    free = np.linalg.lstsq(centered_a, centered_y, rcond=None)[0]
    if _gravity_tilt(free) <= tilt_limit_deg + 1e-10:
        gravity = free
    else:
        quadratic = centered_a.T @ centered_a
        linear = centered_a.T @ centered_y
        tilt = np.deg2rad(tilt_limit_deg)

        def direction(azimuth):
            azimuth = np.asarray(azimuth)
            return np.stack((np.sin(tilt) * np.cos(azimuth),
                             np.sin(tilt) * np.sin(azimuth),
                             -np.cos(tilt) * np.ones_like(azimuth)), axis=-1)

        def objective(azimuth):
            unit = direction(azimuth)
            numerator = unit @ linear
            denominator = np.einsum("...i,ij,...j->...", unit, quadratic, unit)
            return -numerator**2 / np.maximum(denominator, 1e-30)

        if tilt_limit_deg == 0:
            unit = np.array([0., 0., -1.])
        else:
            grid = np.linspace(0, 2*np.pi, 720, endpoint=False)
            scores = objective(grid)
            minima = np.flatnonzero((scores <= np.roll(scores, 1)) &
                                   (scores <= np.roll(scores, -1)))
            # An exactly flat objective needs only one representative.
            if np.ptp(scores) < 1e-14:
                minima = np.array([0])
            candidates = []
            ratio = (np.sqrt(5.) - 1.) / 2.
            for index in minima:
                left, right = grid[index] - 2*np.pi/720, grid[index] + 2*np.pi/720
                for _ in range(60):
                    x = right - ratio*(right-left)
                    y = left + ratio*(right-left)
                    if objective(x) <= objective(y):
                        right = y
                    else:
                        left = x
                candidates.append((left+right)/2)
            azimuth = min(candidates, key=objective)
            unit = direction(azimuth)
        magnitude = (unit @ linear) / (unit @ quadratic @ unit)
        gravity = magnitude * unit
    bias = np.linalg.lstsq(b, target-a@gravity, rcond=None)[0]
    return np.r_[gravity, bias]


def identify_payload(
    poses: list[StaticPose],
    *,
    gravity_m_s2: float = 9.80665,
    huber_delta: float = 2.5,
    gravity_mode: str = "bounded",
    gravity_tilt_limit_deg: float = 4.0,
) -> dict[str, object]:
    if gravity_mode not in ("bounded", "free"):
        raise ValueError("gravity_mode must be bounded or free")
    if not np.isfinite(gravity_tilt_limit_deg) or not 0 <= gravity_tilt_limit_deg < 90:
        raise ValueError("gravity_tilt_limit_deg must be finite and in [0, 90)")
    if not np.isfinite(gravity_m_s2) or gravity_m_s2 <= 0:
        raise ValueError("gravity_m_s2 must be finite and positive")
    if not np.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError("huber_delta must be finite and positive")
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
    free_solution, free_weights = _robust_lstsq(
        force_matrix, forces.reshape(-1), blocks=len(poses), huber_delta=huber_delta
    )
    if gravity_mode == "free":
        force_solution, force_weights = free_solution, free_weights
    else:
        force_solution, force_weights = _robust_lstsq(
            force_matrix, forces.reshape(-1), blocks=len(poses), huber_delta=huber_delta,
            solver=lambda a, b: _bounded_force_solution(a, b, gravity_tilt_limit_deg),
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

    # Report physical down separately from the sensor's signed gravity load.
    gravity_unit_base = gravity_vector_base / gravity_norm
    down_unit_base = (
        gravity_unit_base if gravity_unit_base[2] <= 0.0 else -gravity_unit_base
    )
    tilt_deg = math.degrees(math.acos(float(np.clip(-down_unit_base[2], 0.0, 1.0))))
    tilt_azimuth_deg = math.degrees(
        math.atan2(float(down_unit_base[1]), float(down_unit_base[0]))
    )
    force_condition = float(np.linalg.cond(force_matrix))
    torque_condition = float(np.linalg.cond(torque_matrix))

    return {
        "model": "static_raw_sensor_wrench",
        "gravity_direction_model": gravity_mode,
        "gravity_tilt_limit_deg": gravity_tilt_limit_deg if gravity_mode == "bounded" else None,
        "unconstrained_gravity_tilt_deg": _gravity_tilt(free_solution[:3]),
        "pose_count": len(poses),
        "sample_count": sum(pose.samples for pose in poses),
        "gravity_m_s2": gravity_m_s2,
        "mass_kg": gravity_norm / gravity_m_s2,
        "signed_gravity_force_base_n": gravity_vector_base.tolist(),
        "gravity_direction_base_unit": gravity_unit_base.tolist(),
        "gravity_down_base_unit": down_unit_base.tolist(),
        "gravity_tilt_deg": tilt_deg,
        "gravity_tilt_azimuth_deg": tilt_azimuth_deg,
        # Legacy name for the same quantity, kept for existing JSON consumers.
        "vertical_alignment_error_deg": tilt_deg,
        "center_of_mass_sensor_m": center_of_mass_sensor.tolist(),
        "center_of_mass_sensor_mm": (1000.0 * center_of_mass_sensor).tolist(),
        "force_bias_sensor_n": force_bias.tolist(),
        "torque_bias_sensor_nm": torque_bias.tolist(),
        "fit": {
            "force_rms_n": float(np.sqrt(np.mean(force_residual ** 2))),
            "unconstrained_force_rms_n": float(np.sqrt(np.mean((forces.reshape(-1)-force_matrix@free_solution)**2))),
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
                        help=("minimum accepted rows. In default all-row mode this is the "
                              "total row count; with --static-poses it is the minimum rows "
                              "per capture window (default: 50)"))
    parser.add_argument("--static-poses", action="store_true",
                        help=("use the collector capture_id segmentation and median each "
                              "accepted static window. By default every CSV row is used "
                              "directly as one quasi-static sample."))
    parser.add_argument("--gravity", type=float, default=9.80665)
    parser.add_argument("--gravity-mode", choices=("bounded", "free"), default="bounded",
                        help="bounded (default): constrain physical down near base -Z; free: diagnostic 3D fit")
    parser.add_argument("--gravity-tilt-limit-deg", type=float, default=4.0,
                        help="hard inclination bound for bounded fit (default: 4 deg; 0 fixes vertical)")
    parser.add_argument("--repair-stale-tool0-extrinsics", action="store_true",
                        help="repair CSV captured after tool0 Z-180 change while using old sensor extrinsics")
    parser.add_argument("--huber-delta", type=float, default=2.5)
    parser.add_argument("--max-com-m", type=float, default=0.5,
                        help="warn if identified sensor-to-CoM distance exceeds this")
    parser.add_argument("--max-tilt-deg", type=float, default=30.0,
                        help=("warn if the fitted gravity direction is more than this far from "
                              "base -Z; a tilted chassis or an inclined floor is a legitimate "
                              "reason; this is a warning threshold, not the fit constraint"))
    return parser.parse_args()


def build_warnings(
    result: dict[str, object],
    *,
    max_tilt_deg: float,
    max_com_m: float,
) -> list[str]:
    """Human-readable sanity notes for one identification result.

    The free-fit diagnostic is retained so a tight bound cannot hide a bad
    frame/calibration or an installation outside the assumed inclination.
    """

    warnings: list[str] = []
    limit = result.get("gravity_tilt_limit_deg")
    free_tilt = result.get("unconstrained_gravity_tilt_deg")
    if limit is not None and free_tilt is not None and float(free_tilt) > float(limit) + .1:
        warnings.append(
            f"gravity constrained to {float(limit):.1f} deg, but unconstrained fit preferred "
            f"{float(free_tilt):.1f} deg; the reported constrained tilt is a model assumption; "
            "check sensor extrinsics, raw mode, static data and actual installation"
        )
    tilt_deg = float(result["gravity_tilt_deg"])
    if tilt_deg > max_tilt_deg:
        warnings.append(
            f"fitted gravity is {tilt_deg:.1f} deg away from base -Z "
            f"(down-slope azimuth {float(result['gravity_tilt_azimuth_deg']):.1f} deg in base XY); "
            "a tilted chassis or an inclined floor explains this and it is fitted, not assumed - "
            "otherwise check FK, the sensor-to-tool rotation, the force sign/frame and the "
            "controller compensation mode"
        )
    if np.linalg.norm(np.asarray(result["center_of_mass_sensor_m"], dtype=float)) > max_com_m:
        warnings.append("identified sensor-to-CoM distance is physically implausible")
    fit = result["fit"]
    if fit["force_matrix_condition"] > 100.0 or fit["torque_matrix_condition"] > 100.0:
        warnings.append("identification is poorly conditioned; collect more diverse orientations")
    return warnings


def main() -> None:
    args = _parse_args()
    if args.min_samples <= 0 or args.gravity <= 0 or args.huber_delta <= 0:
        raise SystemExit("--min-samples, --gravity and --huber-delta must be positive")
    if args.max_tilt_deg <= 0 or args.max_com_m <= 0:
        raise SystemExit("--max-tilt-deg and --max-com-m must be positive")
    csv_path = args.csv.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else csv_path.with_name(f"{csv_path.stem}_identified.json")
    )
    try:
        if args.static_poses:
            poses, transforms = load_static_poses(csv_path, args.min_samples)
            pose_source = "static_capture_windows"
        else:
            poses, transforms = load_all_rows(csv_path, args.min_samples)
            pose_source = "all_csv_rows"
        if args.repair_stale_tool0_extrinsics:
            poses, transforms = repair_stale_tool0_extrinsics(poses, transforms)
        result = identify_payload(
            poses, gravity_m_s2=args.gravity, huber_delta=args.huber_delta,
            gravity_mode=args.gravity_mode, gravity_tilt_limit_deg=args.gravity_tilt_limit_deg,
        )
        result["pose_source"] = pose_source
        result["stale_tool0_extrinsics_repaired"] = args.repair_stale_tool0_extrinsics
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

    warnings = build_warnings(
        result, max_tilt_deg=args.max_tilt_deg, max_com_m=args.max_com_m
    )
    result["warnings"] = warnings

    if output.exists() and not args.overwrite:
        raise SystemExit(
            f"[identify] output JSON already exists: {output} "
            "(choose another --output or pass --overwrite)"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[identify] source: {result['pose_source']}  "
        f"poses: {result['pose_count']} ({result['sample_count']} samples)"
    )
    print(
        f"[identify] mass: {result['mass_kg']:.6f} kg (|h| = "
        f"{result['mass_kg'] * result['gravity_m_s2']:.4f} N)"
    )
    print(
        "[identify] gravity force in base frame: "
        + np.array2string(np.asarray(result["signed_gravity_force_base_n"]), precision=4)
        + " N"
    )
    print(
        f"[identify] gravity direction: tilt {result['gravity_tilt_deg']:.2f} deg from base -Z "
        f"(down-slope azimuth {result['gravity_tilt_azimuth_deg']:.2f} deg in base XY); "
        + (f"bounded model <= {args.gravity_tilt_limit_deg:.2f} deg, not an inclination measurement"
           if args.gravity_mode == "bounded" else "unrestricted diagnostic fit")
    )
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
        f"[identify] residual RMS: force={result['fit']['force_rms_n']:.5f} N, "
        f"torque={result['fit']['torque_rms_nm']:.6f} Nm"
    )
    for warning in warnings:
        print(f"[identify] WARNING: {warning}")
    print(f"[identify] JSON: {output}")


if __name__ == "__main__":
    main()
