"""Offline report regressions for missing sensors, clamp and frame semantics."""
import csv
import json

import numpy as np
import pytest

from replace_disk_robot.applications.jaka_keyboard.analysis import (
    control_analysis, main, write_readable_report,
)
from replace_disk_robot.core.rotation import quaternion_from_rotation_vector


def pose(position, angle=0, frame="base"):
    return {"frame_id": frame, "position_m": position,
            "quaternion_wxyz": quaternion_from_rotation_vector([0, 0, angle]).tolist()}


def sample(sequence=1):
    # Measured TCP axes are rotated +90deg: a tool-X offset is base-Y.
    nominal = pose([1., 2., 3.], np.pi/2)
    return {"sequence": sequence, "elapsed_s": (sequence-1)*.008, "dt_s": .008,
            "admittance_updated": True, "command_sent": True,
            "raw_sensor_wrench": [10., 20., 30., 1., 2., 3.],
            "raw_sensor_frame_id": "sensor",
            "wrench_stages": {"compensated_sensor": [1., 2., 3., .1, .2, .3],
                              "compensated_sensor_frame": "sensor", "frame_id": "tool",
                              "external_tcp_unfiltered": [3., 2., 1., .3, .2, .1],
                              "processed": [2., 1., 0., .2, .1, 0.], "gravity_compensation": True},
            "admittance_input_wrench": [2., 0., 0., .2, 0., 0.],
            "admittance_integrated": {"frame_id": "tool", "offset": [.02, 0, 0, 0, 0, .1],
                                      "velocity": [0]*6, "nominal_pose": nominal},
            "corrected_pose": pose([1., 2.01, 3.], np.pi/2 + .05),
            "measured_tcp_pose": pose([1., 2.004, 3.], np.pi/2)}


def readable_report(sections, path):
    write_readable_report({"rows": 1, "recording_complete": True,
                           "discrete_equation_matches": True, **sections}, path)
    return path.read_text()


def test_clamp_and_actual_motion_use_common_base_axes():
    first = sample()
    second = sample(2)
    # Keyboard motion advances nominal by 10mm in base-X.
    second["admittance_integrated"]["nominal_pose"]["position_m"][0] += .01
    second["corrected_pose"]["position_m"][0] += .01
    second["measured_tcp_pose"]["position_m"][0] += .008
    sections, (_, rows), data = control_analysis([first, second])
    assert sections["admittance_control"]["pose_frame"] == "base"
    np.testing.assert_allclose(data["offset"][0, :3], [20, 0, 0])
    np.testing.assert_allclose(data["limited_compliance_base"][0, :3], [0, 10, 0], atol=1e-10)
    np.testing.assert_allclose(data["actual_relative_nominal_base"][0, :3], [0, 4, 0], atol=1e-10)
    np.testing.assert_allclose(data["actual_displacement_base"][0], np.zeros(6))
    np.testing.assert_allclose(data["actual_displacement_base"][1, :3], [8, 0, 0], atol=1e-10)
    np.testing.assert_allclose(data["actual_relative_nominal_base"][1, :3], [-2, 4, 0], atol=1e-10)
    assert rows[0]["limited_compliance_base_y_mm"] == pytest.approx(10)
    assert rows[0]["input_Fy_N"] == 0
    assert rows[0]["processed_Fy_N"] == 1
    assert sections["force_control"]["table"][0]["compensated_minus_raw"]["mean"] == -9


def test_rotation_delta_handles_wrap_without_euler_subtraction():
    a, b = sample(), sample(2)
    a["measured_tcp_pose"] = pose([0, 0, 0], np.deg2rad(179))
    b["measured_tcp_pose"] = pose([0, 0, 0], np.deg2rad(-179))
    _, _, data = control_analysis([a, b])
    assert data["actual_displacement_base"][1, 5] == pytest.approx(2)


@pytest.mark.parametrize("raw", [None, [None]*6, [1, 2]])
def test_missing_raw_is_compensated_only(raw, tmp_path):
    row = sample()
    row["raw_sensor_wrench"] = raw
    row.pop("raw_sensor_frame_id")
    sections, (force_rows, _), _ = control_analysis([row])
    assert sections["force_control"]["raw_available"] is False
    assert sections["force_control"]["table"][0]["compensated"]["mean"] == 1
    assert not any(key.startswith("raw_") and key != "raw_frame" for key in force_rows[0])
    text = readable_report(sections, tmp_path/"report.md")
    assert "未记录原始数据" in text
    assert "原始均值" not in text


def test_partial_missing_samples_are_not_zero_or_interpolated():
    a, b = sample(), sample(2)
    b["raw_sensor_wrench"][0] = None
    b["wrench_stages"]["compensated_sensor"][1] = None
    b["admittance_updated"] = False  # A stale snapshot must not appear as an update.
    sections, (rows, adm_rows), data = control_analysis([a, b])
    assert sections["force_control"]["table"][0]["raw"]["count"] == 1
    assert sections["force_control"]["table"][1]["compensated"]["mean"] == 2
    assert rows[1]["raw_Fx_N"] is None
    assert rows[1]["compensated_Fy_N"] is None
    assert np.isnan(data["offset"][1]).all()
    assert adm_rows[1]["offset_x_mm"] is None
    json.dumps(sections, allow_nan=False)


def test_fallback_processed_wrench_does_not_compare_sensor_axes():
    row = sample()
    row["wrench_stages"].pop("compensated_sensor")
    sections, (rows, _), data = control_analysis([row])
    assert sections["force_control"]["compensated_source"] == "processed"
    assert sections["force_control"]["paired_comparison_available"] is False
    assert sections["force_control"]["table"][0]["compensated_frame"] == "tool"
    assert sections["force_control"]["table"][0]["compensated_minus_raw"]["count"] == 0
    assert data["comparable"] is False
    assert not any(key.startswith("compensated_minus_raw") for key in rows[0])


@pytest.mark.parametrize("signal", ["pose", "input", "varying"])
def test_coordinate_mismatch_is_rejected(signal):
    a, b = sample(), sample(2)
    if signal == "pose":
        b["corrected_pose"]["frame_id"] = "other"
    elif signal == "input":
        b["admittance_integrated"]["frame_id"] = "base"
    else:
        b["raw_sensor_frame_id"] = "other_sensor"
    with pytest.raises(ValueError, match="frame|axes"):
        control_analysis([a, b])


def test_no_admittance_and_empty_log_can_be_reported(tmp_path):
    row = sample()
    for key in ("admittance_integrated", "admittance_input_wrench", "corrected_pose"):
        row.pop(key)
    row["admittance_updated"] = False
    for rows in ([row], []):
        sections, _, _ = control_analysis(rows)
        text = readable_report(sections, tmp_path/"report.md")
        assert "2.1 外力" in text and "2.2 机械臂" in text
        assert sections["admittance_control"]["table"][0]["offset"]["count"] == 0


def test_cli_writes_tables_and_three_plots(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import sys
    from replace_disk_robot.applications.jaka_keyboard import analysis
    rows = [sample(), sample(2)]
    sections, _, _ = control_analysis(rows)
    report = {"rows": 2, "recording_complete": True, "discrete_equation_matches": True, **sections}
    monkeypatch.setattr(analysis, "analyze_session", lambda directory: (report, [{"sequence": 1}], rows))
    monkeypatch.setattr(sys, "argv", ["analysis", str(tmp_path), "--plots"])
    main()
    output = tmp_path/"analysis"
    assert (output/"report.md").is_file()
    for filename in ("force_control_overview.png", "admittance_control_overview.png", "admittance_tracking_overview.png"):
        assert (output/filename).stat().st_size > 1000
    with (output/"admittance_control_samples.csv").open(encoding="utf-8-sig") as file:
        exported = list(csv.DictReader(file))
    assert exported[1]["actual_displacement_base_y_mm"] == '0.0'
    assert exported[0]["offset_frame"] == 'tool'
    assert exported[0]["pose_frame"] == 'base'
