"""Offline regression of pose tracking with delayed actuator feedback.

The synthetic actuator is a delay, not an identified JAKA/contact model.
Historical corrected references can be replayed with actual JAKA kinematics;
this isolates the Servo and does not predict new environmental forces.
No SDK, controller, joystick, viewer or hardware session is created.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np

from replace_disk_robot.control import CartesianServo, ServoConfig
from replace_disk_robot.core import JointState, Pose
from replace_disk_robot.core.rotation import quaternion_from_rotation_vector, rotation_matrix, rotation_vector_from_matrix


class CartesianTestKinematics:
    """Exact linear translation / single-axis rotation used by sine tests."""
    joint_names = ("x", "y", "z", "rx", "ry", "rz")
    dof = 6
    joint_limits_rad = (-np.ones(6), np.ones(6))

    def forward(self, state):
        return Pose("test_base", state.position_rad[:3],
                    quaternion_from_rotation_vector(state.position_rad[3:]))

    def jacobian(self, state):
        return np.eye(6)


class LegacyMeasuredErrorServo(CartesianServo):
    """Reproduce ONLY the old measured-error integration for comparison."""
    pose_tracking_mode = "legacy_measured_error_integration"

    def update(self, measured, dt_s, now_s):
        self._validation_measured = measured
        return super().update(measured, dt_s, now_s)

    def _pose_tracking_twist(self, reference, commanded, dt_s):
        actual = self.kinematics.forward(self._validation_measured)
        linear = reference.position_m - actual.position_m
        angular = rotation_vector_from_matrix(
            rotation_matrix(reference.quaternion_wxyz)
            @ rotation_matrix(actual.quaternion_wxyz).T)
        if np.linalg.norm(linear) < self.config.pose_deadband_m:
            linear = np.zeros(3)
        if np.linalg.norm(angular) < self.config.pose_deadband_rad:
            angular = np.zeros(3)
        return (self._limit(self.config.pose_linear_gain_s_inv*linear, self.config.pose_linear_speed_m_s),
                self._limit(self.config.pose_angular_gain_s_inv*angular, self.config.pose_angular_speed_rad_s))


class DelayedActuator:
    def __init__(self, initial, delay_s, dt_s):
        self.actual = initial
        self._queue = deque(initial for _ in range(round(delay_s/dt_s)))

    def advance(self, target):
        self._queue.append(target)
        self.actual = self._queue.popleft()
        return self.actual


def harmonic_metrics(time, reference, measured, commanded, frequency_hz, *, after_s):
    selected = np.asarray(time) >= after_s
    t = np.asarray(time)[selected]
    basis = np.c_[np.ones(len(t)), np.sin(2*np.pi*frequency_hz*t), np.cos(2*np.pi*frequency_hz*t)]
    signal = np.c_[reference, measured, commanded][selected]
    fit = np.linalg.lstsq(basis, signal, rcond=None)[0]
    phasor = fit[1] + 1j*fit[2]
    transfer = phasor[1:] / phasor[0]
    return {"actual_amplitude_ratio": float(abs(transfer[0])),
            "command_amplitude_ratio": float(abs(transfer[1])),
            "actual_phase_deg": float(np.rad2deg(np.angle(transfer[0]))),
            "command_phase_deg": float(np.rad2deg(np.angle(transfer[1])))}


def run_harmonic(servo_type=CartesianServo, *, delay_s=.144, frequency_hz=1., axis=0,
                 duration_s=12., dt_s=.008):
    kin = CartesianTestKinematics()
    servo = servo_type(kin, kin.joint_limits_rad)
    zero = JointState(kin.joint_names, np.zeros(6))
    servo.reset(zero)
    actuator = DelayedActuator(zero, delay_s, dt_s)
    amplitude = .002 if axis < 3 else np.deg2rad(.2)
    traces = []
    for t in np.arange(0., duration_s, dt_s):
        value = np.zeros(6)
        value[axis] = amplitude*np.sin(2*np.pi*frequency_hz*t)
        reference = kin.forward(JointState(kin.joint_names, value))
        servo.submit_pose(reference, float(t))
        target = servo.update(actuator.actual, dt_s, float(t))
        actual = actuator.advance(target)
        traces.append([t, value[axis], actual.position_rad[axis], target.position_rad[axis]])
        if servo.fault:
            raise RuntimeError(f"harmonic validation unexpectedly stopped: {servo.fault}")
    trace = np.asarray(traces)
    metrics = harmonic_metrics(*trace.T, frequency_hz, after_s=duration_s/2)
    return metrics, trace


def replay_session(directory, output):
    """Use the SAME historical reference for old and new Servo, not live force replay."""
    from replace_disk_robot.kinematics.jaka import JakaKinematics
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    rows = [r for r in map(json.loads, (directory / "samples.jsonl").open())
            if r.get("admittance_updated") and not r.get("fault")]
    if not rows:
        raise ValueError(f"no valid pose reference in {directory}")
    kin = JakaKinematics()
    config = ServoConfig(**metadata["servo_config"])
    replay = {}
    traces = {}
    for servo_type, label in ((LegacyMeasuredErrorServo, "before"), (CartesianServo, "after")):
        servo = servo_type(kin, kin.joint_limits_rad, config)
        state = JointState(kin.joint_names, rows[0]["measured_q_rad"])
        servo.reset(state)
        actuator = DelayedActuator(state, .144, .008)
        positions, errors, command_errors = [], [], []
        for row in rows:
            t, dt = row["elapsed_s"], row["dt_s"]
            reference = Pose(**row["corrected_pose"])
            servo.submit_pose(reference, t)
            target = servo.update(actuator.actual, dt, t)
            actual = actuator.advance(target)
            actual_pose, command_pose = kin.forward(actual), kin.forward(target)
            positions.append([t, *reference.position_m, *actual_pose.position_m, *command_pose.position_m])
            errors.append(np.linalg.norm(reference.position_m-actual_pose.position_m))
            command_errors.append(np.linalg.norm(reference.position_m-command_pose.position_m))
            if servo.fault:
                break
        traces[label] = np.asarray(positions)
        replay[label] = {"rows": len(positions), "fault": servo.fault,
                         "reference_to_actual_rms_mm": float(np.sqrt(np.mean(np.square(errors)))*1000),
                         "reference_to_command_rms_mm": float(np.sqrt(np.mean(np.square(command_errors)))*1000)}
    np.savez_compressed(output / f"{directory.name}_reference_replay.npz", **traces)
    return {"session": str(directory), "scope": "same recorded references; fixed 18-sample actuator delay; no contact prediction", **replay}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, nargs="*", default=[])
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"scope": "offline delayed-actuator regression; not a hardware/contact acceptance",
              "harmonic_cases": [], "reference_replays": []}
    example = {}
    for delay in (0., .072, .144, .200):
        for axis in (0, 5):
            for frequency in (.5, 1., 2.):
                case = {"delay_s": delay, "axis": axis, "frequency_hz": frequency}
                for kind, name in ((LegacyMeasuredErrorServo, "before"), (CartesianServo, "after")):
                    metrics, trace = run_harmonic(kind, delay_s=delay, axis=axis, frequency_hz=frequency)
                    case[name] = metrics
                    if delay == .144 and axis == 0 and frequency == 1.:
                        example[name] = trace
                report["harmonic_cases"].append(case)
    for session in args.sessions:
        report["reference_replays"].append(replay_session(session, args.output_dir))
    report["synthetic_no_amplification_passed"] = all(
        case["after"]["actual_amplitude_ratio"] <= 1.02
        and case["after"]["command_amplitude_ratio"] <= 1.02
        for case in report["harmonic_cases"])
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, sharey=True,
                                 constrained_layout=True)
        for ax, label in zip(axes, ("before", "after")):
            trace = example[label]
            selected = (trace[:, 0] >= 8.) & (trace[:, 0] <= 12.)
            for j, name in enumerate(("reference", "delayed actual", "command"), 1):
                ax.plot(trace[selected, 0], trace[selected, j]*1000, label=name)
            ax.set_title(f"{label}: 1 Hz, 2 mm amplitude, 144 ms synthetic actuator delay")
            ax.set_ylabel("mm")
            ax.grid(alpha=.3)
            ax.legend(loc="upper right")
        axes[-1].set_xlabel("s")
        fig.savefig(args.output_dir / "servo_delay_before_after.png", dpi=160)
        plt.close(fig)
    print(json.dumps({"passed": report["synthetic_no_amplification_passed"],
                      "harmonic_cases": len(report["harmonic_cases"]),
                      "reference_replays": report["reference_replays"]}, indent=2))
    if not report["synthetic_no_amplification_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
