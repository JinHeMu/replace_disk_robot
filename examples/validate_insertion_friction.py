#!/usr/bin/env python3
"""Bounded position-driven friction QA; this is not closed-loop force control."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from replace_disk_robot.adapters.mujoco.insertion_validation import run_friction_probe


def validate():
    # Calibration fixture bounds only; production Servo's 10 N / 1 Nm guard
    # stays unchanged. Entry/reversal transients can exceed sliding resistance.
    limits = dict(force_limit_n=15., torque_limit_nm=2.)
    normal = run_friction_probe(.0005, .6, **limits)
    half = run_friction_probe(.00025, .6, **limits)
    zero = run_friction_probe(.0005, 0., **limits)
    stable = all(abs(normal[k]['mean_sensor_axial_n']-half[k]['mean_sensor_axial_n']) < .3
                 for k in ('insert', 'withdraw'))
    contrast = (abs(zero['insert']['mean_contact_axial_n']) < .05 and
                normal['insert']['mean_contact_axial_n'] > .3)
    target_passed = all(abs(abs(r[k]['mean_sensor_axial_n'])-7.5) <= .5
                        for r in (normal, half) for k in ('insert', 'withdraw'))
    return dict(passed=all(r['passed'] for r in (normal, half, zero)) and stable and contrast and target_passed,
                target_sliding_force_n=7.5, sliding_force_tolerance_n=.5,
                target_force_passed=target_passed,
                fixture_force_limit_n=15., fixture_torque_limit_nm=2.,
                timestep_comparison_passed=stable, friction_comparison_passed=contrast,
                validation_scope='Position-driven contact friction QA, not closed-loop insertion',
                tests=[normal, half, zero])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('simulation/mujoco/reports/friction_validation.json'))
    args = parser.parse_args()
    report = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
