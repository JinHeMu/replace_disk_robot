#!/usr/bin/env python3
"""Position-driven contact/sensor validation; does not implement force control."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from replace_disk_robot.adapters.mujoco.insertion_validation import run_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    results = [run_probe(kind, dt) for kind in ('axial', 'side') for dt in (.001, .0005)]
    import numpy as np
    stable = all(np.allclose(results[i]['mean_tared_wrench'], results[i+1]['mean_tared_wrench'],
                            rtol=.1, atol=.02) for i in (0, 2))
    report = dict(passed=all(r['passed'] for r in results) and stable,
                  timestep_comparison_passed=stable, tests=results,
                  validation_scope='Scene and sensor chain only; no closed-loop insertion')
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+'\n')
    print(text)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
