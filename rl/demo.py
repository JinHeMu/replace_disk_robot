#!/usr/bin/env python3
"""Command-line demo for the residual insertion environment.

Examples
--------
Run the classical baseline and a small scripted residual::

    python -m rl --policies zero scripted --episodes 1 --seed 7

Write a JSON report under ``rl/artifacts``::

    python -m rl --policies zero scripted --output rl/artifacts/demo_report.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

from .config import load_rl_config
from .env import ResidualInsertionEnv
from .policy import ConstantPolicy, ResidualPolicy, ScriptedPolicy, ZeroPolicy
from .runner import evaluate_policy


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["zero", "scripted"],
        choices=["zero", "scripted", "constant"],
        help="Policies to evaluate.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--record-history", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "demo_report.json",
    )
    return parser


def _make_policy(name: str) -> ResidualPolicy:
    if name == "zero":
        return ZeroPolicy()
    if name == "scripted":
        return ScriptedPolicy()
    if name == "constant":
        return ConstantPolicy([0.0, 0.0, 0.0, 0.0, 0.0])
    raise ValueError(f"unknown policy {name!r}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    config = load_rl_config(args.config)
    env = ResidualInsertionEnv(config, seed=args.seed, randomize=args.randomize)
    seeds = [args.seed + index for index in range(args.episodes)]
    report: dict[str, object] = {
        "config": str(args.config) if args.config else "built-in-defaults",
        "seeds": seeds,
        "randomize": bool(args.randomize),
        "observation_schema": env.observation_schema,
        "policies": {},
    }
    for name in args.policies:
        policy = _make_policy(name)
        summary = evaluate_policy(
            env,
            policy,
            seeds=seeds,
            policy_name=name,
            record_history=args.record_history,
        )
        report["policies"][name] = summary  # type: ignore[index]
        result = summary["results"][0]
        print(
            f"[{name}] success={result['success']} depth={result['final_depth_m']:.4f} m "
            f"time={result['episode_time_s']:.2f} s peak_force={result['peak_force_n']:.2f} N "
            f"fault={result['fault_reason'] or 'none'}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
