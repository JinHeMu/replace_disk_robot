#!/usr/bin/env python3
"""Evaluate baseline, scripted or exported residual policies on fixed seeds.

Examples::

    python -m rl.evaluate --policy zero --episodes 5
    python -m rl.evaluate --policy linear --model rl/artifacts/cem_policy.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_rl_config
from .env import ResidualInsertionEnv
from .policy import LinearPolicy, ResidualPolicy, ScriptedPolicy, ZeroPolicy
from .runner import evaluate_policy


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=["zero", "scripted", "linear"],
        default="zero",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--record-history", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "evaluation.json",
    )
    return parser


def _load_policy(name: str, model: Path | None) -> ResidualPolicy:
    if name == "zero":
        return ZeroPolicy()
    if name == "scripted":
        return ScriptedPolicy()
    if model is None:
        raise SystemExit("--model is required when --policy linear")
    return LinearPolicy.load(model)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    config = load_rl_config(args.config)
    env = ResidualInsertionEnv(config, seed=args.seed, randomize=args.randomize)
    seeds = [args.seed + index for index in range(args.episodes)]
    policy = _load_policy(args.policy, args.model)
    summary = evaluate_policy(
        env,
        policy,
        seeds=seeds,
        policy_name=args.policy,
        record_history=args.record_history,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
