#!/usr/bin/env python3
"""Runnable derivative-free trainer for the first residual-RL demo.

The production algorithm choice remains SAC (see ``config.SACConfig``).  This
entry point deliberately uses a small NumPy cross-entropy method over a linear
policy so the whole pipeline can be exercised without installing PyTorch.  The
policy interface and environment contract are identical to a future SAC actor;
only the optimiser differs.

Example::

    python -m rl.train --generations 3 --population 6 --episodes-per-candidate 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np

from .config import load_rl_config
from .env import ResidualInsertionEnv
from .features import FEATURE_NAMES, observation_schema
from .policy import LinearPolicy
from .residual import ACTION_NAMES
from .runner import run_episode


@dataclass
class CandidateResult:
    parameters: np.ndarray
    fitness: float
    success_rate: float
    mean_peak_force_n: float


def _unpack(parameters: np.ndarray) -> LinearPolicy:
    size = len(ACTION_NAMES) * len(FEATURE_NAMES)
    weights = parameters[:size].reshape(len(ACTION_NAMES), len(FEATURE_NAMES))
    bias = parameters[size : size + len(ACTION_NAMES)]
    return LinearPolicy(weights, bias)


def _pack(policy: LinearPolicy) -> np.ndarray:
    return np.concatenate((policy.weights.reshape(-1), policy.bias))


def _evaluate_candidate(
    env: ResidualInsertionEnv,
    parameters: np.ndarray,
    seeds: list[int],
) -> CandidateResult:
    policy = _unpack(parameters)
    results = [run_episode(env, policy, seed=seed) for seed in seeds]
    return CandidateResult(
        parameters=parameters.copy(),
        fitness=float(np.mean([result.total_reward for result in results])),
        success_rate=float(np.mean([result.success for result in results])),
        mean_peak_force_n=float(np.mean([result.peak_force_n for result in results])),
    )


def train_cem(
    *,
    generations: int,
    population: int,
    elite_fraction: float,
    seeds: list[int],
    config_path: str | Path | None,
    randomize: bool,
    output: Path,
) -> dict[str, object]:
    if generations <= 0 or population <= 1:
        raise ValueError("generations must be positive and population must be greater than one")
    if not 0.0 < elite_fraction < 1.0:
        raise ValueError("elite_fraction must be in (0, 1)")
    elite_count = max(2, int(round(population * elite_fraction)))

    config = load_rl_config(config_path)
    env = ResidualInsertionEnv(config, seed=seeds[0], randomize=randomize)
    parameter_size = len(ACTION_NAMES) * len(FEATURE_NAMES) + len(ACTION_NAMES)
    rng = np.random.default_rng(config.sac.seed)
    mean = np.zeros(parameter_size, dtype=float)
    std = np.full(parameter_size, 0.35, dtype=float)
    history: list[dict[str, object]] = []
    best: CandidateResult | None = None
    wall_start = time.perf_counter()

    for generation in range(generations):
        samples = rng.normal(mean, std, size=(population, parameter_size))
        candidates = [
            _evaluate_candidate(env, sample, seeds) for sample in samples
        ]
        ranked = sorted(candidates, key=lambda item: (item.success_rate, item.fitness), reverse=True)
        elite = ranked[:elite_count]
        elite_params = np.stack([item.parameters for item in elite], axis=0)
        mean = elite_params.mean(axis=0)
        std = elite_params.std(axis=0) + 0.05
        if best is None or (ranked[0].success_rate, ranked[0].fitness) > (
            best.success_rate,
            best.fitness,
        ):
            best = ranked[0]
        generation_record = {
            "generation": generation,
            "best_fitness": ranked[0].fitness,
            "best_success_rate": ranked[0].success_rate,
            "elite_mean_fitness": float(np.mean([item.fitness for item in elite])),
            "elite_mean_peak_force_n": float(np.mean([item.mean_peak_force_n for item in elite])),
        }
        history.append(generation_record)
        print(json.dumps(generation_record, ensure_ascii=False))

    assert best is not None
    policy = _unpack(best.parameters)
    output.parent.mkdir(parents=True, exist_ok=True)
    policy.save(output)
    manifest = {
        "algorithm": "cem-linear-policy-demo",
        "note": "SAC entry point remains optional; this is a dependency-free demo trainer.",
        "generations": generations,
        "population": population,
        "elite_fraction": elite_fraction,
        "seeds": seeds,
        "randomize": randomize,
        "config": str(config_path) if config_path else "built-in-defaults",
        "observation_schema": observation_schema(),
        "action_names": list(ACTION_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "best_fitness": best.fitness,
        "best_success_rate": best.success_rate,
        "best_mean_peak_force_n": best.mean_peak_force_n,
        "wall_time_s": time.perf_counter() - wall_start,
        "history": history,
        "policy_path": str(output),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--elite-fraction", type=float, default=0.34)
    parser.add_argument("--episodes-per-candidate", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "cem_policy.npz",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    seeds = [args.seed + index for index in range(args.episodes_per_candidate)]
    manifest = train_cem(
        generations=args.generations,
        population=args.population,
        elite_fraction=args.elite_fraction,
        seeds=seeds,
        config_path=args.config,
        randomize=args.randomize,
        output=args.output,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
