from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import shlex
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np

from cognitive_tools import EcoEnv
from cognitive_tools.env import reward_definition, reward_identity
from cognitive_tools.model import HIGH_EXTRACT, LOW_EXTRACT
from cognitive_tools.qlearning import STATE_NAMES, QLearningPolicy, resource_state
from cognitive_tools.scenarios import SCENARIOS, build_environment_maps
from cognitive_tools.social import (
    REWIRING_MODES,
    SOCIAL_NETWORK_MODES,
    SOCIAL_STATE_NAMES,
    copy_sources,
    init_barabasi_albert_attention,
    init_random_attention,
    joint_state,
    network_edges,
    network_turnover,
    observed_low_fraction,
    observer_social_diagnostics,
    prediction_errors,
    rewire_epoch,
    social_bin,
    social_metrics,
    social_observations,
    update_forecasts,
    visibility_counts,
)

RESULTS_ROOT = Path("results") / "q_learning_baseline" / "experiments"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

RUNTIME_PACKAGES = ("numpy", "mesa", "pettingzoo", "gymnasium", "matplotlib", "networkx", "pytest")

SOCIAL_MODES = ("none", "fixed")

NETWORK_EVAL_MODES = ("frozen", "adaptive")


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------


def _git_output(*arguments: str) -> str | None:
    """Run a read-only Git command from the repository root."""
    try:
        completed = subprocess.run(
            ["git", *arguments], cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _package_version(package_name: str) -> str | None:
    """Return an installed package version when available."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_run_metadata() -> dict[str, object]:
    """Return reproducibility metadata without consuming simulation RNGs."""
    status = _git_output("status", "--porcelain")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("rev-parse", "--abbrev-ref", "HEAD"),
        "git_worktree_dirty": None if status is None else bool(status),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": {
            package_name: _package_version(package_name) for package_name in RUNTIME_PACKAGES
        },
        "command": shlex.join(sys.argv),
    }


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def load_matched_rewire_schedule(
    path_string: str,
) -> dict[tuple[str, int, int, int], dict[str, object]]:
    """
    Load an adaptive rewiring schedule for a matched-random R0 run.

    Rows are keyed by:
        (scenario, population, replicate, time)
    """
    path = Path(path_string).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Matched rewiring schedule does not exist: {path}")

    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "scenario",
            "population",
            "replicate",
            "time",
            "rewiring",
            "rewire_theta",
            "rewire_every",
            "base_seed",
            "social_network",
            "social_k",
            "training_steps",
            "successful_rewires",
        }
        fieldnames = set(reader.fieldnames or [])
        missing = required - fieldnames
        if missing:
            raise ValueError(
                "Matched rewiring schedule is missing columns: " + ", ".join(sorted(missing))
            )

        schedule: dict[tuple[str, int, int, int], dict[str, object]] = {}
        for row in reader:
            key = (
                str(row["scenario"]),
                int(row["population"]),
                int(row["replicate"]),
                int(row["time"]),
            )
            if key in schedule:
                raise ValueError(f"Duplicate matched rewiring schedule row for {key}.")

            successful_rewires = int(row["successful_rewires"])
            if successful_rewires < 0:
                raise ValueError("successful_rewires must be non-negative.")

            schedule[key] = {
                "rewiring": str(row["rewiring"]),
                "rewire_theta": float(row["rewire_theta"]),
                "rewire_every": int(row["rewire_every"]),
                "base_seed": int(row["base_seed"]),
                "social_network": str(row["social_network"]),
                "social_k": int(row["social_k"]),
                "training_steps": int(row["training_steps"]),
                "successful_rewires": successful_rewires,
                "reward_definition": json.loads(row["reward_definition"])
                if row.get("reward_definition")
                else reward_definition(),
            }

    return schedule


def matched_rewire_target(
    args, *, scenario_name: str, population: int, replicate: int, time: int
) -> int:
    """Return the exact adaptive event count for one paired R0 checkpoint."""
    if not args.matched_rewire_schedule:
        raise ValueError("random_matched rewiring requires --matched-rewire-schedule.")

    resolved_path = str(Path(args.matched_rewire_schedule).expanduser().resolve())
    schedule = load_matched_rewire_schedule(resolved_path)
    key = (scenario_name, int(population), int(replicate), int(time))

    if key not in schedule:
        raise ValueError(
            "Matched rewiring schedule has no row for "
            f"scenario={scenario_name}, population={population}, "
            f"replicate={replicate}, time={time}."
        )

    row = schedule[key]
    if row["reward_definition"] != reward_identity(vars(args)):
        raise ValueError("Matched R0 reward definition does not match the adaptive schedule.")

    if row["rewiring"] != "prediction_error":
        raise ValueError("Matched R0 schedule must come from a prediction_error run.")

    if not np.isclose(float(row["rewire_theta"]), float(args.rewire_theta), rtol=0.0, atol=1e-12):
        raise ValueError(
            "Matched R0 theta does not match the adaptive schedule: "
            f"current={args.rewire_theta}, source={row['rewire_theta']}."
        )

    if int(row["rewire_every"]) != int(args.rewire_every):
        raise ValueError("Matched R0 rewire interval does not match the adaptive schedule.")

    if int(row["base_seed"]) != int(args.seed):
        raise ValueError("Matched R0 base seed does not match the adaptive schedule.")

    if row["social_network"] != "random_k":
        raise ValueError("Matched R0 schedule must come from a random_k adaptive run.")

    if int(row["social_k"]) != int(args.social_k):
        raise ValueError("Matched R0 social_k does not match the adaptive schedule.")

    if int(row["training_steps"]) != int(args.training_steps):
        raise ValueError("Matched R0 training length does not match the adaptive schedule.")

    return int(row["successful_rewires"])


def treatment_name(args) -> str:
    """Return a compact treatment identifier for run metadata."""
    if args.social_mode == "none":
        return "B0"
    if args.social_network == "ba":
        return "S2"
    if args.rewiring == "none":
        return "S1"
    if args.rewiring == "random_matched":
        return "R0"
    if args.rewiring == "prediction_error":
        if np.isclose(args.rewire_theta, 0.0):
            return "R1"
        if np.isclose(args.rewire_theta, 0.25):
            return "R2"
        if np.isclose(args.rewire_theta, 1.0):
            return "R3"
        return "R_adaptive"
    raise ValueError(f"Unknown rewiring mode: {args.rewiring}")


def validate_configuration(args) -> None:
    """Validate ecological, social-network, rewiring, and evaluation settings."""
    reward_definition(getattr(args, "reward_mode", "harvest"), args.metabolism)
    network_eval = getattr(args, "network_eval", "frozen")

    if network_eval not in NETWORK_EVAL_MODES:
        raise ValueError(f"Unknown network evaluation mode: {network_eval}")

    for scenario in args.scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")

    if not 0.0 <= args.rewire_theta <= 1.0:
        raise ValueError("--rewire-theta must be between 0 and 1.")
    if not 0.0 <= args.rewire_mu <= 1.0:
        raise ValueError("--rewire-mu must be between 0 and 1.")
    if args.rewire_every <= 0:
        raise ValueError("--rewire-every must be positive.")
    if not 0.0 <= args.rewire_threshold <= 1.0:
        raise ValueError("--rewire-threshold must be between 0 and 1.")
    if not 0.0 <= args.forecast_alpha <= 1.0:
        raise ValueError("--forecast-alpha must be between 0 and 1.")
    if args.record_network_every <= 0:
        raise ValueError("--record-network-every must be positive.")

    if args.social_mode == "none":
        if args.rewiring != "none":
            raise ValueError("Rewiring requires --social-mode fixed.")
        if args.matched_rewire_schedule:
            raise ValueError(
                "--matched-rewire-schedule requires --social-mode fixed and "
                "--rewiring random_matched."
            )
        if network_eval == "adaptive":
            raise ValueError("Adaptive network evaluation requires --social-mode fixed.")
        return

    if args.social_mode != "fixed":
        raise ValueError(f"Unknown social mode: {args.social_mode}")

    if args.social_network not in SOCIAL_NETWORK_MODES:
        raise ValueError(f"Unknown social network: {args.social_network}")

    if args.social_network == "random_k":
        if args.social_k < 1:
            raise ValueError("--social-k must be at least 1.")
        for population in args.populations:
            if args.social_k >= population:
                raise ValueError(
                    "--social-k must be smaller than every population size. "
                    f"Received k={args.social_k}, population={population}."
                )
            if args.rewiring != "none" and args.social_k >= population - 1:
                raise ValueError(
                    "Rewiring requires at least one unobserved replacement "
                    "candidate for every observer, so --social-k must be at "
                    "most population - 2."
                )

    elif args.social_network == "ba":
        if args.rewiring != "none":
            raise ValueError(
                "BA social networks are fixed S2 controls in the current "
                "experiment and cannot be rewired."
            )
        if args.ba_m < 1:
            raise ValueError("--ba-m must be at least 1.")
        for population in args.populations:
            if args.ba_m >= population:
                raise ValueError(
                    "--ba-m must be smaller than every population size. "
                    f"Received m={args.ba_m}, population={population}."
                )

    if network_eval == "adaptive":
        if args.social_network != "random_k":
            raise ValueError(
                "Adaptive network evaluation is currently supported only "
                "for random_k social networks."
            )
        if args.rewiring != "prediction_error":
            if args.rewiring == "random_matched":
                raise ValueError(
                    "Adaptive network evaluation is not yet available for "
                    "random_matched R0 because no matched evaluation-phase "
                    "rewiring schedule exists. Use --network-eval frozen."
                )
            raise ValueError(
                "Adaptive network evaluation requires a training rewiring rule: prediction_error."
            )

    if args.rewiring == "random_matched":
        if not args.matched_rewire_schedule:
            raise ValueError("random_matched rewiring requires --matched-rewire-schedule.")
        load_matched_rewire_schedule(str(Path(args.matched_rewire_schedule).expanduser().resolve()))
    elif args.matched_rewire_schedule:
        raise ValueError("--matched-rewire-schedule is only valid with --rewiring random_matched.")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def gini(values) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 0:
        return 0.0
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    n = len(values)
    index = np.arange(1, n + 1)
    value = 2.0 * np.sum(index * values) / (n * total) - (n + 1.0) / n
    return float(max(0.0, value))


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


def mean_or_nan(values) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


# ---------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------


def landscape_seed(replicate: int, base_seed: int) -> int:
    return base_seed + replicate


def agent_seed(replicate: int, base_seed: int) -> int:
    return base_seed + 100_000 + replicate


def social_seed(replicate: int, population: int, base_seed: int) -> int:
    return base_seed + 300_000 + 10_000 * replicate + population


def rewiring_seed(replicate: int, population: int, base_seed: int) -> int:
    return base_seed + 500_000 + 10_000 * replicate + population


def evaluation_rewiring_seed(replicate: int, population: int, base_seed: int) -> int:
    """Independent RNG stream for adaptive network evaluation."""
    return base_seed + 600_000 + 10_000 * replicate + population


# ---------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------


def make_environment(scenario_name: str, population: int, replicate: int, max_steps: int, args):
    capacity, recovery, equilibrium, regions = build_environment_maps(
        scenario_name,
        width=args.width,
        height=args.height,
        seed=landscape_seed(replicate, args.seed),
    )

    env = EcoEnv(
        reward_mode=getattr(args, "reward_mode", "harvest"),
        width=args.width,
        height=args.height,
        n_agents=population,
        max_steps=max_steps,
        regeneration_rate=recovery,
        equilibrium_fraction=equilibrium,
        coupling_rate=args.coupling,
        cooperative_harvest_amount=args.low_harvest,
        defective_harvest_amount=args.high_harvest,
        metabolism_rate=args.metabolism,
        initial_energy=args.initial_energy,
        energy_capacity=args.energy_capacity,
        initial_resource_fraction=args.initial_resource_fraction,
        capacity_map=capacity,
    )

    observations, _ = env.reset(seed=agent_seed(replicate, args.seed))
    return env, observations, regions


# ---------------------------------------------------------------------
# Social observation
# ---------------------------------------------------------------------


def make_social_sources(env: EcoEnv, replicate: int, args) -> dict[str, list[str]] | None:
    """Create the requested initial social-information network."""
    if args.social_mode == "none":
        return None
    if args.social_mode != "fixed":
        raise ValueError(f"Unknown social mode: {args.social_mode}")

    rng = np.random.default_rng(social_seed(replicate, len(env.possible_agents), args.seed))
    agents = list(env.possible_agents)

    if args.social_network == "random_k":
        return init_random_attention(agents, args.social_k, rng)
    if args.social_network == "ba":
        return init_barabasi_albert_attention(agents, args.ba_m, rng)
    raise ValueError(f"Unknown social network: {args.social_network}")


def learner_state_count(social_mode: str) -> int:
    if social_mode == "none":
        return 3
    if social_mode == "fixed":
        return 9
    raise ValueError(f"Unknown social mode: {social_mode}")


def encode_states(
    observations: dict,
    *,
    social_mode: str,
    sources: dict[str, list[str]] | None,
    previous_actions: dict[str, int] | None,
) -> dict[str, int]:
    ecological_states = {
        name: resource_state(observation) for name, observation in observations.items()
    }

    if social_mode == "none":
        return ecological_states
    if social_mode != "fixed":
        raise ValueError(f"Unknown social mode: {social_mode}")
    if sources is None:
        raise ValueError("Social observation requires sources.")

    states: dict[str, int] = {}
    for name, ecological_state in ecological_states.items():
        low_fraction = observed_low_fraction(name, sources, previous_actions)
        states[name] = joint_state(ecological_state, social_bin(low_fraction))
    return states


# ---------------------------------------------------------------------
# Q-learning
# ---------------------------------------------------------------------


def make_learners(env: EcoEnv, replicate: int, args) -> dict[str, QLearningPolicy]:
    learners: dict[str, QLearningPolicy] = {}
    n_states = learner_state_count(args.social_mode)

    for index, name in enumerate(env.possible_agents):
        learners[name] = QLearningPolicy(
            n_states=n_states,
            n_actions=2,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            seed=args.seed + 200_000 + 10_000 * replicate + index,
        )
    return learners


# ---------------------------------------------------------------------
# System and social metrics
# ---------------------------------------------------------------------


def system_metrics(
    env: EcoEnv, actions: dict[str, int], *, sources: dict[str, list[str]] | None = None
) -> dict:
    agents = list(env.model.by_name.values())
    low_rate = float(np.mean([a == LOW_EXTRACT for a in actions.values()]))

    if sources is None:
        social = {
            "visibility_gini": float("nan"),
            "max_visibility_share": float("nan"),
            "zero_visibility_fraction": float("nan"),
            "reciprocity": float("nan"),
            "degree_assortativity": float("nan"),
            "population_low_fraction": float("nan"),
            "visible_low_fraction": float("nan"),
            "visible_population_bias": float("nan"),
            "mean_perception_error": float("nan"),
            "signed_perception_bias": float("nan"),
            "majority_mismatch_rate": float("nan"),
            "majority_tie_rate": float("nan"),
            "degree_action_correlation": float("nan"),
        }
    else:
        social = social_metrics(sources, actions)

    return {
        "low_extraction_rate": low_rate,
        "collective_order": abs(2.0 * low_rate - 1.0),
        "action_entropy": binary_entropy(low_rate),
        "mean_resource_fraction": float(np.mean(env.model.resource / env.model.capacity)),
        "total_resource": float(env.model.resource.sum()),
        "mean_reserve_welfare": float(np.mean([agent.reserve_welfare for agent in agents])),
        "mean_need_satisfaction": float(np.mean([agent.need_satisfaction for agent in agents])),
        "deprivation_rate": float(np.mean([agent.metabolic_shortfall > 1e-12 for agent in agents])),
        "mean_metabolic_shortfall": float(np.mean([agent.metabolic_shortfall for agent in agents])),
        "mean_energy": float(np.mean([agent.energy for agent in agents])),
        "wealth_gini": gini([agent.wealth for agent in agents]),
        "mean_wealth": float(np.mean([agent.wealth for agent in agents])),
        "mean_social_low_fraction": social["visible_low_fraction"],
        "visibility_gini": social["visibility_gini"],
        "max_visibility_share": social["max_visibility_share"],
        "zero_visibility_fraction": social["zero_visibility_fraction"],
        "reciprocity": social["reciprocity"],
        "degree_assortativity": social["degree_assortativity"],
        "visible_population_bias": social["visible_population_bias"],
        "social_perception_error": social["mean_perception_error"],
        "signed_perception_bias": social["signed_perception_bias"],
        "majority_mismatch_rate": social["majority_mismatch_rate"],
        "majority_tie_rate": social["majority_tie_rate"],
        "degree_action_correlation": social["degree_action_correlation"],
    }


def make_network_record(
    *,
    scenario_name: str,
    population: int,
    replicate: int,
    time: int,
    args,
    sources: dict[str, list[str]],
    previous_recorded_sources: dict[str, list[str]],
    actions: dict[str, int] | None,
    prediction_error_values: dict[str, float] | None,
    events_since_record: list[dict[str, object]],
    cumulative_rewires: int,
) -> dict:
    metrics = social_metrics(sources, actions)
    turnover = network_turnover(previous_recorded_sources, sources)

    if prediction_error_values:
        mean_prediction_error = float(np.mean(list(prediction_error_values.values())))
    else:
        mean_prediction_error = float("nan")

    if events_since_record:
        global_rewire_fraction = float(
            np.mean([event["used_scope"] == "global" for event in events_since_record])
        )
        requested_global_fraction = float(
            np.mean([event["requested_scope"] == "global" for event in events_since_record])
        )
        local_fallbacks = int(sum(bool(event["fallback"]) for event in events_since_record))
    else:
        global_rewire_fraction = 0.0
        requested_global_fraction = 0.0
        local_fallbacks = 0

    return {
        "scenario": scenario_name,
        "population": population,
        "replicate": replicate,
        "social_mode": args.social_mode,
        "social_network": getattr(args, "social_network", "random_k"),
        "rewiring": args.rewiring,
        "time": time,
        "visibility_gini": metrics["visibility_gini"],
        "max_visibility_share": metrics["max_visibility_share"],
        "zero_visibility_fraction": metrics["zero_visibility_fraction"],
        "reciprocity": metrics["reciprocity"],
        "degree_assortativity": metrics["degree_assortativity"],
        "population_low_fraction": metrics["population_low_fraction"],
        "visible_low_fraction": metrics["visible_low_fraction"],
        "visible_population_bias": metrics["visible_population_bias"],
        "mean_perception_error": metrics["mean_perception_error"],
        "signed_perception_bias": metrics["signed_perception_bias"],
        "majority_mismatch_rate": metrics["majority_mismatch_rate"],
        "majority_tie_rate": metrics["majority_tie_rate"],
        "degree_action_correlation": metrics["degree_action_correlation"],
        "mean_prediction_error": mean_prediction_error,
        "edge_turnover": turnover,
        "rewires_since_record": len(events_since_record),
        "cumulative_rewires": cumulative_rewires,
        "global_rewire_fraction": global_rewire_fraction,
        "requested_global_fraction": requested_global_fraction,
        "local_fallbacks": local_fallbacks,
    }


# ---------------------------------------------------------------------
# Per-agent social summaries and network snapshots
# ---------------------------------------------------------------------


def make_agent_social_accumulators(agent_names) -> dict[str, dict[str, float | int]]:
    """Create zeroed training-time social diagnostic accumulators."""

    return {
        str(name): {
            "social_steps": 0,
            "observed_low_sum": 0.0,
            "population_low_excluding_sum": 0.0,
            "perception_error_sum": 0.0,
            "signed_bias_sum": 0.0,
            "majority_tie_count": 0,
            "majority_valid_count": 0,
            "majority_mismatch_count": 0,
            "visibility_degree_sum": 0.0,
        }
        for name in agent_names
    }


def update_agent_social_accumulators(
    accumulators: dict[str, dict[str, float | int]],
    focal_diagnostics: dict[str, dict[str, float | bool]],
) -> None:
    """Accumulate one pre-rewiring social observation for each observer."""

    if set(accumulators) != set(focal_diagnostics):
        raise ValueError("Social accumulators and focal diagnostics must contain the same agents.")

    for name, diagnostic in focal_diagnostics.items():
        actual = float(diagnostic["population_low_fraction_excluding"])

        if not np.isfinite(actual):
            continue

        accumulator = accumulators[name]
        accumulator["social_steps"] = int(accumulator["social_steps"]) + 1
        accumulator["observed_low_sum"] = float(accumulator["observed_low_sum"]) + float(
            diagnostic["observed_low_fraction"]
        )
        accumulator["population_low_excluding_sum"] = (
            float(accumulator["population_low_excluding_sum"]) + actual
        )
        accumulator["perception_error_sum"] = float(accumulator["perception_error_sum"]) + float(
            diagnostic["perception_error"]
        )
        accumulator["signed_bias_sum"] = float(accumulator["signed_bias_sum"]) + float(
            diagnostic["signed_perception_bias"]
        )
        accumulator["visibility_degree_sum"] = float(accumulator["visibility_degree_sum"]) + float(
            diagnostic["visibility_degree"]
        )

        if bool(diagnostic["majority_tied"]):
            accumulator["majority_tie_count"] = int(accumulator["majority_tie_count"]) + 1
        else:
            mismatch = float(diagnostic["majority_mismatch"])

            if np.isfinite(mismatch):
                accumulator["majority_valid_count"] = int(accumulator["majority_valid_count"]) + 1
                accumulator["majority_mismatch_count"] = int(
                    accumulator["majority_mismatch_count"]
                ) + int(mismatch > 0.5)


def summarize_agent_social_training(
    env: EcoEnv,
    regions: np.ndarray,
    *,
    scenario_name: str,
    population: int,
    replicate: int,
    args,
    terminal_sources: dict[str, list[str]] | None,
    rewire_counts: dict[str, int],
    mean_prediction_errors: dict[str, float],
    accumulators: dict[str, dict[str, float | int]],
) -> list[dict]:
    """Build one normalized training/social outcome row per agent."""

    if terminal_sources is None:
        final_visibility = {name: 0 for name in env.possible_agents}
    else:
        final_visibility = visibility_counts(terminal_sources)

    rows: list[dict] = []

    for name in env.possible_agents:
        agent = env.model.by_name[name]
        x = int(agent.position[0])
        y = int(agent.position[1])

        accumulator = accumulators[name]
        social_steps = int(accumulator["social_steps"])
        valid_majority_steps = int(accumulator["majority_valid_count"])

        def mean_from_sum(key: str) -> float:
            if social_steps <= 0:
                return float("nan")

            return float(accumulator[key]) / social_steps

        rows.append(
            {
                "scenario": scenario_name,
                "population": population,
                "replicate": replicate,
                "treatment": treatment_name(args),
                "social_mode": args.social_mode,
                "social_network": getattr(args, "social_network", "random_k"),
                "rewiring": args.rewiring,
                "agent": name,
                "x": x,
                "y": y,
                "region": str(regions[y, x]),
                "social_steps": social_steps,
                "mean_observed_low_fraction": mean_from_sum("observed_low_sum"),
                "mean_population_low_fraction_excluding": mean_from_sum(
                    "population_low_excluding_sum"
                ),
                "mean_perception_error": mean_from_sum("perception_error_sum"),
                "mean_signed_perception_bias": mean_from_sum("signed_bias_sum"),
                "majority_tie_rate": (
                    float(accumulator["majority_tie_count"]) / social_steps
                    if social_steps > 0
                    else float("nan")
                ),
                "majority_valid_steps": valid_majority_steps,
                "majority_mismatch_rate": (
                    float(accumulator["majority_mismatch_count"]) / valid_majority_steps
                    if valid_majority_steps > 0
                    else float("nan")
                ),
                "mean_visibility_degree": mean_from_sum("visibility_degree_sum"),
                "final_visibility_degree": int(final_visibility[name]),
                "final_attention_size": (
                    0 if terminal_sources is None else len(terminal_sources[name])
                ),
                "rewires_initiated": int(rewire_counts[name]),
                "mean_prediction_error": float(mean_prediction_errors[name]),
                "final_wealth": float(agent.wealth),
                "final_energy": float(agent.energy),
                "final_reserve_welfare": float(agent.reserve_welfare),
                "final_need_satisfaction": float(agent.need_satisfaction),
                "final_metabolic_shortfall": float(agent.metabolic_shortfall),
                "final_cumulative_shortfall": float(agent.cumulative_shortfall),
                "final_deprivation_steps": int(agent.deprivation_steps),
            }
        )

    return rows


def network_snapshot_rows(
    *,
    scenario_name: str,
    population: int,
    replicate: int,
    args,
    checkpoint: str,
    time: int,
    sources: dict[str, list[str]] | None,
) -> list[dict]:
    """Return normalized directed edge rows for a selected graph snapshot."""

    if sources is None:
        return []

    visibility = visibility_counts(sources)

    return [
        {
            "scenario": scenario_name,
            "population": population,
            "replicate": replicate,
            "treatment": treatment_name(args),
            "social_mode": args.social_mode,
            "social_network": getattr(args, "social_network", "random_k"),
            "rewiring": args.rewiring,
            "checkpoint": checkpoint,
            "time": time,
            "source": source,
            "observer": observer,
            "source_visibility_degree": int(visibility[source]),
            "observer_visibility_degree": int(visibility[observer]),
            "source_attention_size": len(sources[source]),
            "observer_attention_size": len(sources[observer]),
        }
        for source, observer in sorted(network_edges(sources))
    ]


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def train_q_learning(scenario_name: str, population: int, replicate: int, args):
    total_steps = args.training_steps + args.evaluation_steps
    env, observations, regions = make_environment(
        scenario_name, population, replicate, total_steps, args
    )
    sources = make_social_sources(env, replicate, args)
    initial_sources = None if sources is None else copy_sources(sources)
    learners = make_learners(env, replicate, args)

    state_visit_counts = np.zeros(learner_state_count(args.social_mode), dtype=int)
    agent_social_accumulators = make_agent_social_accumulators(env.possible_agents)

    timeseries: list[dict] = []
    network_timeseries: list[dict] = []
    rewiring_schedule: list[dict] = []
    previous_actions = None

    rewire_counts = {name: 0 for name in env.possible_agents}
    prediction_error_sums = {name: 0.0 for name in env.possible_agents}
    prediction_error_counts = {name: 0 for name in env.possible_agents}

    if sources is None:
        forecasts = None
        rewire_rng = None
        previous_recorded_sources = None
    else:
        forecasts = {name: 0.5 for name in env.possible_agents}
        rewire_rng = np.random.default_rng(rewiring_seed(replicate, population, args.seed))
        previous_recorded_sources = copy_sources(sources)
        network_timeseries.append(
            make_network_record(
                scenario_name=scenario_name,
                population=population,
                replicate=replicate,
                time=0,
                args=args,
                sources=sources,
                previous_recorded_sources=previous_recorded_sources,
                actions=None,
                prediction_error_values=None,
                events_since_record=[],
                cumulative_rewires=0,
            )
        )

    events_since_record: list[dict[str, object]] = []
    cumulative_rewires = 0

    for time in range(1, args.training_steps + 1):
        # (G_t, a_(t-1), R_t) -> s_t
        states = encode_states(
            observations,
            social_mode=args.social_mode,
            sources=sources,
            previous_actions=previous_actions,
        )

        for state in states.values():
            state_visit_counts[int(state)] += 1

        # s_t -> a_t
        actions = {
            name: learners[name].choose_action(states[name], explore=True) for name in env.agents
        }

        # a_t -> R_(t+1)
        (next_observations, rewards, terminations, truncations, _) = env.step(actions)

        current_prediction_errors = None

        # Observe a_t through G_t and optionally construct G_(t+1).
        if sources is not None:
            assert forecasts is not None
            assert rewire_rng is not None

            observed_before_rewire = social_observations(sources, actions)

            focal_diagnostics = observer_social_diagnostics(sources, actions)
            update_agent_social_accumulators(agent_social_accumulators, focal_diagnostics)

            current_prediction_errors = prediction_errors(forecasts, observed_before_rewire)

            for name, error in current_prediction_errors.items():
                prediction_error_sums[name] += error
                prediction_error_counts[name] += 1

            events: list[dict[str, object]] = []

            if time % args.rewire_every == 0:
                target_count = None
                if args.rewiring == "random_matched":
                    target_count = matched_rewire_target(
                        args,
                        scenario_name=scenario_name,
                        population=population,
                        replicate=replicate,
                        time=time,
                    )

                events = rewire_epoch(
                    sources,
                    mode=args.rewiring,
                    theta=args.rewire_theta,
                    mu=args.rewire_mu,
                    rng=rewire_rng,
                    prediction_error_values=current_prediction_errors,
                    threshold=args.rewire_threshold,
                    target_count=target_count,
                )

                if args.rewiring != "none":
                    rewiring_schedule.append(
                        {
                            "scenario": scenario_name,
                            "population": population,
                            "replicate": replicate,
                            "time": time,
                            "rewiring": args.rewiring,
                            "rewire_theta": args.rewire_theta,
                            "rewire_every": args.rewire_every,
                            "base_seed": args.seed,
                            "reward_definition": json.dumps(
                                reward_identity(vars(args)), sort_keys=True
                            ),
                            "social_network": args.social_network,
                            "social_k": args.social_k,
                            "training_steps": args.training_steps,
                            "target_rewires": ("" if target_count is None else target_count),
                            "successful_rewires": len(events),
                        }
                    )

                for event in events:
                    observer = str(event["observer"])
                    rewire_counts[observer] += 1

                cumulative_rewires += len(events)
                events_since_record.extend(events)

            # Forecast update uses the pre-rewiring observation.
            update_forecasts(forecasts, observed_before_rewire, alpha=args.forecast_alpha)

        # R_(t+1), G_(t+1), a_t -> s_(t+1).
        # This must remain after rewiring.
        next_states = encode_states(
            next_observations,
            social_mode=args.social_mode,
            sources=sources,
            previous_actions=actions,
        )

        for name in states:
            done = terminations[name] or truncations[name]
            learner_next_state = states[name] if done else next_states[name]
            learners[name].update(
                states[name], actions[name], rewards[name], learner_next_state, done=done
            )
            learners[name].decay_exploration()

        if time == 1 or time % args.record_every == 0 or time == args.training_steps:
            timeseries.append(
                {
                    "scenario": scenario_name,
                    "population": population,
                    "replicate": replicate,
                    "social_mode": args.social_mode,
                    "social_network": getattr(args, "social_network", "random_k"),
                    "rewiring": args.rewiring,
                    "time": time,
                    **system_metrics(env, actions, sources=sources),
                    "mean_epsilon": float(
                        np.mean([learner.epsilon for learner in learners.values()])
                    ),
                }
            )

        if sources is not None and (
            time == 1 or time % args.record_network_every == 0 or time == args.training_steps
        ):
            assert previous_recorded_sources is not None
            network_timeseries.append(
                make_network_record(
                    scenario_name=scenario_name,
                    population=population,
                    replicate=replicate,
                    time=time,
                    args=args,
                    sources=sources,
                    previous_recorded_sources=previous_recorded_sources,
                    actions=actions,
                    prediction_error_values=current_prediction_errors,
                    events_since_record=events_since_record,
                    cumulative_rewires=cumulative_rewires,
                )
            )
            previous_recorded_sources = copy_sources(sources)
            events_since_record = []

        previous_actions = actions.copy()
        observations = next_observations

    mean_prediction_errors: dict[str, float] = {}
    for name in env.possible_agents:
        count = prediction_error_counts[name]
        if count > 0:
            mean_prediction_errors[name] = prediction_error_sums[name] / count
        else:
            mean_prediction_errors[name] = float("nan")

    final_forecasts = (
        None if forecasts is None else {name: float(value) for name, value in forecasts.items()}
    )

    agent_social_rows = summarize_agent_social_training(
        env,
        regions,
        scenario_name=scenario_name,
        population=population,
        replicate=replicate,
        args=args,
        terminal_sources=sources,
        rewire_counts=rewire_counts,
        mean_prediction_errors=mean_prediction_errors,
        accumulators=agent_social_accumulators,
    )

    training_measurements = {
        "state_visit_counts": state_visit_counts,
        "agent_social_summary": agent_social_rows,
    }

    return (
        env,
        observations,
        regions,
        learners,
        timeseries,
        network_timeseries,
        sources,
        previous_actions,
        rewire_counts,
        mean_prediction_errors,
        rewiring_schedule,
        initial_sources,
        final_forecasts,
        training_measurements,
    )


# ---------------------------------------------------------------------
# State-count diagnostics
# ---------------------------------------------------------------------


def empty_state_counts(social_mode: str) -> dict[str, np.ndarray]:
    if social_mode == "none":
        social_states = 1
    elif social_mode == "fixed":
        social_states = 3
    else:
        raise ValueError(f"Unknown social mode: {social_mode}")

    shape = (3, social_states)
    return {"visits": np.zeros(shape, dtype=int), "low_actions": np.zeros(shape, dtype=int)}


def decode_state(state: int, social_mode: str) -> tuple[int, int]:
    if social_mode == "none":
        return int(state), 0
    if social_mode == "fixed":
        return int(state) // 3, int(state) % 3
    raise ValueError(f"Unknown social mode: {social_mode}")


def update_state_counts(
    counts: dict[str, np.ndarray],
    states: dict[str, int],
    actions: dict[str, int],
    *,
    social_mode: str,
) -> None:
    for name, state in states.items():
        ecological_state, social_state = decode_state(state, social_mode)
        counts["visits"][ecological_state, social_state] += 1
        if actions[name] == LOW_EXTRACT:
            counts["low_actions"][ecological_state, social_state] += 1


def summarize_state_counts(counts: dict[str, np.ndarray], *, social_mode: str) -> dict:
    visits = counts["visits"]
    low_actions = counts["low_actions"]
    total = int(visits.sum())
    result: dict[str, float] = {}

    for ecological_index, ecological_name in enumerate(STATE_NAMES):
        ecological_visits = int(visits[ecological_index, :].sum())
        ecological_low = int(low_actions[ecological_index, :].sum())
        result[f"state_occupancy_{ecological_name}"] = (
            ecological_visits / total if total > 0 else float("nan")
        )
        result[f"low_given_{ecological_name}"] = (
            ecological_low / ecological_visits if ecological_visits > 0 else float("nan")
        )

    if social_mode == "fixed":
        for social_index, social_name in enumerate(SOCIAL_STATE_NAMES):
            social_visits = int(visits[:, social_index].sum())
            result[f"social_occupancy_{social_name}"] = (
                social_visits / total if total > 0 else float("nan")
            )

        for ecological_index, ecological_name in enumerate(STATE_NAMES):
            for social_index, social_name in enumerate(SOCIAL_STATE_NAMES):
                joint_visits = int(visits[ecological_index, social_index])
                joint_low = int(low_actions[ecological_index, social_index])
                result[f"joint_occupancy_{ecological_name}_{social_name}"] = (
                    joint_visits / total if total > 0 else float("nan")
                )
                result[f"low_given_{ecological_name}_{social_name}"] = (
                    joint_low / joint_visits if joint_visits > 0 else float("nan")
                )

    return result


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def action_rule(
    strategy: str,
    observations: dict,
    learners: dict[str, QLearningPolicy] | None,
    rng: np.random.Generator,
    *,
    social_mode: str = "none",
    sources: dict[str, list[str]] | None = None,
    previous_actions: dict[str, int] | None = None,
):
    states = encode_states(
        observations, social_mode=social_mode, sources=sources, previous_actions=previous_actions
    )

    if strategy == "q_learning":
        if learners is None:
            raise ValueError("q_learning evaluation requires learners.")
        actions = {
            name: learners[name].choose_action(states[name], explore=False) for name in observations
        }
    elif strategy == "always_low":
        actions = {name: LOW_EXTRACT for name in observations}
    elif strategy == "always_high":
        actions = {name: HIGH_EXTRACT for name in observations}
    elif strategy == "random_50":
        actions = {name: int(rng.integers(2)) for name in observations}
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return states, actions


def evaluate_policy(
    env: EcoEnv,
    observations: dict,
    *,
    scenario_name: str,
    population: int,
    replicate: int,
    strategy: str,
    evaluation_mode: str,
    learners: dict[str, QLearningPolicy] | None,
    sources: dict[str, list[str]] | None,
    previous_actions: dict[str, int] | None,
    args,
    network_start: str = "none",
    adaptive_network: bool = False,
    forecasts: dict[str, float] | None = None,
):
    """
    Evaluate a learned or fixed policy.

    Frozen evaluation is the default. When ``adaptive_network`` is true,
    Q-tables remain frozen but the supplied starting social graph continues
    to adapt using the run's training rewiring rule. The starting graph and
    forecast dictionary are copied so evaluation cannot mutate training
    state retained by the caller.
    """
    rng = np.random.default_rng(args.seed + 400_000 + 10_000 * replicate + population)

    evaluation_sources = None if sources is None else copy_sources(sources)

    evaluation_forecasts = None
    evaluation_rewire_rng = None

    if adaptive_network:
        if evaluation_sources is None:
            raise ValueError("Adaptive network evaluation requires a social network.")
        if getattr(args, "social_network", "random_k") != "random_k":
            raise ValueError("Adaptive network evaluation currently requires random_k.")
        if args.rewiring != "prediction_error":
            raise ValueError(
                "Adaptive network evaluation currently supports only prediction_error rewiring."
            )

        if forecasts is None:
            if args.rewiring == "prediction_error":
                raise ValueError(
                    "Prediction-error adaptive evaluation requires carried terminal forecasts."
                )
            evaluation_forecasts = {name: 0.5 for name in evaluation_sources}
        else:
            if set(forecasts) != set(evaluation_sources):
                raise ValueError("Evaluation forecasts must contain every social observer.")
            evaluation_forecasts = {name: float(value) for name, value in forecasts.items()}

        evaluation_rewire_rng = np.random.default_rng(
            evaluation_rewiring_seed(replicate, population, args.seed)
        )

    state_counts = empty_state_counts(args.social_mode)
    snapshots: list[dict] = []
    timeseries: list[dict] = []
    cumulative_evaluation_rewires = 0
    names = env.possible_agents
    utility_sum = np.zeros(len(names))
    utility_discounted = np.zeros(len(names))
    harvest_sum = np.zeros(len(names))
    harvest_discounted = np.zeros(len(names))

    for time in range(1, args.evaluation_steps + 1):
        states, actions = action_rule(
            strategy,
            observations,
            learners,
            rng,
            social_mode=args.social_mode,
            sources=evaluation_sources,
            previous_actions=previous_actions,
        )
        update_state_counts(state_counts, states, actions, social_mode=args.social_mode)

        next_observations, rewards, _, truncations, infos = env.step(actions)
        utility = np.array([rewards[name] for name in names])
        harvest = np.array([infos[name]["harvested"] for name in names])
        utility_sum += utility
        utility_discounted += args.gamma ** (time - 1) * utility
        harvest_sum += harvest
        harvest_discounted += args.gamma ** (time - 1) * harvest

        evaluation_rewires_step = 0

        if adaptive_network:
            assert evaluation_sources is not None
            assert evaluation_forecasts is not None
            assert evaluation_rewire_rng is not None

            observed_before_rewire = social_observations(evaluation_sources, actions)
            current_prediction_errors = prediction_errors(
                evaluation_forecasts, observed_before_rewire
            )

            if time % args.rewire_every == 0:
                events = rewire_epoch(
                    evaluation_sources,
                    mode=args.rewiring,
                    theta=args.rewire_theta,
                    mu=args.rewire_mu,
                    rng=evaluation_rewire_rng,
                    prediction_error_values=current_prediction_errors,
                    threshold=args.rewire_threshold,
                    target_count=None,
                )
                evaluation_rewires_step = len(events)
                cumulative_evaluation_rewires += evaluation_rewires_step

            update_forecasts(
                evaluation_forecasts, observed_before_rewire, alpha=args.forecast_alpha
            )

        metrics = system_metrics(env, actions, sources=evaluation_sources)
        snapshots.append(metrics)

        if time == 1 or time % args.record_every == 0 or time == args.evaluation_steps:
            timeseries.append(
                {
                    "scenario": scenario_name,
                    "population": population,
                    "replicate": replicate,
                    "social_mode": args.social_mode,
                    "social_network": getattr(args, "social_network", "random_k"),
                    "rewiring": args.rewiring,
                    "strategy": strategy,
                    "evaluation_mode": evaluation_mode,
                    "network_start": network_start,
                    "network_adaptive": adaptive_network,
                    "evaluation_rewires_step": evaluation_rewires_step,
                    "evaluation_rewires_cumulative": (cumulative_evaluation_rewires),
                    "time": time,
                    **metrics,
                }
            )

        previous_actions = actions.copy()
        observations = next_observations
        if any(truncations.values()):
            break

    summary = {
        "scenario": scenario_name,
        "scenario_label": SCENARIOS[scenario_name]["label"],
        "scenario_group": SCENARIOS[scenario_name]["group"],
        "population": population,
        "replicate": replicate,
        "social_mode": args.social_mode,
        "social_network": getattr(args, "social_network", "random_k"),
        "rewiring": args.rewiring,
        "strategy": strategy,
        "evaluation_mode": evaluation_mode,
        "network_start": network_start,
        "network_adaptive": adaptive_network,
        "evaluation_total_rewires": cumulative_evaluation_rewires,
        "steps_evaluated": len(snapshots),
        "reward_mode": env.reward_mode,
        "evaluation_mean_utility_sum": float(utility_sum.mean()),
        "evaluation_mean_utility_discounted": float(utility_discounted.mean()),
        "evaluation_mean_harvest_sum": float(harvest_sum.mean()),
        "evaluation_mean_harvest_discounted": float(harvest_discounted.mean()),
    }

    metric_names = list(snapshots[0].keys()) if snapshots else []
    for metric in metric_names:
        values = [row[metric] for row in snapshots]
        finite_values = [value for value in values if np.isfinite(value)]
        summary[f"eval_mean_{metric}"] = mean_or_nan(finite_values)
        summary[f"final_{metric}"] = float(values[-1]) if values else float("nan")

    summary.update(summarize_state_counts(state_counts, social_mode=args.social_mode))
    return summary, timeseries


# ---------------------------------------------------------------------
# Learned-policy diagnostics
# ---------------------------------------------------------------------


def policy_diagnostics(
    env: EcoEnv,
    regions: np.ndarray,
    learners: dict[str, QLearningPolicy],
    scenario_name: str,
    population: int,
    replicate: int,
    *,
    social_mode: str,
    rewiring: str,
    sources: dict[str, list[str]] | None,
    rewire_counts: dict[str, int],
    mean_prediction_errors: dict[str, float],
    state_visit_counts: np.ndarray,
):
    agent_rows: list[dict] = []
    policies = {name: learner.greedy_policy() for name, learner in learners.items()}

    if sources is None:
        visibility = {name: 0 for name in learners}
    else:
        visibility = visibility_counts(sources)

    for name, learner in learners.items():
        agent = env.model.by_name[name]
        x = int(agent.position[0])
        y = int(agent.position[1])
        policy = policies[name]

        row = {
            "scenario": scenario_name,
            "population": population,
            "replicate": replicate,
            "social_mode": social_mode,
            "rewiring": rewiring,
            "agent": name,
            "x": x,
            "y": y,
            "region": str(regions[y, x]),
            "visibility_degree": int(visibility[name]),
            "attention_sources": "" if sources is None else "|".join(sources[name]),
            "rewires_initiated": int(rewire_counts[name]),
            "mean_prediction_error": float(mean_prediction_errors[name]),
        }

        if social_mode == "none":
            for ecological_index, ecological_name in enumerate(STATE_NAMES):
                row[f"policy_{ecological_name}"] = (
                    "L" if policy[ecological_index] == LOW_EXTRACT else "H"
                )
                row[f"q_{ecological_name}_low"] = float(learner.q[ecological_index, LOW_EXTRACT])
                row[f"q_{ecological_name}_high"] = float(learner.q[ecological_index, HIGH_EXTRACT])
        else:
            for ecological_index, ecological_name in enumerate(STATE_NAMES):
                for social_index, social_name in enumerate(SOCIAL_STATE_NAMES):
                    state = joint_state(ecological_index, social_index)
                    label = f"{ecological_name}_{social_name}"
                    row[f"policy_{label}"] = "L" if policy[state] == LOW_EXTRACT else "H"
                    row[f"q_{label}_low"] = float(learner.q[state, LOW_EXTRACT])
                    row[f"q_{label}_high"] = float(learner.q[state, HIGH_EXTRACT])

        agent_rows.append(row)

    policy_list = list(policies.values())
    pairwise = [
        float(np.mean(np.asarray(first) != np.asarray(second)))
        for first, second in combinations(policy_list, 2)
    ]

    policy_length = len(policy_list[0]) if policy_list else 0
    state_visit_counts = np.asarray(state_visit_counts, dtype=float)

    if len(state_visit_counts) != policy_length:
        raise ValueError("state_visit_counts must match the learned policy length.")

    total_state_visits = float(state_visit_counts.sum())

    if total_state_visits > 0.0:
        state_visit_weights = state_visit_counts / total_state_visits
    else:
        state_visit_weights = np.zeros(policy_length, dtype=float)

    pairwise_visit_weighted = [
        float(np.sum(state_visit_weights * (np.asarray(first) != np.asarray(second))))
        for first, second in combinations(policy_list, 2)
    ]

    counts = Counter(policy_list)
    probabilities = np.asarray(list(counts.values()), dtype=float)
    probabilities /= probabilities.sum()
    entropy = float(
        -np.sum(
            [
                probability * math.log2(probability)
                for probability in probabilities
                if probability > 0.0
            ]
        )
    )

    max_types = min(len(policy_list), 2**policy_length)
    if max_types > 1:
        entropy /= math.log2(max_types)
    else:
        entropy = 0.0

    summary = {
        "scenario": scenario_name,
        "population": population,
        "replicate": replicate,
        "social_mode": social_mode,
        "rewiring": rewiring,
        "unique_policy_count": len(counts),
        "policy_entropy": entropy,
        "policy_hamming_mean": float(np.mean(pairwise)) if pairwise else 0.0,
        "policy_hamming_visit_weighted_mean": (
            float(np.mean(pairwise_visit_weighted)) if pairwise_visit_weighted else 0.0
        ),
        "visited_state_fraction": (
            float(np.mean(state_visit_counts > 0.0)) if policy_length > 0 else float("nan")
        ),
        "total_training_state_visits": int(total_state_visits),
        "total_rewires": int(sum(rewire_counts.values())),
        "mean_rewires_per_agent": float(np.mean(list(rewire_counts.values()))),
    }

    finite_prediction_errors = [
        value for value in mean_prediction_errors.values() if np.isfinite(value)
    ]
    summary["mean_prediction_error"] = mean_or_nan(finite_prediction_errors)

    if sources is None:
        summary["visibility_gini"] = float("nan")
        summary["max_visibility_degree"] = float("nan")
        summary["max_visibility_share"] = float("nan")
        summary["reciprocity"] = float("nan")
        summary["degree_assortativity"] = float("nan")
    else:
        visibility_values = list(visibility.values())
        summary["visibility_gini"] = gini(visibility_values)
        summary["max_visibility_degree"] = int(max(visibility_values))
        total_visibility = float(sum(visibility_values))
        summary["max_visibility_share"] = (
            float(max(visibility_values) / total_visibility) if total_visibility > 0.0 else 0.0
        )
        terminal_network_metrics = social_metrics(sources, None)
        summary["reciprocity"] = terminal_network_metrics["reciprocity"]
        summary["degree_assortativity"] = terminal_network_metrics["degree_assortativity"]

    if social_mode == "none":
        for ecological_index, ecological_name in enumerate(STATE_NAMES):
            summary[f"policy_low_{ecological_name}"] = float(
                np.mean([policy[ecological_index] == LOW_EXTRACT for policy in policy_list])
            )
            summary[f"training_visit_fraction_{ecological_name}"] = float(
                state_visit_weights[ecological_index]
            )
    else:
        for ecological_index, ecological_name in enumerate(STATE_NAMES):
            summary[f"policy_low_{ecological_name}"] = float(
                np.mean(
                    [
                        policy[joint_state(ecological_index, social_index)] == LOW_EXTRACT
                        for policy in policy_list
                        for social_index in range(3)
                    ]
                )
            )
            summary[f"training_visit_fraction_{ecological_name}"] = float(
                np.sum(state_visit_weights[ecological_index * 3 : ecological_index * 3 + 3])
            )

        for social_index, social_name in enumerate(SOCIAL_STATE_NAMES):
            summary[f"training_visit_fraction_social_{social_name}"] = float(
                np.sum(state_visit_weights[social_index::3])
            )

        for ecological_index, ecological_name in enumerate(STATE_NAMES):
            for social_index, social_name in enumerate(SOCIAL_STATE_NAMES):
                state = joint_state(ecological_index, social_index)
                summary[f"policy_low_{ecological_name}_{social_name}"] = float(
                    np.mean([policy[state] == LOW_EXTRACT for policy in policy_list])
                )
                summary[f"training_visit_fraction_{ecological_name}_{social_name}"] = float(
                    state_visit_weights[state]
                )

    return summary, agent_rows


# ---------------------------------------------------------------------
# One experimental condition
# ---------------------------------------------------------------------


def run_condition(scenario_name: str, population: int, replicate: int, args):
    (
        trained_env,
        trained_observations,
        regions,
        learners,
        training_timeseries,
        network_timeseries,
        terminal_sources,
        final_training_actions,
        rewire_counts,
        mean_prediction_errors,
        rewiring_schedule,
        initial_sources,
        final_forecasts,
        training_measurements,
    ) = train_q_learning(scenario_name, population, replicate, args)

    policy_summary, agent_policy_rows = policy_diagnostics(
        trained_env,
        regions,
        learners,
        scenario_name,
        population,
        replicate,
        social_mode=args.social_mode,
        rewiring=args.rewiring,
        sources=terminal_sources,
        rewire_counts=rewire_counts,
        mean_prediction_errors=mean_prediction_errors,
        state_visit_counts=training_measurements["state_visit_counts"],
    )

    network_snapshots = [
        *network_snapshot_rows(
            scenario_name=scenario_name,
            population=population,
            replicate=replicate,
            args=args,
            checkpoint="initial",
            time=0,
            sources=initial_sources,
        ),
        *network_snapshot_rows(
            scenario_name=scenario_name,
            population=population,
            replicate=replicate,
            args=args,
            checkpoint="terminal",
            time=args.training_steps,
            sources=terminal_sources,
        ),
    ]

    network_start_terminal = "none" if terminal_sources is None else "terminal"

    # 1. Continuation: trained ecology + terminal network, frozen.
    continuation_summary, continuation_ts = evaluate_policy(
        trained_env,
        trained_observations,
        scenario_name=scenario_name,
        population=population,
        replicate=replicate,
        strategy="q_learning",
        evaluation_mode="continuation",
        learners=learners,
        sources=terminal_sources,
        previous_actions=final_training_actions,
        args=args,
        network_start=network_start_terminal,
        adaptive_network=False,
    )

    # 2. Fresh ecology + terminal network, frozen.
    #
    # The historical label ``fresh_reset`` is intentionally retained so the
    # existing baseline figure script continues to work. The explicit
    # ``network_start=terminal`` field records the network semantics.
    fresh_carried_env, fresh_carried_obs, _ = make_environment(
        scenario_name, population, replicate, args.evaluation_steps, args
    )

    fresh_carried_summary, fresh_carried_ts = evaluate_policy(
        fresh_carried_env,
        fresh_carried_obs,
        scenario_name=scenario_name,
        population=population,
        replicate=replicate,
        strategy="q_learning",
        evaluation_mode="fresh_reset",
        learners=learners,
        sources=terminal_sources,
        previous_actions=None,
        args=args,
        network_start=network_start_terminal,
        adaptive_network=False,
    )

    reset_network_summaries: list[dict] = []
    reset_network_timeseries: list[dict] = []

    # 3. Fresh ecology + exact initial training graph, frozen.
    if initial_sources is not None:
        fresh_reset_env, fresh_reset_obs, _ = make_environment(
            scenario_name, population, replicate, args.evaluation_steps, args
        )

        fresh_reset_summary, fresh_reset_ts = evaluate_policy(
            fresh_reset_env,
            fresh_reset_obs,
            scenario_name=scenario_name,
            population=population,
            replicate=replicate,
            strategy="q_learning",
            evaluation_mode="fresh_reset_network",
            learners=learners,
            sources=initial_sources,
            previous_actions=None,
            args=args,
            network_start="initial",
            adaptive_network=False,
        )
        reset_network_summaries.append(fresh_reset_summary)
        reset_network_timeseries.extend(fresh_reset_ts)

    adaptive_summaries: list[dict] = []
    adaptive_timeseries: list[dict] = []

    # 4. Optional robustness evaluation:
    # fresh ecology + terminal network + continued network adaptation.
    # Q-tables remain frozen. Prediction-error runs carry terminal forecasts.
    if getattr(args, "network_eval", "frozen") == "adaptive":
        adaptive_env, adaptive_obs, _ = make_environment(
            scenario_name, population, replicate, args.evaluation_steps, args
        )

        adaptive_summary, adaptive_ts = evaluate_policy(
            adaptive_env,
            adaptive_obs,
            scenario_name=scenario_name,
            population=population,
            replicate=replicate,
            strategy="q_learning",
            evaluation_mode="fresh_adaptive_network",
            learners=learners,
            sources=terminal_sources,
            previous_actions=None,
            args=args,
            network_start="terminal",
            adaptive_network=True,
            forecasts=final_forecasts,
        )
        adaptive_summaries.append(adaptive_summary)
        adaptive_timeseries.extend(adaptive_ts)

    # Fixed-policy controls use fresh ecology and the terminal graph, frozen.
    control_summaries: list[dict] = []
    control_timeseries: list[dict] = []

    for strategy in ("always_low", "always_high", "random_50"):
        control_env, control_obs, _ = make_environment(
            scenario_name, population, replicate, args.evaluation_steps, args
        )
        summary, timeseries = evaluate_policy(
            control_env,
            control_obs,
            scenario_name=scenario_name,
            population=population,
            replicate=replicate,
            strategy=strategy,
            evaluation_mode="fresh_reset",
            learners=None,
            sources=terminal_sources,
            previous_actions=None,
            args=args,
            network_start=network_start_terminal,
            adaptive_network=False,
        )
        control_summaries.append(summary)
        control_timeseries.extend(timeseries)

    return {
        "evaluation_summary": [
            continuation_summary,
            fresh_carried_summary,
            *reset_network_summaries,
            *adaptive_summaries,
            *control_summaries,
        ],
        "training_timeseries": training_timeseries,
        "evaluation_timeseries": [
            *continuation_ts,
            *fresh_carried_ts,
            *reset_network_timeseries,
            *adaptive_timeseries,
            *control_timeseries,
        ],
        "policy_summary": policy_summary,
        "agent_policies": agent_policy_rows,
        "network_timeseries": network_timeseries,
        "rewiring_schedule": rewiring_schedule,
        "agent_social_summary": training_measurements["agent_social_summary"],
        "network_edges_checkpoints": network_snapshots,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the canonical experiment CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Ecological Q-learning validation with optional social "
            "observation, fixed visibility-skew controls, decentralized "
            "rewiring, and decomposed network evaluation."
        )
    )

    parser.add_argument("--run-name", default="baseline_validation_v1")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--populations", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent conditions; 0 uses all available logical CPUs.",
    )
    parser.add_argument(
        "--resume-conditions",
        action="store_true",
        help="Reuse completed condition checkpoints from this exact configuration.",
    )
    parser.add_argument("--training-steps", type=int, default=5000)
    parser.add_argument("--evaluation-steps", type=int, default=1000)
    parser.add_argument("--record-every", type=int, default=50)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coupling", type=float, default=0.10)
    parser.add_argument("--low-harvest", type=float, default=0.002)
    parser.add_argument("--high-harvest", type=float, default=0.020)
    parser.add_argument("--metabolism", type=float, default=0.002)
    parser.add_argument("--reward-mode", choices=("harvest", "capped_harvest"), default="harvest")
    parser.add_argument("--initial-energy", type=float, default=1.0)
    parser.add_argument("--energy-capacity", type=float, default=1.0)
    parser.add_argument("--initial-resource-fraction", type=float, default=0.50)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.20)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)

    # Social information.
    parser.add_argument(
        "--social-mode",
        choices=SOCIAL_MODES,
        default="none",
        help=(
            "'none' reproduces the ecological baseline. 'fixed' enables "
            "previous-action social observation."
        ),
    )
    parser.add_argument(
        "--social-network",
        choices=SOCIAL_NETWORK_MODES,
        default="random_k",
        help=(
            "Initial social-network family. 'random_k' gives the directed "
            "fixed-attention network used by S1 and rewiring treatments. "
            "'ba' gives the fixed visibility-skew S2 control."
        ),
    )
    parser.add_argument(
        "--social-k",
        type=int,
        default=4,
        help="Number of information sources per observer in random_k.",
    )
    parser.add_argument(
        "--ba-m",
        type=int,
        default=2,
        help=(
            "Preferential-attachment parameter for --social-network ba. "
            "m=2 gives mean degree close to 4 at moderate/large N."
        ),
    )

    # Decentralized rewiring.
    parser.add_argument(
        "--rewiring",
        choices=REWIRING_MODES,
        default="none",
        help=(
            "'none' gives a fixed network; 'random_matched' gives the paired "
            "event-count-matched R0 control; 'prediction_error' gives "
            "adaptive rewiring."
        ),
    )
    parser.add_argument(
        "--matched-rewire-schedule",
        default=None,
        help=(
            "Path to an adaptive run's data/rewiring_schedule.csv. "
            "Required only for --rewiring random_matched."
        ),
    )
    parser.add_argument(
        "--rewire-theta",
        type=float,
        default=0.25,
        help="Probability of global rather than local replacement search.",
    )
    parser.add_argument(
        "--rewire-mu",
        type=float,
        default=0.10,
        help=(
            "Probability an eligible observer rewires. random_matched uses "
            "the paired adaptive event count instead."
        ),
    )
    parser.add_argument(
        "--rewire-every",
        type=int,
        default=50,
        help="Environment steps between rewiring checkpoints.",
    )
    parser.add_argument(
        "--rewire-threshold",
        type=float,
        default=0.25,
        help="Prediction-error threshold for adaptive rewiring.",
    )
    parser.add_argument(
        "--forecast-alpha",
        type=float,
        default=0.50,
        help="EWMA learning rate for local social forecasts.",
    )
    parser.add_argument(
        "--record-network-every",
        type=int,
        default=50,
        help="Interval for network/perception diagnostics.",
    )

    # Network evaluation.
    parser.add_argument(
        "--network-eval",
        choices=NETWORK_EVAL_MODES,
        default="frozen",
        help=(
            "'frozen' performs continuation plus fresh-ecology evaluation "
            "with carried and reset frozen networks. 'adaptive' keeps those "
            "frozen evaluations and additionally runs a fresh-ecology "
            "robustness evaluation in which the terminal random_k network "
            "continues to adapt while Q-tables remain frozen."
        ),
    )

    return parser


def condition_checkpoint(task):
    """Each process owns one condition; publish its checkpoint atomically."""
    scenario, population, replicate, args, path = task
    output = run_condition(scenario, population, replicate, args)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(output))
    temporary.replace(path)
    return scenario, population, replicate


def run_conditions(args, run_dir):
    workers = args.workers
    if workers == 0:
        workers = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 1
        )
    checkpoint_dir = run_dir / "conditions"
    checkpoint_dir.mkdir(exist_ok=True)
    signature = {k: v for k, v in vars(args).items() if k not in ("workers", "resume_conditions")}
    signature["reward_mode"] = getattr(args, "reward_mode", "harvest")
    signature["reward_definition"] = reward_identity(signature)
    signature_path = run_dir / "conditions_signature.json"
    if signature_path.exists():
        if json.loads(signature_path.read_text()) != signature:
            raise ValueError(
                "Cannot resume conditions with different scientific/reward configuration"
            )
    else:
        if any(checkpoint_dir.glob("*.json")):
            raise ValueError(
                "Cannot resume unverifiable legacy condition checkpoints; use a new run"
            )
        temporary = signature_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(signature, sort_keys=True))
        temporary.replace(signature_path)
    tasks = [
        (
            scenario,
            population,
            replicate,
            args,
            checkpoint_dir / f"{scenario}_N{population}_rep{replicate}.json",
        )
        for replicate in range(args.replicates)
        for scenario in args.scenarios
        for population in args.populations
    ]
    pending = [task for task in tasks if not task[-1].exists()]
    completed = len(tasks) - len(pending)
    print(
        f"CPU workers: {min(workers, len(pending))}; recovered conditions: {completed}", flush=True
    )
    if workers == 1:
        for task in pending:
            scenario, population, replicate = condition_checkpoint(task)
            completed += 1
            print(
                f"[{completed}/{len(tasks)}] {scenario} N={population} replicate={replicate}",
                flush=True,
            )
    elif pending:
        # Spawn avoids inheriting random state or library threads from the parent.
        with ProcessPoolExecutor(
            max_workers=min(workers, len(pending)), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [pool.submit(condition_checkpoint, task) for task in pending]
            for future in as_completed(futures):
                scenario, population, replicate = future.result()
                completed += 1
                print(
                    f"[{completed}/{len(tasks)}] {scenario} N={population} replicate={replicate}",
                    flush=True,
                )
    # Canonical order makes final CSVs independent of worker completion order.
    for task in tasks:
        yield json.loads(task[-1].read_text())


def main() -> None:
    args = build_parser().parse_args()
    validate_configuration(args)
    if args.workers < 0:
        raise ValueError("--workers must be non-negative.")

    run_dir = RESULTS_ROOT / args.run_name
    data_dir = run_dir / "data"
    figures_dir = run_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["reward_definition"] = reward_definition(args.reward_mode, args.metabolism)
    config["scenario_definitions"] = {name: SCENARIOS[name] for name in args.scenarios}
    config["learner_n_states"] = learner_state_count(args.social_mode)
    config["treatment"] = treatment_name(args)
    config["social_measurement_schema"] = "stage4_v1"
    config["social_network_semantics"] = (
        "none"
        if args.social_mode == "none"
        else ("directed_fixed_k" if args.social_network == "random_k" else "fixed_symmetric_ba")
    )
    config["network_evaluation"] = args.network_eval
    config["fresh_network"] = "carried_terminal_network" if args.social_mode == "fixed" else "none"
    config["fresh_network_evaluations"] = (
        ["carried_terminal_network", "reset_initial_network"]
        if args.social_mode == "fixed"
        else ["none"]
    )
    config["adaptive_network_start"] = (
        "terminal_training_network" if args.network_eval == "adaptive" else None
    )
    config["adaptive_forecast_start"] = (
        "terminal_training_forecast"
        if args.network_eval == "adaptive" and args.rewiring == "prediction_error"
        else None
    )
    config["evaluation_mode_semantics"] = {
        "continuation": ("trained ecology; terminal training graph; graph frozen"),
        "fresh_reset": (
            "fresh ecology; terminal training graph carried forward; graph "
            "frozen; mode name is part of the analysis schema"
        ),
        "fresh_reset_network": (
            "fresh ecology; exact initial training graph restored; graph frozen"
        ),
        "fresh_adaptive_network": (
            "fresh ecology; terminal training graph; Q frozen; network adapts"
        ),
    }
    config["matched_rewire_schedule_sha256"] = (
        sha256_file(args.matched_rewire_schedule) if args.matched_rewire_schedule else None
    )
    config["run_metadata"] = build_run_metadata()

    config_path = run_dir / "config.json"
    if config_path.exists():
        if not args.resume_conditions:
            raise ValueError("Run already exists; use --resume-conditions or a new run name.")
        previous = json.loads(config_path.read_text())
        previous["reward_definition"] = reward_identity(previous)
        previous.setdefault("reward_mode", "harvest")
        ignored = {"workers", "resume_conditions", "run_metadata"}
        if {k: v for k, v in previous.items() if k not in ignored} != {
            k: v for k, v in config.items() if k not in ignored
        } or previous["run_metadata"]["git_commit_sha"] != config["run_metadata"]["git_commit_sha"]:
            raise ValueError("Cannot resume: scientific configuration or code revision differs.")
        if (run_dir / "complete.json").exists():
            print(f"Already complete: {run_dir}")
            return

    with (run_dir / "config.json").open("w") as file:
        json.dump(config, file, indent=2)

    evaluation_rows: list[dict] = []
    training_rows: list[dict] = []
    evaluation_timeseries_rows: list[dict] = []
    policy_rows: list[dict] = []
    agent_policy_rows: list[dict] = []
    network_rows: list[dict] = []
    rewiring_schedule_rows: list[dict] = []
    agent_social_rows: list[dict] = []
    network_edge_rows: list[dict] = []

    total = len(args.scenarios) * len(args.populations) * args.replicates

    print(f"Running {total} Q-learning training conditions...")
    print(f"Treatment: {config['treatment']}")
    print(f"Social mode: {args.social_mode}")
    if args.social_mode == "fixed":
        print(f"Social network: {args.social_network}")
    print(f"Rewiring: {args.rewiring}")
    print(f"Network evaluation: {args.network_eval}")

    if args.social_mode == "fixed" and args.social_network == "random_k":
        print(f"Attention capacity k: {args.social_k}")
    if args.social_mode == "fixed" and args.social_network == "ba":
        print(f"BA attachment m: {args.ba_m}")
    if args.rewiring != "none":
        print(f"Search scope theta: {args.rewire_theta}")
    if args.rewiring == "random_matched":
        print(f"Matched schedule: {args.matched_rewire_schedule}")

    metadata = config["run_metadata"]
    print(f"Git commit: {metadata['git_commit_sha']}")
    print(f"Git worktree dirty: {metadata['git_worktree_dirty']}")

    if args.network_eval == "frozen":
        print(
            "Evaluation uses frozen social graphs; social runs include both "
            "carried-terminal and reset-initial fresh-network evaluations."
        )
    else:
        print(
            "Evaluation includes the frozen decomposition plus a fresh "
            "adaptive-network robustness condition."
        )

    for output in run_conditions(args, run_dir):
        evaluation_rows.extend(output["evaluation_summary"])
        training_rows.extend(output["training_timeseries"])
        evaluation_timeseries_rows.extend(output["evaluation_timeseries"])
        policy_rows.append(output["policy_summary"])
        agent_policy_rows.extend(output["agent_policies"])
        network_rows.extend(output["network_timeseries"])
        rewiring_schedule_rows.extend(output["rewiring_schedule"])
        agent_social_rows.extend(output["agent_social_summary"])
        network_edge_rows.extend(output["network_edges_checkpoints"])

    write_csv(data_dir / "evaluation_summary.csv", evaluation_rows)
    write_csv(data_dir / "training_timeseries.csv", training_rows)
    write_csv(data_dir / "evaluation_timeseries.csv", evaluation_timeseries_rows)
    write_csv(data_dir / "policy_summary.csv", policy_rows)
    write_csv(data_dir / "agent_policies.csv", agent_policy_rows)
    write_csv(data_dir / "network_timeseries.csv", network_rows)
    write_csv(data_dir / "rewiring_schedule.csv", rewiring_schedule_rows)
    write_csv(data_dir / "agent_social_summary.csv", agent_social_rows)
    write_csv(data_dir / "network_edges_checkpoints.csv", network_edge_rows)
    (run_dir / "complete.json").write_text(json.dumps({"conditions": total}))

    print("\nSimulation complete.")
    print(f"Results: {run_dir.resolve()}")
    print("Analyze this run with:")
    print(
        "python -m cognitive_tools.analysis "
        f"--run {shlex.quote(str(run_dir))} --analysis-name {shlex.quote(args.run_name)}"
    )


if __name__ == "__main__":
    main()
