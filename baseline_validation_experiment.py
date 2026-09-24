from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from Cognitive_tools import EcoEnv

from Cognitive_tools.model import (
    HIGH_EXTRACT,
    LOW_EXTRACT,
)
from Cognitive_tools.qlearning import (
    QLearningPolicy,
    STATE_NAMES,
    resource_state,
)

from qlearning_experiment import SCENARIOS, build_environment_maps


RESULTS_ROOT = Path("results") / "q_learning_baseline" / "experiments"


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
    total = values.sum()

    if len(values) == 0 or total <= 0.0:
        return 0.0

    n = len(values)
    index = np.arange(1, n + 1)

    value = (
        2.0 * np.sum(index * values) / (n * total)
        - (n + 1.0) / n
    )

    return float(max(0.0, value))


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0

    return float(
        -p * math.log2(p)
        - (1.0 - p) * math.log2(1.0 - p)
    )


def mean_or_nan(values) -> float:
    return float(np.mean(values)) if values else float("nan")


# ---------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------


def landscape_seed(replicate: int, base_seed: int) -> int:
    """Use the same landscape seed across scenarios within a replicate."""

    return base_seed + replicate


def agent_seed(replicate: int, base_seed: int) -> int:
    """Use the same agent-position seed across scenarios within a replicate."""

    return base_seed + 100_000 + replicate


def make_environment(
    scenario_name: str,
    population: int,
    replicate: int,
    max_steps: int,
    args,
):
    (
        capacity,
        recovery,
        equilibrium,
        regions,
    ) = build_environment_maps(
        scenario_name,
        width=args.width,
        height=args.height,
        seed=landscape_seed(replicate, args.seed),
    )

    env = EcoEnv(
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

    observations, _ = env.reset(
        seed=agent_seed(replicate, args.seed)
    )

    return env, observations, regions


# ---------------------------------------------------------------------
# Q-learning
# ---------------------------------------------------------------------


def make_learners(
    env: EcoEnv,
    replicate: int,
    args,
) -> dict[str, QLearningPolicy]:
    learners = {}

    for index, name in enumerate(env.possible_agents):
        learners[name] = QLearningPolicy(
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            seed=(
                args.seed
                + 200_000
                + 10_000 * replicate
                + index
            ),
        )

    return learners


def system_metrics(env: EcoEnv, actions: dict[str, int]) -> dict:
    agents = list(env.model.by_name.values())

    low_rate = float(
        np.mean(
            [
                action == COOPERATE
                for action in actions.values()
            ]
        )
    )

    return {
        "low_extraction_rate": low_rate,
        "collective_order": abs(2.0 * low_rate - 1.0),
        "action_entropy": binary_entropy(low_rate),
        "mean_resource_fraction": float(
            np.mean(env.model.resource / env.model.capacity)
        ),
        "total_resource": float(env.model.resource.sum()),
        # Primary welfare measures.
        "mean_reserve_welfare": float(
            np.mean([agent.reserve_welfare for agent in agents])
        ),
        "mean_need_satisfaction": float(
            np.mean([agent.need_satisfaction for agent in agents])
        ),
        "deprivation_rate": float(
            np.mean(
                [
                    agent.metabolic_shortfall > 1e-12
                    for agent in agents
                ]
            )
        ),
        "mean_metabolic_shortfall": float(
            np.mean([agent.metabolic_shortfall for agent in agents])
        ),
        # Secondary welfare/resilience state.
        "mean_energy": float(
            np.mean([agent.energy for agent in agents])
        ),
        # Primary distributional outcome.
        "wealth_gini": gini([agent.wealth for agent in agents]),
        "mean_wealth": float(
            np.mean([agent.wealth for agent in agents])
        ),
    }


def train_q_learning(
    scenario_name: str,
    population: int,
    replicate: int,
    args,
):
    total_steps = args.training_steps + args.evaluation_steps

    env, observations, regions = make_environment(
        scenario_name,
        population,
        replicate,
        total_steps,
        args,
    )

    learners = make_learners(env, replicate, args)
    timeseries = []

    for time in range(1, args.training_steps + 1):
        states = {
            name: resource_state(observations[name])
            for name in env.agents
        }

        actions = {
            name: learners[name].choose_action(
                states[name],
                explore=True,
            )
            for name in env.agents
        }

        (
            next_observations,
            rewards,
            terminations,
            truncations,
            _,
        ) = env.step(actions)

        for name in states:
            done = terminations[name] or truncations[name]

            next_state = (
                states[name]
                if done
                else resource_state(next_observations[name])
            )

            learners[name].update(
                states[name],
                actions[name],
                rewards[name],
                next_state,
                done=done,
            )
            learners[name].decay_exploration()

        if (
            time == 1
            or time % args.record_every == 0
            or time == args.training_steps
        ):
            timeseries.append(
                {
                    "scenario": scenario_name,
                    "population": population,
                    "replicate": replicate,
                    "time": time,
                    **system_metrics(env, actions),
                    "mean_epsilon": float(
                        np.mean(
                            [
                                learner.epsilon
                                for learner in learners.values()
                            ]
                        )
                    ),
                }
            )

        observations = next_observations

    return env, observations, regions, learners, timeseries


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def empty_state_counts() -> dict[str, np.ndarray]:
    return {
        "visits": np.zeros(3, dtype=int),
        "low_actions": np.zeros(3, dtype=int),
    }


def update_state_counts(
    counts: dict[str, np.ndarray],
    states: dict[str, int],
    actions: dict[str, int],
) -> None:
    for name, state in states.items():
        counts["visits"][state] += 1

        if actions[name] == COOPERATE:
            counts["low_actions"][state] += 1


def summarize_state_counts(counts: dict[str, np.ndarray]) -> dict:
    visits = counts["visits"]
    low_actions = counts["low_actions"]
    total = int(visits.sum())

    result = {}

    for index, state_name in enumerate(STATE_NAMES):
        state_visits = int(visits[index])

        result[f"state_occupancy_{state_name}"] = (
            state_visits / total
            if total > 0
            else float("nan")
        )

        result[f"low_given_{state_name}"] = (
            int(low_actions[index]) / state_visits
            if state_visits > 0
            else float("nan")
        )

    return result


def action_rule(
    strategy: str,
    observations: dict,
    learners: dict[str, QLearningPolicy] | None,
    rng: np.random.Generator,
):
    states = {
        name: resource_state(observation)
        for name, observation in observations.items()
    }

    if strategy == "q_learning":
        if learners is None:
            raise ValueError("q_learning evaluation requires learners.")

        actions = {
            name: learners[name].choose_action(
                states[name],
                explore=False,
            )
            for name in observations
        }

    elif strategy == "always_low":
        actions = {
            name: COOPERATE
            for name in observations
        }

    elif strategy == "always_high":
        actions = {
            name: DEFECT
            for name in observations
        }

    elif strategy == "random_50":
        actions = {
            name: int(rng.integers(2))
            for name in observations
        }

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
    args,
):
    # Same random-policy stream across ecological scenarios for matched runs.
    rng = np.random.default_rng(
        args.seed
        + 400_000
        + 10_000 * replicate
        + population
    )

    state_counts = empty_state_counts()
    snapshots = []
    timeseries = []

    for time in range(1, args.evaluation_steps + 1):
        states, actions = action_rule(
            strategy,
            observations,
            learners,
            rng,
        )

        update_state_counts(
            state_counts,
            states,
            actions,
        )

        (
            next_observations,
            _,
            _,
            truncations,
            _,
        ) = env.step(actions)

        metrics = system_metrics(env, actions)
        snapshots.append(metrics)

        if (
            time == 1
            or time % args.record_every == 0
            or time == args.evaluation_steps
        ):
            timeseries.append(
                {
                    "scenario": scenario_name,
                    "population": population,
                    "replicate": replicate,
                    "strategy": strategy,
                    "evaluation_mode": evaluation_mode,
                    "time": time,
                    **metrics,
                }
            )

        observations = next_observations

        if any(truncations.values()):
            break

    summary = {
        "scenario": scenario_name,
        "scenario_label": SCENARIOS[scenario_name]["label"],
        "scenario_group": SCENARIOS[scenario_name]["group"],
        "population": population,
        "replicate": replicate,
        "strategy": strategy,
        "evaluation_mode": evaluation_mode,
        "steps_evaluated": len(snapshots),
    }

    metric_names = list(snapshots[0].keys()) if snapshots else []

    for metric in metric_names:
        values = [row[metric] for row in snapshots]
        summary[f"eval_mean_{metric}"] = mean_or_nan(values)
        summary[f"final_{metric}"] = (
            float(values[-1])
            if values
            else float("nan")
        )

    summary.update(summarize_state_counts(state_counts))

    return summary, timeseries


# ---------------------------------------------------------------------
# Learned-policy heterogeneity
# ---------------------------------------------------------------------


def policy_diagnostics(
    env: EcoEnv,
    regions: np.ndarray,
    learners: dict[str, QLearningPolicy],
    scenario_name: str,
    population: int,
    replicate: int,
):
    agent_rows = []
    policies = {}

    for name, learner in learners.items():
        agent = env.model.by_name[name]
        x = int(agent.position[0])
        y = int(agent.position[1])
        policy = learner.greedy_policy()

        policies[name] = policy

        agent_rows.append(
            {
                "scenario": scenario_name,
                "population": population,
                "replicate": replicate,
                "agent": name,
                "x": x,
                "y": y,
                "region": str(regions[y, x]),
                "policy_scarce": "L" if policy[0] == COOPERATE else "H",
                "policy_moderate": "L" if policy[1] == COOPERATE else "H",
                "policy_abundant": "L" if policy[2] == COOPERATE else "H",
                "q_scarce_low": float(learner.q[0, COOPERATE]),
                "q_scarce_high": float(learner.q[0, DEFECT]),
                "q_moderate_low": float(learner.q[1, COOPERATE]),
                "q_moderate_high": float(learner.q[1, DEFECT]),
                "q_abundant_low": float(learner.q[2, COOPERATE]),
                "q_abundant_high": float(learner.q[2, DEFECT]),
            }
        )

    policy_list = list(policies.values())
    pairwise = []

    for first, second in combinations(policy_list, 2):
        pairwise.append(
            float(
                np.mean(
                    np.asarray(first)
                    != np.asarray(second)
                )
            )
        )

    counts = Counter(policy_list)
    probabilities = np.asarray(list(counts.values()), dtype=float)
    probabilities /= probabilities.sum()

    entropy = float(
        -np.sum(
            [
                p * math.log2(p)
                for p in probabilities
                if p > 0.0
            ]
        )
    )

    max_types = min(len(policy_list), 2 ** len(STATE_NAMES))

    if max_types > 1:
        entropy /= math.log2(max_types)
    else:
        entropy = 0.0

    summary = {
        "scenario": scenario_name,
        "population": population,
        "replicate": replicate,
        "unique_policy_count": len(counts),
        "policy_entropy": entropy,
        "policy_hamming_mean": (
            float(np.mean(pairwise))
            if pairwise
            else 0.0
        ),
    }

    for index, state_name in enumerate(STATE_NAMES):
        summary[f"policy_low_{state_name}"] = float(
            np.mean(
                [
                    policy[index] == COOPERATE
                    for policy in policy_list
                ]
            )
        )

    return summary, agent_rows


# ---------------------------------------------------------------------
# One condition
# ---------------------------------------------------------------------


def run_condition(
    scenario_name: str,
    population: int,
    replicate: int,
    args,
):
    (
        trained_env,
        trained_observations,
        regions,
        learners,
        training_timeseries,
    ) = train_q_learning(
        scenario_name,
        population,
        replicate,
        args,
    )

    policy_summary, agent_policy_rows = policy_diagnostics(
        trained_env,
        regions,
        learners,
        scenario_name,
        population,
        replicate,
    )

    # A. Continue from the ecology produced during training.
    continuation_summary, continuation_ts = evaluate_policy(
        trained_env,
        trained_observations,
        scenario_name=scenario_name,
        population=population,
        replicate=replicate,
        strategy="q_learning",
        evaluation_mode="continuation",
        learners=learners,
        args=args,
    )

    # B. Evaluate the same learned policy from a fresh common initial state.
    fresh_env, fresh_obs, _ = make_environment(
        scenario_name,
        population,
        replicate,
        args.evaluation_steps,
        args,
    )

    fresh_summary, fresh_ts = evaluate_policy(
        fresh_env,
        fresh_obs,
        scenario_name=scenario_name,
        population=population,
        replicate=replicate,
        strategy="q_learning",
        evaluation_mode="fresh_reset",
        learners=learners,
        args=args,
    )

    # C. Fixed-policy controls from the same fresh initial state.
    control_summaries = []
    control_timeseries = []

    for strategy in (
        "always_low",
        "always_high",
        "random_50",
    ):
        control_env, control_obs, _ = make_environment(
            scenario_name,
            population,
            replicate,
            args.evaluation_steps,
            args,
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
            args=args,
        )

        control_summaries.append(summary)
        control_timeseries.extend(timeseries)

    return {
        "evaluation_summary": [
            continuation_summary,
            fresh_summary,
            *control_summaries,
        ],
        "training_timeseries": training_timeseries,
        "evaluation_timeseries": [
            *continuation_ts,
            *fresh_ts,
            *control_timeseries,
        ],
        "policy_summary": policy_summary,
        "agent_policies": agent_policy_rows,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline validation: independent ecological Q-learning with "
            "welfare, inequality, state occupancy, fresh-reset evaluation, "
            "and fixed-policy controls."
        )
    )

    parser.add_argument("--run-name", default="baseline_validation_v1")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--populations", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--replicates", type=int, default=20)
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
    parser.add_argument("--initial-energy", type=float, default=1.0)
    parser.add_argument("--energy-capacity", type=float, default=1.0)
    parser.add_argument("--initial-resource-fraction", type=float, default=0.50)

    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.20)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)

    args = parser.parse_args()

    for scenario in args.scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")

    run_dir = RESULTS_ROOT / args.run_name
    data_dir = run_dir / "data"
    figures_dir = run_dir / "figures"

    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["scenario_definitions"] = {
        name: SCENARIOS[name]
        for name in args.scenarios
    }

    with (run_dir / "config.json").open("w") as file:
        json.dump(config, file, indent=2)

    evaluation_rows = []
    training_rows = []
    evaluation_timeseries_rows = []
    policy_rows = []
    agent_policy_rows = []

    total = (
        len(args.scenarios)
        * len(args.populations)
        * args.replicates
    )

    completed = 0

    print(f"Running {total} Q-learning training conditions...")
    print(
        "Each condition also receives continuation evaluation, fresh-reset "
        "evaluation, and three fixed-policy controls."
    )

    for replicate in range(args.replicates):
        for scenario in args.scenarios:
            for population in args.populations:
                output = run_condition(
                    scenario,
                    population,
                    replicate,
                    args,
                )

                evaluation_rows.extend(output["evaluation_summary"])
                training_rows.extend(output["training_timeseries"])
                evaluation_timeseries_rows.extend(output["evaluation_timeseries"])
                policy_rows.append(output["policy_summary"])
                agent_policy_rows.extend(output["agent_policies"])

                completed += 1
                print(
                    f"[{completed}/{total}] "
                    f"{scenario} N={population} replicate={replicate}"
                )

    write_csv(data_dir / "evaluation_summary.csv", evaluation_rows)
    write_csv(data_dir / "training_timeseries.csv", training_rows)
    write_csv(data_dir / "evaluation_timeseries.csv", evaluation_timeseries_rows)
    write_csv(data_dir / "policy_summary.csv", policy_rows)
    write_csv(data_dir / "agent_policies.csv", agent_policy_rows)

    print("\nSimulation complete.")
    print(f"Results: {run_dir.resolve()}")
    print("Generate figures with:")
    print(
        "python baseline_validation_figures.py "
        f"--run-name {args.run_name}"
    )


if __name__ == "__main__":
    main()
