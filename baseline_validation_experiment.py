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
from Cognitive_tools.social import (
    SOCIAL_STATE_NAMES,
    init_random_attention,
    joint_state,
    observed_low_fraction,
    social_bin,
    visibility_counts,
)
from qlearning_experiment import (
    SCENARIOS,
    build_environment_maps,
)


RESULTS_ROOT = (
    Path("results")
    / "q_learning_baseline"
    / "experiments"
)

SOCIAL_MODES = (
    "none",
    "fixed",
)


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print(
        f"Saved {path}"
    )


def gini(
    values,
) -> float:
    values = np.sort(
        np.asarray(
            values,
            dtype=float,
        )
    )

    total = values.sum()

    if (
        len(values) == 0
        or total <= 0.0
    ):
        return 0.0

    n = len(
        values
    )

    index = np.arange(
        1,
        n + 1,
    )

    value = (
        2.0
        * np.sum(
            index
            * values
        )
        / (
            n
            * total
        )
        - (
            n + 1.0
        )
        / n
    )

    return float(
        max(
            0.0,
            value,
        )
    )


def binary_entropy(
    p: float,
) -> float:
    if (
        p <= 0.0
        or p >= 1.0
    ):
        return 0.0

    return float(
        -p
        * math.log2(
            p
        )
        - (
            1.0
            - p
        )
        * math.log2(
            1.0
            - p
        )
    )


def mean_or_nan(
    values,
) -> float:
    if not values:
        return float(
            "nan"
        )

    return float(
        np.mean(
            values
        )
    )


# ---------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------


def landscape_seed(
    replicate: int,
    base_seed: int,
) -> int:
    """
    Use the same landscape seed across ecological scenarios within a
    replicate.
    """

    return (
        base_seed
        + replicate
    )


def agent_seed(
    replicate: int,
    base_seed: int,
) -> int:
    """
    Use the same agent-position seed across ecological scenarios within a
    replicate.
    """

    return (
        base_seed
        + 100_000
        + replicate
    )


def social_seed(
    replicate: int,
    population: int,
    base_seed: int,
) -> int:
    """
    Dedicated initial social-network seed.

    It does not depend on ecological scenario, so matched ecological
    treatments receive the same social graph for a given population and
    replicate.
    """

    return (
        base_seed
        + 300_000
        + 10_000
        * replicate
        + population
    )


# ---------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------


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
        seed=landscape_seed(
            replicate,
            args.seed,
        ),
    )

    env = EcoEnv(
        width=args.width,
        height=args.height,
        n_agents=population,
        max_steps=max_steps,
        regeneration_rate=recovery,
        equilibrium_fraction=equilibrium,
        coupling_rate=args.coupling,
        cooperative_harvest_amount=(
            args.low_harvest
        ),
        defective_harvest_amount=(
            args.high_harvest
        ),
        metabolism_rate=args.metabolism,
        initial_energy=args.initial_energy,
        energy_capacity=args.energy_capacity,
        initial_resource_fraction=(
            args.initial_resource_fraction
        ),
        capacity_map=capacity,
    )

    observations, _ = env.reset(
        seed=agent_seed(
            replicate,
            args.seed,
        )
    )

    return (
        env,
        observations,
        regions,
    )


# ---------------------------------------------------------------------
# Social observation
# ---------------------------------------------------------------------


def make_social_sources(
    env: EcoEnv,
    replicate: int,
    args,
) -> dict[str, list[str]] | None:
    """
    Create the fixed social-information network when social observation
    is enabled.

    The ecological baseline returns None and therefore remains unchanged.
    """

    if (
        args.social_mode
        == "none"
    ):
        return None

    if (
        args.social_mode
        != "fixed"
    ):
        raise ValueError(
            f"Unknown social mode: "
            f"{args.social_mode}"
        )

    rng = np.random.default_rng(
        social_seed(
            replicate,
            len(
                env.possible_agents
            ),
            args.seed,
        )
    )

    return init_random_attention(
        list(
            env.possible_agents
        ),
        args.social_k,
        rng,
    )


def learner_state_count(
    social_mode: str,
) -> int:
    if (
        social_mode
        == "none"
    ):
        return 3

    if (
        social_mode
        == "fixed"
    ):
        return 9

    raise ValueError(
        f"Unknown social mode: "
        f"{social_mode}"
    )


def encode_states(
    observations: dict,
    *,
    social_mode: str,
    sources: (
        dict[
            str,
            list[str],
        ]
        | None
    ),
    previous_actions: (
        dict[
            str,
            int,
        ]
        | None
    ),
) -> dict[
    str,
    int,
]:
    """
    Encode learner states.

    Ecological baseline:

        state = ecological state
        -> 3 states

    Fixed social observation:

        state
        =
        3 * ecological_state
        + social_state

        -> 9 states

    The social signal for action a_t is based on observed actions from
    t-1.
    """

    ecological_states = {
        name: resource_state(
            observation
        )
        for (
            name,
            observation,
        )
        in observations.items()
    }

    if (
        social_mode
        == "none"
    ):
        return ecological_states

    if (
        social_mode
        != "fixed"
    ):
        raise ValueError(
            f"Unknown social mode: "
            f"{social_mode}"
        )

    if (
        sources
        is None
    ):
        raise ValueError(
            "Fixed social observation "
            "requires sources."
        )

    states = {}

    for (
        name,
        ecological_state,
    ) in ecological_states.items():
        low_fraction = (
            observed_low_fraction(
                name,
                sources,
                previous_actions,
            )
        )

        social_state = (
            social_bin(
                low_fraction
            )
        )

        states[
            name
        ] = joint_state(
            ecological_state,
            social_state,
        )

    return states


# ---------------------------------------------------------------------
# Q-learning
# ---------------------------------------------------------------------


def make_learners(
    env: EcoEnv,
    replicate: int,
    args,
) -> dict[
    str,
    QLearningPolicy,
]:
    learners = {}

    n_states = (
        learner_state_count(
            args.social_mode
        )
    )

    for index, name in enumerate(
        env.possible_agents
    ):
        learners[
            name
        ] = QLearningPolicy(
            n_states=n_states,
            n_actions=2,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=(
                args.epsilon_min
            ),
            epsilon_decay=(
                args.epsilon_decay
            ),
            seed=(
                args.seed
                + 200_000
                + 10_000
                * replicate
                + index
            ),
        )

    return learners


def system_metrics(
    env: EcoEnv,
    actions: dict[
        str,
        int,
    ],
    *,
    sources: (
        dict[
            str,
            list[str],
        ]
        | None
    ) = None,
) -> dict:
    agents = list(
        env.model.by_name.values()
    )

    low_rate = float(
        np.mean(
            [
                action
                == LOW_EXTRACT
                for action
                in actions.values()
            ]
        )
    )

    if (
        sources
        is None
    ):
        mean_social_low_fraction = (
            float(
                "nan"
            )
        )

    else:
        mean_social_low_fraction = float(
            np.mean(
                [
                    observed_low_fraction(
                        name,
                        sources,
                        actions,
                    )
                    for name
                    in actions
                ]
            )
        )

    return {
        "low_extraction_rate": (
            low_rate
        ),
        "collective_order": abs(
            2.0
            * low_rate
            - 1.0
        ),
        "action_entropy": (
            binary_entropy(
                low_rate
            )
        ),
        "mean_resource_fraction": float(
            np.mean(
                env.model.resource
                / env.model.capacity
            )
        ),
        "total_resource": float(
            env.model.resource.sum()
        ),
        "mean_reserve_welfare": float(
            np.mean(
                [
                    agent.reserve_welfare
                    for agent
                    in agents
                ]
            )
        ),
        "mean_need_satisfaction": float(
            np.mean(
                [
                    agent.need_satisfaction
                    for agent
                    in agents
                ]
            )
        ),
        "deprivation_rate": float(
            np.mean(
                [
                    agent.metabolic_shortfall
                    > 1e-12
                    for agent
                    in agents
                ]
            )
        ),
        "mean_metabolic_shortfall": float(
            np.mean(
                [
                    agent.metabolic_shortfall
                    for agent
                    in agents
                ]
            )
        ),
        "mean_energy": float(
            np.mean(
                [
                    agent.energy
                    for agent
                    in agents
                ]
            )
        ),
        "wealth_gini": gini(
            [
                agent.wealth
                for agent
                in agents
            ]
        ),
        "mean_wealth": float(
            np.mean(
                [
                    agent.wealth
                    for agent
                    in agents
                ]
            )
        ),
        "mean_social_low_fraction": (
            mean_social_low_fraction
        ),
    }


def train_q_learning(
    scenario_name: str,
    population: int,
    replicate: int,
    args,
):
    total_steps = (
        args.training_steps
        + args.evaluation_steps
    )

    (
        env,
        observations,
        regions,
    ) = make_environment(
        scenario_name,
        population,
        replicate,
        total_steps,
        args,
    )

    sources = make_social_sources(
        env,
        replicate,
        args,
    )

    learners = make_learners(
        env,
        replicate,
        args,
    )

    timeseries = []

    previous_actions = None

    for time in range(
        1,
        args.training_steps
        + 1,
    ):
        states = encode_states(
            observations,
            social_mode=(
                args.social_mode
            ),
            sources=sources,
            previous_actions=(
                previous_actions
            ),
        )

        actions = {
            name: learners[
                name
            ].choose_action(
                states[
                    name
                ],
                explore=True,
            )
            for name
            in env.agents
        }

        (
            next_observations,
            rewards,
            terminations,
            truncations,
            _,
        ) = env.step(
            actions
        )

        # The next social state uses actions at time t.
        next_states = encode_states(
            next_observations,
            social_mode=(
                args.social_mode
            ),
            sources=sources,
            previous_actions=actions,
        )

        for name in states:
            done = (
                terminations[
                    name
                ]
                or truncations[
                    name
                ]
            )

            if done:
                learner_next_state = (
                    states[
                        name
                    ]
                )

            else:
                learner_next_state = (
                    next_states[
                        name
                    ]
                )

            learners[
                name
            ].update(
                states[
                    name
                ],
                actions[
                    name
                ],
                rewards[
                    name
                ],
                learner_next_state,
                done=done,
            )

            learners[
                name
            ].decay_exploration()

        if (
            time == 1
            or time
            % args.record_every
            == 0
            or time
            == args.training_steps
        ):
            timeseries.append(
                {
                    "scenario": (
                        scenario_name
                    ),
                    "population": (
                        population
                    ),
                    "replicate": (
                        replicate
                    ),
                    "social_mode": (
                        args.social_mode
                    ),
                    "time": time,
                    **system_metrics(
                        env,
                        actions,
                        sources=sources,
                    ),
                    "mean_epsilon": float(
                        np.mean(
                            [
                                learner.epsilon
                                for learner
                                in learners.values()
                            ]
                        )
                    ),
                }
            )

        previous_actions = (
            actions.copy()
        )

        observations = (
            next_observations
        )

    return (
        env,
        observations,
        regions,
        learners,
        timeseries,
        sources,
        previous_actions,
    )


# ---------------------------------------------------------------------
# State-count diagnostics
# ---------------------------------------------------------------------


def empty_state_counts(
    social_mode: str,
) -> dict[
    str,
    np.ndarray,
]:
    if (
        social_mode
        == "none"
    ):
        social_states = 1

    elif (
        social_mode
        == "fixed"
    ):
        social_states = 3

    else:
        raise ValueError(
            f"Unknown social mode: "
            f"{social_mode}"
        )

    shape = (
        3,
        social_states,
    )

    return {
        "visits": np.zeros(
            shape,
            dtype=int,
        ),
        "low_actions": np.zeros(
            shape,
            dtype=int,
        ),
    }


def decode_state(
    state: int,
    social_mode: str,
) -> tuple[
    int,
    int,
]:
    if (
        social_mode
        == "none"
    ):
        return (
            int(
                state
            ),
            0,
        )

    if (
        social_mode
        == "fixed"
    ):
        return (
            int(
                state
            )
            // 3,
            int(
                state
            )
            % 3,
        )

    raise ValueError(
        f"Unknown social mode: "
        f"{social_mode}"
    )


def update_state_counts(
    counts: dict[
        str,
        np.ndarray,
    ],
    states: dict[
        str,
        int,
    ],
    actions: dict[
        str,
        int,
    ],
    *,
    social_mode: str,
) -> None:
    for (
        name,
        state,
    ) in states.items():
        (
            ecological_state,
            social_state,
        ) = decode_state(
            state,
            social_mode,
        )

        counts[
            "visits"
        ][
            ecological_state,
            social_state,
        ] += 1

        if (
            actions[
                name
            ]
            == LOW_EXTRACT
        ):
            counts[
                "low_actions"
            ][
                ecological_state,
                social_state,
            ] += 1


def summarize_state_counts(
    counts: dict[
        str,
        np.ndarray,
    ],
    *,
    social_mode: str,
) -> dict:
    visits = counts[
        "visits"
    ]

    low_actions = counts[
        "low_actions"
    ]

    total = int(
        visits.sum()
    )

    result = {}

    # Preserve the existing ecological marginal diagnostics.
    for (
        ecological_index,
        ecological_name,
    ) in enumerate(
        STATE_NAMES
    ):
        ecological_visits = int(
            visits[
                ecological_index,
                :,
            ].sum()
        )

        ecological_low = int(
            low_actions[
                ecological_index,
                :,
            ].sum()
        )

        result[
            f"state_occupancy_{ecological_name}"
        ] = (
            ecological_visits
            / total
            if total > 0
            else float(
                "nan"
            )
        )

        result[
            f"low_given_{ecological_name}"
        ] = (
            ecological_low
            / ecological_visits
            if ecological_visits > 0
            else float(
                "nan"
            )
        )

    if (
        social_mode
        == "fixed"
    ):
        for (
            social_index,
            social_name,
        ) in enumerate(
            SOCIAL_STATE_NAMES
        ):
            social_visits = int(
                visits[
                    :,
                    social_index,
                ].sum()
            )

            result[
                f"social_occupancy_{social_name}"
            ] = (
                social_visits
                / total
                if total > 0
                else float(
                    "nan"
                )
            )

        for (
            ecological_index,
            ecological_name,
        ) in enumerate(
            STATE_NAMES
        ):
            for (
                social_index,
                social_name,
            ) in enumerate(
                SOCIAL_STATE_NAMES
            ):
                joint_visits = int(
                    visits[
                        ecological_index,
                        social_index,
                    ]
                )

                joint_low = int(
                    low_actions[
                        ecological_index,
                        social_index,
                    ]
                )

                result[
                    (
                        "joint_occupancy_"
                        f"{ecological_name}_"
                        f"{social_name}"
                    )
                ] = (
                    joint_visits
                    / total
                    if total > 0
                    else float(
                        "nan"
                    )
                )

                result[
                    (
                        "low_given_"
                        f"{ecological_name}_"
                        f"{social_name}"
                    )
                ] = (
                    joint_low
                    / joint_visits
                    if joint_visits > 0
                    else float(
                        "nan"
                    )
                )

    return result


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def action_rule(
    strategy: str,
    observations: dict,
    learners: (
        dict[
            str,
            QLearningPolicy,
        ]
        | None
    ),
    rng: np.random.Generator,
    *,
    social_mode: str = "none",
    sources: (
        dict[
            str,
            list[str],
        ]
        | None
    ) = None,
    previous_actions: (
        dict[
            str,
            int,
        ]
        | None
    ) = None,
):
    states = encode_states(
        observations,
        social_mode=(
            social_mode
        ),
        sources=sources,
        previous_actions=(
            previous_actions
        ),
    )

    if (
        strategy
        == "q_learning"
    ):
        if learners is None:
            raise ValueError(
                "q_learning evaluation "
                "requires learners."
            )

        actions = {
            name: learners[
                name
            ].choose_action(
                states[
                    name
                ],
                explore=False,
            )
            for name
            in observations
        }

    elif (
        strategy
        == "always_low"
    ):
        actions = {
            name: LOW_EXTRACT
            for name
            in observations
        }

    elif (
        strategy
        == "always_high"
    ):
        actions = {
            name: HIGH_EXTRACT
            for name
            in observations
        }

    elif (
        strategy
        == "random_50"
    ):
        actions = {
            name: int(
                rng.integers(
                    2
                )
            )
            for name
            in observations
        }

    else:
        raise ValueError(
            f"Unknown strategy: "
            f"{strategy}"
        )

    return (
        states,
        actions,
    )


def evaluate_policy(
    env: EcoEnv,
    observations: dict,
    *,
    scenario_name: str,
    population: int,
    replicate: int,
    strategy: str,
    evaluation_mode: str,
    learners: (
        dict[
            str,
            QLearningPolicy,
        ]
        | None
    ),
    sources: (
        dict[
            str,
            list[str],
        ]
        | None
    ),
    previous_actions: (
        dict[
            str,
            int,
        ]
        | None
    ),
    args,
):
    rng = np.random.default_rng(
        args.seed
        + 400_000
        + 10_000
        * replicate
        + population
    )

    state_counts = (
        empty_state_counts(
            args.social_mode
        )
    )

    snapshots = []
    timeseries = []

    for time in range(
        1,
        args.evaluation_steps
        + 1,
    ):
        (
            states,
            actions,
        ) = action_rule(
            strategy,
            observations,
            learners,
            rng,
            social_mode=(
                args.social_mode
            ),
            sources=sources,
            previous_actions=(
                previous_actions
            ),
        )

        update_state_counts(
            state_counts,
            states,
            actions,
            social_mode=(
                args.social_mode
            ),
        )

        (
            next_observations,
            _,
            _,
            truncations,
            _,
        ) = env.step(
            actions
        )

        metrics = system_metrics(
            env,
            actions,
            sources=sources,
        )

        snapshots.append(
            metrics
        )

        if (
            time == 1
            or time
            % args.record_every
            == 0
            or time
            == args.evaluation_steps
        ):
            timeseries.append(
                {
                    "scenario": (
                        scenario_name
                    ),
                    "population": (
                        population
                    ),
                    "replicate": (
                        replicate
                    ),
                    "social_mode": (
                        args.social_mode
                    ),
                    "strategy": (
                        strategy
                    ),
                    "evaluation_mode": (
                        evaluation_mode
                    ),
                    "time": time,
                    **metrics,
                }
            )

        previous_actions = (
            actions.copy()
        )

        observations = (
            next_observations
        )

        if any(
            truncations.values()
        ):
            break

    summary = {
        "scenario": (
            scenario_name
        ),
        "scenario_label": (
            SCENARIOS[
                scenario_name
            ][
                "label"
            ]
        ),
        "scenario_group": (
            SCENARIOS[
                scenario_name
            ][
                "group"
            ]
        ),
        "population": (
            population
        ),
        "replicate": (
            replicate
        ),
        "social_mode": (
            args.social_mode
        ),
        "strategy": (
            strategy
        ),
        "evaluation_mode": (
            evaluation_mode
        ),
        "steps_evaluated": len(
            snapshots
        ),
    }

    metric_names = (
        list(
            snapshots[
                0
            ].keys()
        )
        if snapshots
        else []
    )

    for metric in metric_names:
        values = [
            row[
                metric
            ]
            for row
            in snapshots
        ]

        finite_values = [
            value
            for value
            in values
            if np.isfinite(
                value
            )
        ]

        summary[
            f"eval_mean_{metric}"
        ] = (
            mean_or_nan(
                finite_values
            )
        )

        summary[
            f"final_{metric}"
        ] = (
            float(
                values[
                    -1
                ]
            )
            if values
            else float(
                "nan"
            )
        )

    summary.update(
        summarize_state_counts(
            state_counts,
            social_mode=(
                args.social_mode
            ),
        )
    )

    return (
        summary,
        timeseries,
    )


# ---------------------------------------------------------------------
# Learned-policy diagnostics
# ---------------------------------------------------------------------


def policy_diagnostics(
    env: EcoEnv,
    regions: np.ndarray,
    learners: dict[
        str,
        QLearningPolicy,
    ],
    scenario_name: str,
    population: int,
    replicate: int,
    *,
    social_mode: str,
    sources: (
        dict[
            str,
            list[str],
        ]
        | None
    ),
):
    agent_rows = []

    policies = {
        name: (
            learner.greedy_policy()
        )
        for (
            name,
            learner,
        )
        in learners.items()
    }

    if (
        sources
        is None
    ):
        visibility = {
            name: 0
            for name
            in learners
        }

    else:
        visibility = (
            visibility_counts(
                sources
            )
        )

    for (
        name,
        learner,
    ) in learners.items():
        agent = (
            env.model.by_name[
                name
            ]
        )

        x = int(
            agent.position[
                0
            ]
        )

        y = int(
            agent.position[
                1
            ]
        )

        policy = (
            policies[
                name
            ]
        )

        row = {
            "scenario": (
                scenario_name
            ),
            "population": (
                population
            ),
            "replicate": (
                replicate
            ),
            "social_mode": (
                social_mode
            ),
            "agent": name,
            "x": x,
            "y": y,
            "region": str(
                regions[
                    y,
                    x,
                ]
            ),
            "visibility_degree": int(
                visibility[
                    name
                ]
            ),
            "attention_sources": (
                ""
                if sources is None
                else "|".join(
                    sources[
                        name
                    ]
                )
            ),
        }

        if (
            social_mode
            == "none"
        ):
            for (
                ecological_index,
                ecological_name,
            ) in enumerate(
                STATE_NAMES
            ):
                row[
                    f"policy_{ecological_name}"
                ] = (
                    "L"
                    if policy[
                        ecological_index
                    ]
                    == LOW_EXTRACT
                    else "H"
                )

                row[
                    f"q_{ecological_name}_low"
                ] = float(
                    learner.q[
                        ecological_index,
                        LOW_EXTRACT,
                    ]
                )

                row[
                    f"q_{ecological_name}_high"
                ] = float(
                    learner.q[
                        ecological_index,
                        HIGH_EXTRACT,
                    ]
                )

        else:
            for (
                ecological_index,
                ecological_name,
            ) in enumerate(
                STATE_NAMES
            ):
                for (
                    social_index,
                    social_name,
                ) in enumerate(
                    SOCIAL_STATE_NAMES
                ):
                    state = joint_state(
                        ecological_index,
                        social_index,
                    )

                    label = (
                        f"{ecological_name}_"
                        f"{social_name}"
                    )

                    row[
                        f"policy_{label}"
                    ] = (
                        "L"
                        if policy[
                            state
                        ]
                        == LOW_EXTRACT
                        else "H"
                    )

                    row[
                        f"q_{label}_low"
                    ] = float(
                        learner.q[
                            state,
                            LOW_EXTRACT,
                        ]
                    )

                    row[
                        f"q_{label}_high"
                    ] = float(
                        learner.q[
                            state,
                            HIGH_EXTRACT,
                        ]
                    )

        agent_rows.append(
            row
        )

    policy_list = list(
        policies.values()
    )

    pairwise = []

    for (
        first,
        second,
    ) in combinations(
        policy_list,
        2,
    ):
        pairwise.append(
            float(
                np.mean(
                    np.asarray(
                        first
                    )
                    != np.asarray(
                        second
                    )
                )
            )
        )

    counts = Counter(
        policy_list
    )

    probabilities = np.asarray(
        list(
            counts.values()
        ),
        dtype=float,
    )

    probabilities /= (
        probabilities.sum()
    )

    entropy = float(
        -np.sum(
            [
                probability
                * math.log2(
                    probability
                )
                for probability
                in probabilities
                if probability > 0.0
            ]
        )
    )

    policy_length = (
        len(
            policy_list[
                0
            ]
        )
        if policy_list
        else 0
    )

    max_types = min(
        len(
            policy_list
        ),
        2
        ** policy_length,
    )

    if (
        max_types > 1
    ):
        entropy /= math.log2(
            max_types
        )

    else:
        entropy = 0.0

    summary = {
        "scenario": (
            scenario_name
        ),
        "population": (
            population
        ),
        "replicate": (
            replicate
        ),
        "social_mode": (
            social_mode
        ),
        "unique_policy_count": len(
            counts
        ),
        "policy_entropy": (
            entropy
        ),
        "policy_hamming_mean": (
            float(
                np.mean(
                    pairwise
                )
            )
            if pairwise
            else 0.0
        ),
    }

    if (
        sources
        is None
    ):
        summary[
            "visibility_gini"
        ] = float(
            "nan"
        )

        summary[
            "max_visibility_degree"
        ] = float(
            "nan"
        )

    else:
        visibility_values = list(
            visibility.values()
        )

        summary[
            "visibility_gini"
        ] = gini(
            visibility_values
        )

        summary[
            "max_visibility_degree"
        ] = int(
            max(
                visibility_values
            )
        )

    if (
        social_mode
        == "none"
    ):
        for (
            ecological_index,
            ecological_name,
        ) in enumerate(
            STATE_NAMES
        ):
            summary[
                f"policy_low_{ecological_name}"
            ] = float(
                np.mean(
                    [
                        policy[
                            ecological_index
                        ]
                        == LOW_EXTRACT
                        for policy
                        in policy_list
                    ]
                )
            )

    else:
        # Ecological marginal across the three social states.
        for (
            ecological_index,
            ecological_name,
        ) in enumerate(
            STATE_NAMES
        ):
            summary[
                f"policy_low_{ecological_name}"
            ] = float(
                np.mean(
                    [
                        (
                            policy[
                                joint_state(
                                    ecological_index,
                                    social_index,
                                )
                            ]
                            == LOW_EXTRACT
                        )
                        for policy
                        in policy_list
                        for social_index
                        in range(
                            3
                        )
                    ]
                )
            )

        # Full ecological x social policy matrix.
        for (
            ecological_index,
            ecological_name,
        ) in enumerate(
            STATE_NAMES
        ):
            for (
                social_index,
                social_name,
            ) in enumerate(
                SOCIAL_STATE_NAMES
            ):
                state = joint_state(
                    ecological_index,
                    social_index,
                )

                summary[
                    (
                        "policy_low_"
                        f"{ecological_name}_"
                        f"{social_name}"
                    )
                ] = float(
                    np.mean(
                        [
                            policy[
                                state
                            ]
                            == LOW_EXTRACT
                            for policy
                            in policy_list
                        ]
                    )
                )

    return (
        summary,
        agent_rows,
    )


# ---------------------------------------------------------------------
# One experimental condition
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
        sources,
        final_training_actions,
    ) = train_q_learning(
        scenario_name,
        population,
        replicate,
        args,
    )

    (
        policy_summary,
        agent_policy_rows,
    ) = policy_diagnostics(
        trained_env,
        regions,
        learners,
        scenario_name,
        population,
        replicate,
        social_mode=(
            args.social_mode
        ),
        sources=sources,
    )

    # A. Continue from the ecology produced during training.
    (
        continuation_summary,
        continuation_ts,
    ) = evaluate_policy(
        trained_env,
        trained_observations,
        scenario_name=(
            scenario_name
        ),
        population=population,
        replicate=replicate,
        strategy="q_learning",
        evaluation_mode=(
            "continuation"
        ),
        learners=learners,
        sources=sources,
        previous_actions=(
            final_training_actions
        ),
        args=args,
    )

    # B. Same learned Q-policy from fresh ecological conditions.
    #
    # The S1 social graph is fixed and therefore carried unchanged.
    # Social action memory is reset to neutral at the first step.
    (
        fresh_env,
        fresh_obs,
        _,
    ) = make_environment(
        scenario_name,
        population,
        replicate,
        args.evaluation_steps,
        args,
    )

    (
        fresh_summary,
        fresh_ts,
    ) = evaluate_policy(
        fresh_env,
        fresh_obs,
        scenario_name=(
            scenario_name
        ),
        population=population,
        replicate=replicate,
        strategy="q_learning",
        evaluation_mode=(
            "fresh_reset"
        ),
        learners=learners,
        sources=sources,
        previous_actions=None,
        args=args,
    )

    # C. Fixed-policy controls from the same fresh ecological state.
    control_summaries = []
    control_timeseries = []

    for strategy in (
        "always_low",
        "always_high",
        "random_50",
    ):
        (
            control_env,
            control_obs,
            _,
        ) = make_environment(
            scenario_name,
            population,
            replicate,
            args.evaluation_steps,
            args,
        )

        (
            summary,
            timeseries,
        ) = evaluate_policy(
            control_env,
            control_obs,
            scenario_name=(
                scenario_name
            ),
            population=population,
            replicate=replicate,
            strategy=strategy,
            evaluation_mode=(
                "fresh_reset"
            ),
            learners=None,
            sources=sources,
            previous_actions=None,
            args=args,
        )

        control_summaries.append(
            summary
        )

        control_timeseries.extend(
            timeseries
        )

    return {
        "evaluation_summary": [
            continuation_summary,
            fresh_summary,
            *control_summaries,
        ],
        "training_timeseries": (
            training_timeseries
        ),
        "evaluation_timeseries": [
            *continuation_ts,
            *fresh_ts,
            *control_timeseries,
        ],
        "policy_summary": (
            policy_summary
        ),
        "agent_policies": (
            agent_policy_rows
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main(
) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline validation for independent ecological Q-learning "
            "with optional fixed social observation."
        )
    )

    parser.add_argument(
        "--run-name",
        default=(
            "baseline_validation_v1"
        ),
    )

    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(
            SCENARIOS
        ),
    )

    parser.add_argument(
        "--populations",
        type=int,
        nargs="+",
        default=[
            8,
            16,
            32,
        ],
    )

    parser.add_argument(
        "--replicates",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--training-steps",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--evaluation-steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--record-every",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--coupling",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--low-harvest",
        type=float,
        default=0.002,
    )

    parser.add_argument(
        "--high-harvest",
        type=float,
        default=0.020,
    )

    parser.add_argument(
        "--metabolism",
        type=float,
        default=0.002,
    )

    parser.add_argument(
        "--initial-energy",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--energy-capacity",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--initial-resource-fraction",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.9995,
    )

    parser.add_argument(
        "--social-mode",
        choices=SOCIAL_MODES,
        default="none",
        help=(
            "'none' reproduces the ecological baseline. "
            "'fixed' adds a fixed directed attention network and "
            "previous-action social observation."
        ),
    )

    parser.add_argument(
        "--social-k",
        type=int,
        default=4,
        help=(
            "Number of information sources observed by each agent "
            "when --social-mode fixed."
        ),
    )

    args = (
        parser.parse_args()
    )

    for scenario in (
        args.scenarios
    ):
        if (
            scenario
            not in SCENARIOS
        ):
            raise ValueError(
                f"Unknown scenario: "
                f"{scenario}"
            )

    if (
        args.social_mode
        == "fixed"
    ):
        if (
            args.social_k
            < 1
        ):
            raise ValueError(
                "--social-k must be at least 1."
            )

        for population in (
            args.populations
        ):
            if (
                args.social_k
                >= population
            ):
                raise ValueError(
                    "--social-k must be smaller "
                    "than every population size. "
                    f"Received k={args.social_k}, "
                    f"population={population}."
                )

    run_dir = (
        RESULTS_ROOT
        / args.run_name
    )

    data_dir = (
        run_dir
        / "data"
    )

    figures_dir = (
        run_dir
        / "figures"
    )

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = vars(
        args
    ).copy()

    config[
        "scenario_definitions"
    ] = {
        name: SCENARIOS[
            name
        ]
        for name
        in args.scenarios
    }

    config[
        "learner_n_states"
    ] = learner_state_count(
        args.social_mode
    )

    with (
        run_dir
        / "config.json"
    ).open(
        "w"
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
        )

    evaluation_rows = []
    training_rows = []
    evaluation_timeseries_rows = []
    policy_rows = []
    agent_policy_rows = []

    total = (
        len(
            args.scenarios
        )
        * len(
            args.populations
        )
        * args.replicates
    )

    completed = 0

    print(
        f"Running {total} "
        "Q-learning training conditions..."
    )

    print(
        f"Social mode: "
        f"{args.social_mode}"
    )

    if (
        args.social_mode
        == "fixed"
    ):
        print(
            f"Attention capacity k: "
            f"{args.social_k}"
        )

    print(
        "Each condition also receives continuation evaluation, "
        "fresh-reset evaluation, and three fixed-policy controls."
    )

    for replicate in range(
        args.replicates
    ):
        for scenario in (
            args.scenarios
        ):
            for population in (
                args.populations
            ):
                output = run_condition(
                    scenario,
                    population,
                    replicate,
                    args,
                )

                evaluation_rows.extend(
                    output[
                        "evaluation_summary"
                    ]
                )

                training_rows.extend(
                    output[
                        "training_timeseries"
                    ]
                )

                evaluation_timeseries_rows.extend(
                    output[
                        "evaluation_timeseries"
                    ]
                )

                policy_rows.append(
                    output[
                        "policy_summary"
                    ]
                )

                agent_policy_rows.extend(
                    output[
                        "agent_policies"
                    ]
                )

                completed += 1

                print(
                    f"[{completed}/{total}] "
                    f"{scenario} "
                    f"N={population} "
                    f"replicate={replicate}"
                )

    write_csv(
        data_dir
        / "evaluation_summary.csv",
        evaluation_rows,
    )

    write_csv(
        data_dir
        / "training_timeseries.csv",
        training_rows,
    )

    write_csv(
        data_dir
        / "evaluation_timeseries.csv",
        evaluation_timeseries_rows,
    )

    write_csv(
        data_dir
        / "policy_summary.csv",
        policy_rows,
    )

    write_csv(
        data_dir
        / "agent_policies.csv",
        agent_policy_rows,
    )

    print(
        "\nSimulation complete."
    )

    print(
        f"Results: "
        f"{run_dir.resolve()}"
    )

    print(
        "Generate the existing baseline figures with:"
    )

    print(
        "python baseline_validation_figures.py "
        f"--run-name {args.run_name}"
    )


if __name__ == "__main__":
    main()