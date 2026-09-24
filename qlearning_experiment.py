from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from Cognitive_tools import EcoEnv
from Cognitive_tools.ecology import make_capacity_map
from Cognitive_tools.model import COOPERATE
from Cognitive_tools.qlearning import (
    QLearningPolicy,
    STATE_NAMES,
    resource_state,
)


RESULTS_ROOT = (
    Path("results")
    / "q_learning_baseline"
)


SCENARIOS = {
    "uniform_high": {
        "label": "Uniform high",
        "group": "single",
        "family": "uniform",
        "mean_capacity": 0.75,
        "recovery": 0.05,
        "equilibrium": 0.70,
    },
    "patchy_high": {
        "label": "Patchy high",
        "group": "single",
        "family": "patchy",
        "mean_capacity": 0.75,
        "recovery": 0.05,
        "equilibrium": 0.70,
    },
    "centralized_low": {
        "label": "Centralized low",
        "group": "single",
        "family": "centralized",
        "mean_capacity": 0.35,
        "recovery": 0.03,
        "equilibrium": 0.60,
    },
    "decentralized_high": {
        "label": "Decentralized high",
        "group": "single",
        "family": "decentralized",
        "mean_capacity": 0.75,
        "recovery": 0.05,
        "equilibrium": 0.70,
    },
    "patchy_high_central_low": {
        "label": "Patchy high + central low",
        "group": "mixed",
        "kind": "central_low",
    },
    "central_high_patchy_low": {
        "label": "Central high + patchy low",
        "group": "mixed",
        "kind": "central_high",
    },
    "split_high_low": {
        "label": "High patchy + low fragmented split",
        "group": "mixed",
        "kind": "split",
    },
    "decentralized_high_in_low": {
        "label": "Decentralized high islands + low background",
        "group": "mixed",
        "kind": "high_islands",
    },
}


def make_run_directories(
    run_name: str | None,
) -> dict[str, Path]:
    if run_name:
        name = run_name
    else:
        name = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

    root = (
        RESULTS_ROOT
        / name
    )

    paths = {
        "root": root,
        "data": root / "data",
        "figures": root / "figures",
        "gifs": root / "gifs",
        "frames": root / "frames",
    }

    for path in paths.values():
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    return paths


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        return

    with path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

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

    if len(values) == 0 or total <= 0:
        return 0.0

    n = len(values)

    index = np.arange(
        1,
        n + 1,
    )

    value = (
        2
        * np.sum(
            index
            * values
        )
        / (
            n
            * total
        )
        - (
            n + 1
        )
        / n
    )

    if abs(value) < 1e-12:
        return 0.0

    return float(
        max(
            0.0,
            value,
        )
    )


def binary_entropy(
    p: float,
) -> float:
    """
    Normalized binary entropy.

    0 = everyone chooses the same action.
    1 = a 50/50 collective mixture.
    """

    if p <= 0.0 or p >= 1.0:
        return 0.0

    return float(
        -p * math.log2(p)
        - (1.0 - p)
        * math.log2(1.0 - p)
    )


def gaussian_mask(
    width: int,
    height: int,
    *,
    centres: list[
        tuple[
            float,
            float,
        ]
    ],
    sigma: float,
    threshold: float,
) -> np.ndarray:
    y, x = np.indices(
        (
            height,
            width,
        )
    )

    field = np.zeros(
        (
            height,
            width,
        ),
        dtype=float,
    )

    for (
        centre_x,
        centre_y,
    ) in centres:
        distance_squared = (
            (
                x
                - centre_x
            )
            ** 2
            + (
                y
                - centre_y
            )
            ** 2
        )

        field += np.exp(
            -distance_squared
            / (
                2.0
                * sigma**2
            )
        )

    if (
        field.max()
        > 0
    ):
        field = (
            field
            / field.max()
        )

    return (
        field
        >= threshold
    )


def build_environment_maps(
    scenario_name: str,
    *,
    width: int,
    height: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Return:

        capacity
        regeneration-rate map
        equilibrium-fraction map
        region labels
    """

    spec = SCENARIOS[
        scenario_name
    ]

    if (
        spec[
            "group"
        ]
        == "single"
    ):
        capacity = (
            make_capacity_map(
                family=spec[
                    "family"
                ],
                mean_capacity=spec[
                    "mean_capacity"
                ],
                width=width,
                height=height,
                seed=seed,
            )
        )

        recovery = np.full(
            (
                height,
                width,
            ),
            spec[
                "recovery"
            ],
            dtype=float,
        )

        equilibrium = (
            np.full(
                (
                    height,
                    width,
                ),
                spec[
                    "equilibrium"
                ],
                dtype=float,
            )
        )

        regions = np.full(
            (
                height,
                width,
            ),
            scenario_name,
            dtype=object,
        )

        return (
            capacity,
            recovery,
            equilibrium,
            regions,
        )

    kind = spec[
        "kind"
    ]

    high_patchy = (
        make_capacity_map(
            family="patchy",
            mean_capacity=0.75,
            width=width,
            height=height,
            seed=seed,
        )
    )

    low_patchy = (
        make_capacity_map(
            family="patchy",
            mean_capacity=0.30,
            width=width,
            height=height,
            seed=(
                seed
                + 17
            ),
        )
    )

    if (
        kind
        == "central_low"
    ):
        mask = gaussian_mask(
            width,
            height,
            centres=[
                (
                    (
                        width
                        - 1
                    )
                    / 2.0,
                    (
                        height
                        - 1
                    )
                    / 2.0,
                )
            ],
            sigma=(
                min(
                    width,
                    height,
                )
                / 4.0
            ),
            threshold=0.52,
        )

        capacity = (
            high_patchy.copy()
        )

        capacity[
            mask
        ] = low_patchy[
            mask
        ]

        recovery = np.full(
            (
                height,
                width,
            ),
            0.07,
        )

        recovery[
            mask
        ] = 0.02

        equilibrium = np.full(
            (
                height,
                width,
            ),
            0.78,
        )

        equilibrium[
            mask
        ] = 0.48

        regions = np.full(
            (
                height,
                width,
            ),
            "patchy_high",
            dtype=object,
        )

        regions[
            mask
        ] = "central_low"

    elif (
        kind
        == "central_high"
    ):
        mask = gaussian_mask(
            width,
            height,
            centres=[
                (
                    (
                        width
                        - 1
                    )
                    / 2.0,
                    (
                        height
                        - 1
                    )
                    / 2.0,
                )
            ],
            sigma=(
                min(
                    width,
                    height,
                )
                / 4.0
            ),
            threshold=0.52,
        )

        capacity = (
            low_patchy.copy()
        )

        high_central = (
            make_capacity_map(
                family=(
                    "centralized"
                ),
                mean_capacity=(
                    0.78
                ),
                width=width,
                height=height,
                seed=(
                    seed
                    + 31
                ),
            )
        )

        capacity[
            mask
        ] = high_central[
            mask
        ]

        recovery = np.full(
            (
                height,
                width,
            ),
            0.025,
        )

        recovery[
            mask
        ] = 0.08

        equilibrium = np.full(
            (
                height,
                width,
            ),
            0.52,
        )

        equilibrium[
            mask
        ] = 0.82

        regions = np.full(
            (
                height,
                width,
            ),
            "patchy_low",
            dtype=object,
        )

        regions[
            mask
        ] = "central_high"

    elif (
        kind
        == "split"
    ):
        fragmented_low = (
            make_capacity_map(
                family=(
                    "fragmented"
                ),
                mean_capacity=(
                    0.30
                ),
                width=width,
                height=height,
                seed=(
                    seed
                    + 47
                ),
            )
        )

        mask = np.zeros(
            (
                height,
                width,
            ),
            dtype=bool,
        )

        mask[
            :,
            width // 2:
        ] = True

        capacity = (
            high_patchy.copy()
        )

        capacity[
            mask
        ] = fragmented_low[
            mask
        ]

        recovery = np.full(
            (
                height,
                width,
            ),
            0.07,
        )

        recovery[
            mask
        ] = 0.02

        equilibrium = np.full(
            (
                height,
                width,
            ),
            0.78,
        )

        equilibrium[
            mask
        ] = 0.48

        regions = np.full(
            (
                height,
                width,
            ),
            "left_high_patchy",
            dtype=object,
        )

        regions[
            mask
        ] = (
            "right_low_fragmented"
        )

    elif (
        kind
        == "high_islands"
    ):
        centres = [
            (
                width
                * 0.25,
                height
                * 0.25,
            ),
            (
                width
                * 0.75,
                height
                * 0.25,
            ),
            (
                width
                * 0.25,
                height
                * 0.75,
            ),
            (
                width
                * 0.75,
                height
                * 0.75,
            ),
        ]

        mask = gaussian_mask(
            width,
            height,
            centres=centres,
            sigma=1.4,
            threshold=0.48,
        )

        capacity = (
            low_patchy.copy()
        )

        high_islands = (
            make_capacity_map(
                family=(
                    "decentralized"
                ),
                mean_capacity=(
                    0.78
                ),
                width=width,
                height=height,
                seed=(
                    seed
                    + 73
                ),
            )
        )

        capacity[
            mask
        ] = high_islands[
            mask
        ]

        recovery = np.full(
            (
                height,
                width,
            ),
            0.025,
        )

        recovery[
            mask
        ] = 0.075

        equilibrium = np.full(
            (
                height,
                width,
            ),
            0.52,
        )

        equilibrium[
            mask
        ] = 0.80

        regions = np.full(
            (
                height,
                width,
            ),
            "low_background",
            dtype=object,
        )

        regions[
            mask
        ] = "high_island"

    else:
        raise ValueError(
            "Unknown mixed "
            "scenario kind: "
            f"{kind}"
        )

    return (
        capacity,
        recovery,
        equilibrium,
        regions,
    )


def make_env(
    scenario_name: str,
    population: int,
    replicate: int,
    args,
):
    scenario_index = list(
        SCENARIOS
    ).index(
        scenario_name
    )

    map_seed = (
        args.seed
        + replicate
        + 1000
        * scenario_index
    )

    (
        capacity,
        recovery,
        equilibrium,
        regions,
    ) = build_environment_maps(
        scenario_name,
        width=args.width,
        height=args.height,
        seed=map_seed,
    )

    total_steps = (
        args.training_steps
        + args.evaluation_steps
    )

    env = EcoEnv(
        width=args.width,
        height=args.height,
        n_agents=population,
        max_steps=total_steps,
        regeneration_rate=recovery,
        equilibrium_fraction=(
            equilibrium
        ),
        coupling_rate=(
            args.coupling
        ),
        cooperative_harvest_amount=(
            args.cooperative_harvest
        ),
        defective_harvest_amount=(
            args.defective_harvest
        ),
        metabolism_rate=(
            args.metabolism
        ),
        initial_energy=(
            args.initial_energy
        ),
        initial_resource_fraction=(
            args.initial_resource_fraction
        ),
        capacity_map=capacity,
    )

    observations, _ = (
        env.reset(
            seed=(
                args.seed
                + 100_000
                + replicate
            )
        )
    )

    return (
        env,
        observations,
        regions,
    )


def make_learners(
    env: EcoEnv,
    replicate: int,
    args,
) -> dict[
    str,
    QLearningPolicy,
]:
    learners = {}

    for index, name in enumerate(
        env.possible_agents
    ):
        learners[
            name
        ] = QLearningPolicy(
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
    previous_actions: (
        dict[
            str,
            int,
        ]
        | None
    ),
) -> dict:
    agents = list(
        env.model.by_name.values()
    )

    wealth = [
        agent.wealth
        for agent
        in agents
    ]

    energy = [
        agent.energy
        for agent
        in agents
    ]

    cooperation_rate = float(
        np.mean(
            [
                action
                == COOPERATE
                for action
                in actions.values()
            ]
        )
    )

    if (
        previous_actions
        is None
    ):
        switch_rate = 0.0

    else:
        switch_rate = float(
            np.mean(
                [
                    actions[
                        name
                    ]
                    != previous_actions[
                        name
                    ]
                    for name
                    in actions
                ]
            )
        )

    return {
        "cooperation_rate": (
            cooperation_rate
        ),
        "collective_action_entropy": (
            binary_entropy(
                cooperation_rate
            )
        ),
        "collective_order": abs(
            2.0
            * cooperation_rate
            - 1.0
        ),
        "switch_rate": (
            switch_rate
        ),
        "wealth_gini": gini(
            wealth
        ),
        "mean_welfare": float(
            np.mean(
                energy
            )
        ),
        "mean_wealth": float(
            np.mean(
                wealth
            )
        ),
        "total_resource": float(
            env.model.resource.sum()
        ),
        "mean_resource_fraction": float(
            np.mean(
                env.model.resource
                / env.model.capacity
            )
        ),
    }


def policy_metrics(
    learners: dict[
        str,
        QLearningPolicy,
    ],
) -> dict:
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
        distance = np.mean(
            np.asarray(
                first
            )
            != np.asarray(
                second
            )
        )

        pairwise.append(
            float(
                distance
            )
        )

    mean_hamming = (
        float(
            np.mean(
                pairwise
            )
        )
        if pairwise
        else 0.0
    )

    counts = Counter(
        policy_list
    )

    probabilities = np.array(
        list(
            counts.values()
        ),
        dtype=float,
    )

    probabilities /= (
        probabilities.sum()
    )

    entropy = 0.0

    for p in (
        probabilities
    ):
        if p > 0:
            entropy -= (
                p
                * math.log2(
                    p
                )
            )

    max_types = min(
        len(
            policy_list
        ),
        2
        ** len(
            STATE_NAMES
        ),
    )

    if max_types > 1:
        entropy /= (
            math.log2(
                max_types
            )
        )

    else:
        entropy = 0.0

    state_cooperation = {}

    for (
        state_index,
        state_name,
    ) in enumerate(
        STATE_NAMES
    ):
        state_cooperation[
            f"policy_cooperate_{state_name}"
        ] = float(
            np.mean(
                [
                    policy[
                        state_index
                    ]
                    == COOPERATE
                    for policy
                    in policy_list
                ]
            )
        )

    return {
        "policies": (
            policies
        ),
        "unique_policy_count": len(
            counts
        ),
        "policy_entropy": float(
            entropy
        ),
        "policy_hamming_mean": (
            mean_hamming
        ),
        **state_cooperation,
    }


def build_interaction_network(
    env: EcoEnv,
    policies: dict[
        str,
        tuple[
            int,
            int,
            int,
        ],
    ],
    *,
    radius: int,
) -> nx.Graph:
    """
    Ecological interaction network.

    Two stationary agents are linked when they occupy the same tile
    or tiles within the chosen Manhattan radius. This is not a
    communication network. It represents potential coupling through
    a shared/local commons.
    """

    graph = nx.Graph()

    names = list(
        env.model.by_name
    )

    for name in names:
        agent = (
            env.model.by_name[
                name
            ]
        )

        policy = (
            policies[
                name
            ]
        )

        cooperation_preference = (
            float(
                np.mean(
                    np.asarray(
                        policy
                    )
                    == COOPERATE
                )
            )
        )

        graph.add_node(
            name,
            x=int(
                agent.position[
                    0
                ]
            ),
            y=int(
                agent.position[
                    1
                ]
            ),
            cooperation_preference=(
                cooperation_preference
            ),
        )

    for (
        first,
        second,
    ) in combinations(
        names,
        2,
    ):
        a = (
            env.model.by_name[
                first
            ]
        )

        b = (
            env.model.by_name[
                second
            ]
        )

        distance = (
            abs(
                int(
                    a.position[
                        0
                    ]
                    - b.position[
                        0
                    ]
                )
            )
            + abs(
                int(
                    a.position[
                        1
                    ]
                    - b.position[
                        1
                    ]
                )
            )
        )

        if (
            distance
            <= radius
        ):
            graph.add_edge(
                first,
                second,
                distance=(
                    distance
                ),
                weight=(
                    1.0
                    / (
                        1.0
                        + distance
                    )
                ),
            )

    return graph


def network_metrics(
    graph: nx.Graph,
    policies: dict[
        str,
        tuple[
            int,
            int,
            int,
        ],
    ],
) -> dict:
    n = (
        graph.number_of_nodes()
    )

    if n == 0:
        return {
            "network_density": 0.0,
            "network_components": 0,
            "largest_component_fraction": 0.0,
            "network_clustering": 0.0,
            "network_policy_similarity": float(
                "nan"
            ),
            "policy_assortativity": float(
                "nan"
            ),
        }

    components = list(
        nx.connected_components(
            graph
        )
    )

    largest = max(
        (
            len(
                component
            )
            for component
            in components
        ),
        default=0,
    )

    similarities = []

    for (
        first,
        second,
    ) in graph.edges():
        distance = float(
            np.mean(
                np.asarray(
                    policies[
                        first
                    ]
                )
                != np.asarray(
                    policies[
                        second
                    ]
                )
            )
        )

        similarities.append(
            1.0
            - distance
        )

    if similarities:
        policy_similarity = float(
            np.mean(
                similarities
            )
        )

    else:
        policy_similarity = float(
            "nan"
        )

    preferences = [
        graph.nodes[
            name
        ][
            "cooperation_preference"
        ]
        for name
        in graph.nodes
    ]

    if (
        graph.number_of_edges()
        > 0
        and np.std(
            preferences
        )
        > 1e-12
    ):
        try:
            assortativity = float(
                nx.numeric_assortativity_coefficient(
                    graph,
                    "cooperation_preference",
                )
            )

        except Exception:
            assortativity = float(
                "nan"
            )

    else:
        assortativity = float(
            "nan"
        )

    return {
        "network_density": float(
            nx.density(
                graph
            )
        ),
        "network_components": len(
            components
        ),
        "largest_component_fraction": (
            largest
            / n
        ),
        "network_clustering": float(
            nx.average_clustering(
                graph
            )
        ),
        "network_policy_similarity": (
            policy_similarity
        ),
        "policy_assortativity": (
            assortativity
        ),
    }


def region_for_agent(
    env: EcoEnv,
    regions: np.ndarray,
    name: str,
) -> str:
    agent = (
        env.model.by_name[
            name
        ]
    )

    x, y = (
        agent.position
    )

    return str(
        regions[
            int(
                y
            ),
            int(
                x
            ),
        ]
    )


def run_condition(
    scenario_name: str,
    population: int,
    replicate: int,
    args,
):
    (
        env,
        observations,
        regions,
    ) = make_env(
        scenario_name,
        population,
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

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------

    for time in range(
        1,
        args.training_steps
        + 1,
    ):
        states = {
            name: resource_state(
                observations[
                    name
                ]
            )
            for name
            in env.agents
        }

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
            infos,
        ) = env.step(
            actions
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
                next_state = (
                    states[
                        name
                    ]
                )

            else:
                next_state = (
                    resource_state(
                        next_observations[
                            name
                        ]
                    )
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
                next_state,
                done=done,
            )

            learners[
                name
            ].decay_exploration()

        if (
            time
            % args.record_every
            == 0
            or time == 1
            or time
            == args.training_steps
        ):
            metrics = system_metrics(
                env,
                actions,
                previous_actions,
            )

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
                    "phase": (
                        "training"
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

    # -------------------------------------------------------------
    # Greedy evaluation
    # -------------------------------------------------------------

    evaluation_rows = []

    for evaluation_step in range(
        1,
        args.evaluation_steps
        + 1,
    ):
        states = {
            name: resource_state(
                observations[
                    name
                ]
            )
            for name
            in env.agents
        }

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
            in env.agents
        }

        (
            next_observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = env.step(
            actions
        )

        metrics = system_metrics(
            env,
            actions,
            previous_actions,
        )

        evaluation_rows.append(
            metrics
        )

        absolute_time = (
            args.training_steps
            + evaluation_step
        )

        if (
            evaluation_step
            % args.record_every
            == 0
            or evaluation_step
            == 1
            or evaluation_step
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
                    "phase": (
                        "evaluation"
                    ),
                    "time": (
                        absolute_time
                    ),
                    **metrics,
                }
            )

        previous_actions = (
            actions.copy()
        )

        observations = (
            next_observations
        )

    policy_info = (
        policy_metrics(
            learners
        )
    )

    graph = (
        build_interaction_network(
            env,
            policy_info[
                "policies"
            ],
            radius=(
                args.network_radius
            ),
        )
    )

    net_info = (
        network_metrics(
            graph,
            policy_info[
                "policies"
            ],
        )
    )

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
        "cooperation_rate": float(
            np.mean(
                [
                    row[
                        "cooperation_rate"
                    ]
                    for row
                    in evaluation_rows
                ]
            )
        ),
        "collective_action_entropy": float(
            np.mean(
                [
                    row[
                        "collective_action_entropy"
                    ]
                    for row
                    in evaluation_rows
                ]
            )
        ),
        "collective_order": float(
            np.mean(
                [
                    row[
                        "collective_order"
                    ]
                    for row
                    in evaluation_rows
                ]
            )
        ),
        "switch_rate": float(
            np.mean(
                [
                    row[
                        "switch_rate"
                    ]
                    for row
                    in evaluation_rows
                ]
            )
        ),
        "wealth_gini": float(
            np.mean(
                [
                    row[
                        "wealth_gini"
                    ]
                    for row
                    in evaluation_rows
                ]
            )
        ),
        "mean_welfare": float(
            np.mean(
                [
                    row[
                        "mean_welfare"
                    ]
                    for row
                    in evaluation_rows
                ]
            )
        ),
        "mean_wealth": float(
            np.mean(
                [
                    row[
                        "mean_wealth"
                    ]
                    for row
                    in evaluation_rows
                ]
            )
        ),
        "total_resource": float(
            np.mean(
                [
                    row[
                        "total_resource"
                    ]
                    for row
                    in evaluation_rows
                ]
            )
        ),
        "mean_resource_fraction": float(
            np.mean(
                [
                    row[
                        "mean_resource_fraction"
                    ]
                    for row
                    in evaluation_rows
                ]
            )
        ),
        "unique_policy_count": (
            policy_info[
                "unique_policy_count"
            ]
        ),
        "policy_entropy": (
            policy_info[
                "policy_entropy"
            ]
        ),
        "policy_hamming_mean": (
            policy_info[
                "policy_hamming_mean"
            ]
        ),
        "policy_cooperate_scarce": (
            policy_info[
                "policy_cooperate_scarce"
            ]
        ),
        "policy_cooperate_moderate": (
            policy_info[
                "policy_cooperate_moderate"
            ]
        ),
        "policy_cooperate_abundant": (
            policy_info[
                "policy_cooperate_abundant"
            ]
        ),
        **net_info,
    }

    agent_rows = []

    for (
        name,
        learner,
    ) in learners.items():
        agent = (
            env.model.by_name[
                name
            ]
        )

        policy = (
            learner.greedy_policy()
        )

        agent_rows.append(
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
                "agent": name,
                "x": int(
                    agent.position[
                        0
                    ]
                ),
                "y": int(
                    agent.position[
                        1
                    ]
                ),
                "region": (
                    region_for_agent(
                        env,
                        regions,
                        name,
                    )
                ),
                "policy_scarce": (
                    "C"
                    if policy[
                        0
                    ]
                    == COOPERATE
                    else "D"
                ),
                "policy_moderate": (
                    "C"
                    if policy[
                        1
                    ]
                    == COOPERATE
                    else "D"
                ),
                "policy_abundant": (
                    "C"
                    if policy[
                        2
                    ]
                    == COOPERATE
                    else "D"
                ),
                "cooperation_preference": float(
                    np.mean(
                        np.asarray(
                            policy
                        )
                        == COOPERATE
                    )
                ),
                "wealth": float(
                    agent.wealth
                ),
                "energy": float(
                    agent.energy
                ),
            }
        )

    return (
        summary,
        timeseries,
        agent_rows,
    )


def aggregate_runs(
    rows: list[
        dict
    ],
) -> list[
    dict
]:
    metrics = [
        "cooperation_rate",
        "collective_action_entropy",
        "collective_order",
        "switch_rate",
        "wealth_gini",
        "mean_welfare",
        "mean_wealth",
        "total_resource",
        "mean_resource_fraction",
        "unique_policy_count",
        "policy_entropy",
        "policy_hamming_mean",
        "policy_cooperate_scarce",
        "policy_cooperate_moderate",
        "policy_cooperate_abundant",
        "network_density",
        "network_components",
        "largest_component_fraction",
        "network_clustering",
        "network_policy_similarity",
        "policy_assortativity",
    ]

    groups = {}

    for row in rows:
        key = (
            row[
                "scenario"
            ],
            row[
                "scenario_label"
            ],
            row[
                "scenario_group"
            ],
            row[
                "population"
            ],
        )

        groups.setdefault(
            key,
            [],
        ).append(
            row
        )

    output = []

    for (
        key,
        group,
    ) in groups.items():
        (
            scenario,
            label,
            scenario_group,
            population,
        ) = key

        result = {
            "scenario": (
                scenario
            ),
            "scenario_label": (
                label
            ),
            "scenario_group": (
                scenario_group
            ),
            "population": (
                population
            ),
            "replicates": len(
                group
            ),
        }

        for metric in (
            metrics
        ):
            values = np.array(
                [
                    row[
                        metric
                    ]
                    for row
                    in group
                ],
                dtype=float,
            )

            finite = values[
                np.isfinite(
                    values
                )
            ]

            if (
                len(
                    finite
                )
                == 0
            ):
                mean = float(
                    "nan"
                )

                sd = float(
                    "nan"
                )

            else:
                mean = float(
                    finite.mean()
                )

                sd = (
                    float(
                        finite.std(
                            ddof=1
                        )
                    )
                    if len(
                        finite
                    )
                    > 1
                    else 0.0
                )

            result[
                f"{metric}_mean"
            ] = mean

            result[
                f"{metric}_sd"
            ] = sd

        output.append(
            result
        )

    return sorted(
        output,
        key=lambda row: (
            row[
                "scenario_group"
            ],
            row[
                "scenario"
            ],
            row[
                "population"
            ],
        ),
    )


def plot_metric(
    rows: list[
        dict
    ],
    *,
    metric: str,
    ylabel: str,
    filename: str,
    paths: dict[
        str,
        Path,
    ],
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            13,
            5.5,
        ),
        sharey=True,
        layout="constrained",
    )

    for (
        ax,
        group_name,
    ) in zip(
        axes,
        [
            "single",
            "mixed",
        ],
    ):
        scenarios = [
            name
            for (
                name,
                spec,
            )
            in SCENARIOS.items()
            if (
                spec[
                    "group"
                ]
                == group_name
            )
        ]

        for scenario in (
            scenarios
        ):
            subset = sorted(
                [
                    row
                    for row
                    in rows
                    if (
                        row[
                            "scenario"
                        ]
                        == scenario
                    )
                ],
                key=lambda row: (
                    row[
                        "population"
                    ]
                ),
            )

            if not subset:
                continue

            ax.errorbar(
                [
                    row[
                        "population"
                    ]
                    for row
                    in subset
                ],
                [
                    row[
                        f"{metric}_mean"
                    ]
                    for row
                    in subset
                ],
                yerr=[
                    row[
                        f"{metric}_sd"
                    ]
                    for row
                    in subset
                ],
                marker="o",
                capsize=3,
                label=(
                    SCENARIOS[
                        scenario
                    ][
                        "label"
                    ]
                ),
            )

        ax.set_title(
            (
                "Single-regime environments"
                if (
                    group_name
                    == "single"
                )
                else (
                    "Mixed-regime environments"
                )
            )
        )

        ax.set_xlabel(
            "Population size"
        )

        ax.grid(
            alpha=0.25
        )

    axes[
        0
    ].set_ylabel(
        ylabel
    )

    axes[
        1
    ].legend(
        loc="center left",
        bbox_to_anchor=(
            1.02,
            0.5,
        ),
        title="Environment",
    )

    path = (
        paths[
            "figures"
        ]
        / filename
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved {path}"
    )


def plot_state_conditioned_policy(
    rows: list[
        dict
    ],
    paths: dict[
        str,
        Path,
    ],
) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            row[
                "scenario_group"
            ],
            row[
                "scenario"
            ],
            row[
                "population"
            ],
        ),
    )

    matrix = np.array(
        [
            [
                row[
                    "policy_cooperate_scarce_mean"
                ],
                row[
                    "policy_cooperate_moderate_mean"
                ],
                row[
                    "policy_cooperate_abundant_mean"
                ],
            ]
            for row
            in ordered
        ],
        dtype=float,
    )

    labels = [
        (
            f"{row['scenario']} "
            f"N={row['population']}"
        )
        for row
        in ordered
    ]

    fig, ax = plt.subplots(
        figsize=(
            8,
            max(
                6,
                0.34
                * len(
                    labels
                ),
            ),
        ),
        layout="constrained",
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xticks(
        range(
            3
        )
    )

    ax.set_xticklabels(
        [
            "Scarce",
            "Moderate",
            "Abundant",
        ]
    )

    ax.set_yticks(
        range(
            len(
                labels
            )
        )
    )

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        "Local resource state"
    )

    ax.set_ylabel(
        "Environment and population"
    )

    ax.set_title(
        "Fraction of learned greedy policies "
        "choosing cooperation"
    )

    fig.colorbar(
        image,
        ax=ax,
        label=(
            "Fraction choosing "
            "cooperative extraction"
        ),
    )

    path = (
        paths[
            "figures"
        ]
        / (
            "07_state_conditioned_"
            "cooperation.png"
        )
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved {path}"
    )


def save_environment_catalog(
    args,
    paths: dict[
        str,
        Path,
    ],
) -> None:
    names = list(
        SCENARIOS
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(
            14,
            7,
        ),
        layout="constrained",
    )

    image = None

    for (
        ax,
        name,
    ) in zip(
        axes.ravel(),
        names,
    ):
        (
            capacity,
            _,
            _,
            _,
        ) = build_environment_maps(
            name,
            width=args.width,
            height=args.height,
            seed=args.seed,
        )

        image = ax.imshow(
            capacity,
            vmin=0.0,
            vmax=1.0,
        )

        ax.set_title(
            SCENARIOS[
                name
            ][
                "label"
            ]
        )

        ax.set_xticks(
            []
        )

        ax.set_yticks(
            []
        )

    fig.suptitle(
        "Q-learning baseline environments"
    )

    fig.colorbar(
        image,
        ax=axes,
        label=(
            "Carrying capacity"
        ),
        shrink=0.82,
    )

    path = (
        paths[
            "figures"
        ]
        / (
            "00_environment_"
            "catalog.png"
        )
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved {path}"
    )


def snapshot(
    env: EcoEnv,
    actions: dict[
        str,
        int,
    ],
) -> dict:
    return {
        "resource": (
            env.model.resource.copy()
        ),
        "positions": {
            name: (
                agent.position.copy()
            )
            for (
                name,
                agent,
            )
            in env.model.by_name.items()
        },
        "actions": (
            actions.copy()
        ),
    }


def render_gif_frame(
    state: dict,
    history: list[
        dict
    ],
    frame_index: int,
    scenario: str,
    population: int,
) -> np.ndarray:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            10,
            4.8,
        ),
        layout="constrained",
    )

    image = axes[
        0
    ].imshow(
        state[
            "resource"
        ],
        vmin=0.0,
        vmax=1.0,
    )

    for (
        name,
        position,
    ) in state[
        "positions"
    ].items():
        action = state[
            "actions"
        ][
            name
        ]

        marker = (
            "o"
            if (
                action
                == COOPERATE
            )
            else "x"
        )

        # Small deterministic display offset helps reveal
        # co-located agents.
        agent_index = int(
            name.split(
                "_"
            )[
                -1
            ]
        )

        dx = (
            (
                agent_index
                % 3
            )
            - 1
        ) * 0.08

        dy = (
            (
                (
                    agent_index
                    // 3
                )
                % 3
            )
            - 1
        ) * 0.08

        axes[
            0
        ].scatter(
            float(
                position[
                    0
                ]
            )
            + dx,
            float(
                position[
                    1
                ]
            )
            + dy,
            marker=marker,
            s=45,
            linewidths=1.0,
        )

    axes[
        0
    ].set_title(
        "Resource field and actions\n"
        "circle = cooperate, x = defect"
    )

    axes[
        0
    ].set_xticks(
        []
    )

    axes[
        0
    ].set_yticks(
        []
    )

    fig.colorbar(
        image,
        ax=axes[
            0
        ],
        label=(
            "Resource stock"
        ),
        fraction=0.046,
    )

    times = [
        row[
            "time"
        ]
        for row
        in history[
            : frame_index
            + 1
        ]
    ]

    cooperation = [
        row[
            "cooperation_rate"
        ]
        for row
        in history[
            : frame_index
            + 1
        ]
    ]

    order = [
        row[
            "collective_order"
        ]
        for row
        in history[
            : frame_index
            + 1
        ]
    ]

    axes[
        1
    ].plot(
        times,
        cooperation,
        label=(
            "Cooperation rate"
        ),
    )

    axes[
        1
    ].plot(
        times,
        order,
        label=(
            "Collective order"
        ),
    )

    axes[
        1
    ].set_xlim(
        0,
        history[
            -1
        ][
            "time"
        ],
    )

    axes[
        1
    ].set_ylim(
        0,
        1,
    )

    axes[
        1
    ].set_xlabel(
        "Training step"
    )

    axes[
        1
    ].set_ylabel(
        "Rate"
    )

    axes[
        1
    ].set_title(
        "Collective learning dynamics"
    )

    axes[
        1
    ].grid(
        alpha=0.25
    )

    axes[
        1
    ].legend()

    fig.suptitle(
        f"{scenario} | "
        f"N={population} | "
        f"step="
        f"{history[frame_index]['time']}"
    )

    fig.canvas.draw()

    width, height = (
        fig.canvas.get_width_height()
    )

    frame = np.frombuffer(
        fig.canvas.buffer_rgba(),
        dtype=np.uint8,
    )

    frame = frame.reshape(
        height,
        width,
        4,
    )[
        ...,
        :3,
    ].copy()

    plt.close(
        fig
    )

    return frame


def make_learning_gif(
    scenario_name: str,
    population: int,
    replicate: int,
    args,
    paths: dict[
        str,
        Path,
    ],
) -> None:
    try:
        import imageio.v2 as imageio

    except ImportError as exc:
        raise RuntimeError(
            "GIF export requires imageio and pillow. "
            "Install with: pip install imageio pillow"
        ) from exc

    original_training = (
        args.training_steps
    )

    original_evaluation = (
        args.evaluation_steps
    )

    args.training_steps = (
        args.gif_steps
    )

    args.evaluation_steps = 0

    (
        env,
        observations,
        regions,
    ) = make_env(
        scenario_name,
        population,
        replicate,
        args,
    )

    learners = make_learners(
        env,
        replicate,
        args,
    )

    histories = []
    states = []

    previous_actions = None

    for time in range(
        1,
        args.gif_steps
        + 1,
    ):
        current_states = {
            name: resource_state(
                observations[
                    name
                ]
            )
            for name
            in env.agents
        }

        actions = {
            name: learners[
                name
            ].choose_action(
                current_states[
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
            infos,
        ) = env.step(
            actions
        )

        for name in (
            current_states
        ):
            done = (
                terminations[
                    name
                ]
                or truncations[
                    name
                ]
            )

            if done:
                next_state = (
                    current_states[
                        name
                    ]
                )

            else:
                next_state = (
                    resource_state(
                        next_observations[
                            name
                        ]
                    )
                )

            learners[
                name
            ].update(
                current_states[
                    name
                ],
                actions[
                    name
                ],
                rewards[
                    name
                ],
                next_state,
                done=done,
            )

            learners[
                name
            ].decay_exploration()

        if (
            time == 1
            or (
                time
                % args.frame_every
                == 0
            )
            or time
            == args.gif_steps
        ):
            metrics = system_metrics(
                env,
                actions,
                previous_actions,
            )

            histories.append(
                {
                    "time": (
                        time
                    ),
                    **metrics,
                }
            )

            states.append(
                snapshot(
                    env,
                    actions,
                )
            )

        previous_actions = (
            actions.copy()
        )

        observations = (
            next_observations
        )

    frames = []

    frame_dir = (
        paths[
            "frames"
        ]
        / (
            f"{scenario_name}"
            f"_N{population}"
            f"_rep{replicate}"
        )
    )

    if (
        args.save_frames
    ):
        frame_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    for (
        index,
        state,
    ) in enumerate(
        states
    ):
        frame = (
            render_gif_frame(
                state,
                histories,
                index,
                scenario_name,
                population,
            )
        )

        frames.append(
            frame
        )

        if (
            args.save_frames
        ):
            imageio.imwrite(
                frame_dir
                / (
                    f"frame_"
                    f"{index:04d}.png"
                ),
                frame,
            )

    path = (
        paths[
            "gifs"
        ]
        / (
            f"{scenario_name}"
            f"_N{population}"
            f"_learning.gif"
        )
    )

    imageio.mimsave(
        path,
        frames,
        fps=args.fps,
    )

    print(
        f"Saved {path}"
    )

    args.training_steps = (
        original_training
    )

    args.evaluation_steps = (
        original_evaluation
    )


def main(
) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Independent Q-learning baseline "
            "for a stationary common-pool resource game."
        )
    )

    parser.add_argument(
        "--run-name",
        default="",
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
        default=10,
    )

    parser.add_argument(
        "--training-steps",
        type=int,
        default=5_000,
    )

    parser.add_argument(
        "--evaluation-steps",
        type=int,
        default=1_000,
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
        "--cooperative-harvest",
        type=float,
        default=0.002,
    )

    parser.add_argument(
        "--defective-harvest",
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
        "--network-radius",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--make-gif",
        default="",
        help=(
            "Scenario name for one learning GIF."
        ),
    )

    parser.add_argument(
        "--gif-population",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--gif-replicate",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--gif-steps",
        type=int,
        default=800,
    )

    parser.add_argument(
        "--frame-every",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--save-frames",
        action=(
            "store_true"
        ),
    )

    parser.add_argument(
        "--gif-only",
        action=(
            "store_true"
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
        args.make_gif
        and args.make_gif
        not in SCENARIOS
    ):
        raise ValueError(
            "Unknown GIF scenario: "
            f"{args.make_gif}"
        )

    paths = (
        make_run_directories(
            args.run_name
            or None
        )
    )

    config = vars(
        args
    ).copy()

    config[
        "scenario_definitions"
    ] = SCENARIOS

    with (
        paths[
            "root"
        ]
        / "config.json"
    ).open(
        "w"
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
        )

    save_environment_catalog(
        args,
        paths,
    )

    if (
        args.gif_only
    ):
        if not (
            args.make_gif
        ):
            raise ValueError(
                "--gif-only requires "
                "--make-gif SCENARIO."
            )

        make_learning_gif(
            args.make_gif,
            args.gif_population,
            args.gif_replicate,
            args,
            paths,
        )

        return

    run_rows = []
    timeseries_rows = []
    agent_rows = []

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
        "Q-learning conditions..."
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
                (
                    summary,
                    timeseries,
                    policies,
                ) = run_condition(
                    scenario,
                    population,
                    replicate,
                    args,
                )

                run_rows.append(
                    summary
                )

                timeseries_rows.extend(
                    timeseries
                )

                agent_rows.extend(
                    policies
                )

                completed += 1

                print(
                    f"[{completed}/{total}] "
                    f"{scenario} "
                    f"N={population} "
                    f"replicate={replicate}"
                )

    aggregate_rows = (
        aggregate_runs(
            run_rows
        )
    )

    write_csv(
        paths[
            "data"
        ]
        / "run_summary.csv",
        run_rows,
    )

    write_csv(
        paths[
            "data"
        ]
        / "aggregate_summary.csv",
        aggregate_rows,
    )

    write_csv(
        paths[
            "data"
        ]
        / "timeseries.csv",
        timeseries_rows,
    )

    write_csv(
        paths[
            "data"
        ]
        / "agent_policies.csv",
        agent_rows,
    )

    plot_metric(
        aggregate_rows,
        metric=(
            "cooperation_rate"
        ),
        ylabel=(
            "Cooperation rate"
        ),
        filename=(
            "01_cooperation_"
            "vs_population.png"
        ),
        paths=paths,
    )

    plot_metric(
        aggregate_rows,
        metric=(
            "mean_welfare"
        ),
        ylabel=(
            "Mean energy "
            "(welfare proxy)"
        ),
        filename=(
            "02_welfare_"
            "vs_population.png"
        ),
        paths=paths,
    )

    plot_metric(
        aggregate_rows,
        metric=(
            "wealth_gini"
        ),
        ylabel=(
            "Wealth Gini"
        ),
        filename=(
            "03_wealth_gini_"
            "vs_population.png"
        ),
        paths=paths,
    )

    plot_metric(
        aggregate_rows,
        metric=(
            "policy_hamming_mean"
        ),
        ylabel=(
            "Mean pairwise policy "
            "Hamming distance"
        ),
        filename=(
            "04_policy_heterogeneity_"
            "vs_population.png"
        ),
        paths=paths,
    )

    plot_metric(
        aggregate_rows,
        metric=(
            "collective_order"
        ),
        ylabel=(
            "Collective order parameter"
        ),
        filename=(
            "05_collective_order_"
            "vs_population.png"
        ),
        paths=paths,
    )

    plot_metric(
        aggregate_rows,
        metric=(
            "network_policy_similarity"
        ),
        ylabel=(
            "Interaction-network "
            "policy similarity"
        ),
        filename=(
            "06_network_policy_similarity_"
            "vs_population.png"
        ),
        paths=paths,
    )

    plot_state_conditioned_policy(
        aggregate_rows,
        paths,
    )

    if (
        args.make_gif
    ):
        make_learning_gif(
            args.make_gif,
            args.gif_population,
            args.gif_replicate,
            args,
            paths,
        )

    print(
        "\nExperiment complete."
    )

    print(
        f"Results: "
        f"{paths['root'].resolve()}"
    )


if __name__ == "__main__":
    main()