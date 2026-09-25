"""
Fixed social-observation primitives for cognitive_tools.

This module defines the minimal social-information layer used by the
first social-learning treatment.

An agent observes a fixed number k of other agents. The mapping

    sources[observer]

contains the agents whose actions that observer can see.

Conceptually, information flows:

    source -> observer

The first social treatment uses only the previous low/high extraction
actions of these information sources. It does not contain rewiring,
forecasting heuristics, homophily, prestige, communication noise, or
message passing.

Scientific provenance
---------------------
The idea that agents observe behavior through a restricted social
network is inspired by:

    Schrama, N., Tilman, A. R., & Vasconcelos, V. V. (2025).
    "Majority illusion drives the spontaneous emergence of alternative
    states in common-pool resource games with network-based information."
    iScience, 28(7), 112831.
    https://doi.org/10.1016/j.isci.2025.112831

This implementation is not a reproduction of Schrama et al.'s
Heuristics Switching Model.

The fixed-k directed attention representation is also useful preparation
for later rewiring experiments inspired by Oh & Schauf (2025), but no
rewiring is implemented in this module yet.

No source code from either paper is copied here.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .model import LOW_EXTRACT


SOCIAL_STATE_NAMES = (
    "mostly_high",
    "mixed",
    "mostly_low",
)


def init_random_attention(
    agents: list[str],
    k: int,
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """
    Create a directed fixed-k attention network.

    Each observer receives exactly k distinct information sources.

    Self-observation is excluded.

    Parameters
    ----------
    agents
        Agent names.

    k
        Number of information sources observed by every agent.

    rng
        Dedicated random-number generator for social-network
        initialization.

    Returns
    -------
    dict
        Mapping:

            observer -> list of information sources
    """

    names = list(
        agents
    )

    if len(
        set(
            names
        )
    ) != len(
        names
    ):
        raise ValueError(
            "Agent names must be unique."
        )

    if not (
        1
        <= k
        < len(
            names
        )
    ):
        raise ValueError(
            "k must satisfy "
            "1 <= k < number of agents."
        )

    sources = {}

    for observer in names:
        candidates = [
            name
            for name
            in names
            if name != observer
        ]

        chosen = rng.choice(
            candidates,
            size=k,
            replace=False,
        )

        sources[
            observer
        ] = [
            str(
                name
            )
            for name
            in chosen
        ]

    return sources


def observed_low_fraction(
    observer: str,
    sources: dict[
        str,
        list[str],
    ],
    actions: dict[
        str,
        int,
    ]
    | None,
    *,
    neutral: float = 0.5,
) -> float:
    """
    Return the fraction of an observer's sources choosing low extraction.

    If no previous actions exist yet, return the neutral value. This is
    used at the first timestep before any peer behavior has been
    observed.

    The default neutral value 0.5 maps to the middle social state.
    """

    if not (
        0.0
        <= neutral
        <= 1.0
    ):
        raise ValueError(
            "neutral must be between 0 and 1."
        )

    observer_sources = (
        sources[
            observer
        ]
    )

    if (
        actions is None
        or not observer_sources
    ):
        return float(
            neutral
        )

    return float(
        np.mean(
            [
                actions[
                    source
                ]
                == LOW_EXTRACT
                for source
                in observer_sources
            ]
        )
    )


def social_bin(
    low_fraction: float,
) -> int:
    """
    Discretize observed low-extraction frequency into three states.

        0: mostly high extraction
           low_fraction < 1/3

        1: mixed
           1/3 <= low_fraction < 2/3

        2: mostly low extraction
           low_fraction >= 2/3
    """

    value = float(
        low_fraction
    )

    if not (
        0.0
        <= value
        <= 1.0
    ):
        raise ValueError(
            "low_fraction must be between 0 and 1."
        )

    if (
        value
        < 1.0 / 3.0
    ):
        return 0

    if (
        value
        < 2.0 / 3.0
    ):
        return 1

    return 2


def joint_state(
    ecological_state: int,
    social_state: int,
) -> int:
    """
    Combine one three-level ecological state and one three-level social
    state into a nine-state index.

        joint = 3 * ecological_state + social_state

    The resulting states are numbered 0 through 8.
    """

    ecological_state = int(
        ecological_state
    )

    social_state = int(
        social_state
    )

    if not (
        0
        <= ecological_state
        < 3
    ):
        raise ValueError(
            "ecological_state must be 0, 1, or 2."
        )

    if not (
        0
        <= social_state
        < 3
    ):
        raise ValueError(
            "social_state must be 0, 1, or 2."
        )

    return (
        3
        * ecological_state
        + social_state
    )


def visibility_counts(
    sources: dict[
        str,
        list[str],
    ],
) -> dict[
    str,
    int,
]:
    """
    Count how many observers see each agent.

    This is the social visibility degree.

    For a fixed-k network with N agents:

        sum(visibility_counts.values()) = N * k
    """

    counts = Counter(
        {
            name: 0
            for name
            in sources
        }
    )

    for observer_sources in (
        sources.values()
    ):
        counts.update(
            observer_sources
        )

    return dict(
        counts
    )