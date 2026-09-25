"""
Social observation and decentralized rewiring primitives for cognitive_tools.

The social layer is deliberately separate from the ecological model.

The mapping

    sources[observer]

contains the agents whose actions that observer can see.

Conceptually, information flows:

    source -> observer

Every observer has a fixed attention capacity k. Rewiring replaces one
information source with another; it does not add or remove total
attention.

Scientific provenance
---------------------
Restricted social observation is inspired by:

    Schrama, N., Tilman, A. R., & Vasconcelos, V. V. (2025).
    "Majority illusion drives the spontaneous emergence of alternative
    states in common-pool resource games with network-based information."
    iScience, 28(7), 112831.
    https://doi.org/10.1016/j.isci.2025.112831

The fixed-k local/global replacement mechanism is adapted from:

    Oh, P., & Schauf, A. (2025).
    "Self-organizing group structure through rewiring for collective
    decision-making in evolving environments."
    Scientific Reports, 15, 39947.
    https://doi.org/10.1038/s41598-025-23634-3

Oh & Schauf remove information sources that produced objectively
incorrect opinions. That removal rule is not transferable directly to
this CPR model because low and high extraction actions do not have
externally revealed binary correctness.

The prediction-error trigger implemented in cognitive_tools is therefore
a project mechanism. It measures local social surprise:

    error_i(t)
    =
    |observed_i(t) - forecast_i(t)|

It is not a misinformation detector and does not measure objective
truthfulness.

No source code from either paper is copied here.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .model import (
    HIGH_EXTRACT,
    LOW_EXTRACT,
)


SOCIAL_STATE_NAMES = (
    "mostly_high",
    "mixed",
    "mostly_low",
)

REWIRING_MODES = (
    "none",
    "random",
    "prediction_error",
)


# ---------------------------------------------------------------------
# Basic network utilities
# ---------------------------------------------------------------------


def copy_sources(
    sources: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Return an independent copy of an attention network.
    """

    return {
        observer: list(observer_sources)
        for observer, observer_sources in sources.items()
    }


def init_random_attention(
    agents: list[str],
    k: int,
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """
    Create a directed random fixed-k attention network.

    Every observer receives exactly k distinct information sources.
    Self-observation is excluded.
    """

    names = list(agents)

    if len(set(names)) != len(names):
        raise ValueError(
            "Agent names must be unique."
        )

    if not 1 <= k < len(names):
        raise ValueError(
            "k must satisfy "
            "1 <= k < number of agents."
        )

    sources: dict[str, list[str]] = {}

    for observer in names:
        candidates = [
            name
            for name in names
            if name != observer
        ]

        chosen = rng.choice(
            candidates,
            size=k,
            replace=False,
        )

        sources[observer] = [
            str(name)
            for name in chosen
        ]

    return sources


def network_edges(
    sources: dict[str, list[str]],
) -> set[tuple[str, str]]:
    """
    Return directed social edges as:

        (source, observer)

    This orientation matches the conceptual direction of information
    flow.
    """

    return {
        (
            source,
            observer,
        )
        for observer, observer_sources in sources.items()
        for source in observer_sources
    }


def visibility_counts(
    sources: dict[str, list[str]],
) -> dict[str, int]:
    """
    Count how many observers see each agent.

    This is the social visibility degree.

    For fixed attention capacity k:

        sum(visibility degree) = N * k
    """

    counts = Counter(
        {
            name: 0
            for name in sources
        }
    )

    for observer_sources in sources.values():
        counts.update(
            observer_sources
        )

    return dict(counts)


def network_turnover(
    before: dict[str, list[str]],
    after: dict[str, list[str]],
) -> float:
    """
    Jaccard edge turnover between two directed attention networks.

        retention
        =
        |E_before intersection E_after|
        /
        |E_before union E_after|

        turnover
        =
        1 - retention
    """

    before_edges = network_edges(
        before
    )

    after_edges = network_edges(
        after
    )

    union = (
        before_edges
        | after_edges
    )

    if not union:
        return 0.0

    intersection = (
        before_edges
        & after_edges
    )

    return float(
        1.0
        - (
            len(intersection)
            / len(union)
        )
    )


# ---------------------------------------------------------------------
# Social observation
# ---------------------------------------------------------------------


def observed_low_fraction(
    observer: str,
    sources: dict[str, list[str]],
    actions: dict[str, int] | None,
    *,
    neutral: float = 0.5,
) -> float:
    """
    Fraction of an observer's information sources choosing low
    extraction.

    If no previous action information exists, return the neutral value.
    The default 0.5 maps to the middle social state.
    """

    if not 0.0 <= neutral <= 1.0:
        raise ValueError(
            "neutral must be between 0 and 1."
        )

    observer_sources = (
        sources[observer]
    )

    if (
        actions is None
        or not observer_sources
    ):
        return float(neutral)

    return float(
        np.mean(
            [
                actions[source]
                == LOW_EXTRACT
                for source in observer_sources
            ]
        )
    )


def social_observations(
    sources: dict[str, list[str]],
    actions: dict[str, int] | None,
    *,
    neutral: float = 0.5,
) -> dict[str, float]:
    """
    Return the observed low-extraction fraction for every observer.
    """

    return {
        observer: observed_low_fraction(
            observer,
            sources,
            actions,
            neutral=neutral,
        )
        for observer in sources
    }


def social_bin(
    low_fraction: float,
) -> int:
    """
    Discretize observed low-extraction frequency.

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

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            "low_fraction must be between 0 and 1."
        )

    if value < 1.0 / 3.0:
        return 0

    if value < 2.0 / 3.0:
        return 1

    return 2


def joint_state(
    ecological_state: int,
    social_state: int,
) -> int:
    """
    Combine ecological and social state into a nine-state index.

        joint_state
        =
        3 * ecological_state
        + social_state
    """

    ecological_state = int(
        ecological_state
    )

    social_state = int(
        social_state
    )

    if not 0 <= ecological_state < 3:
        raise ValueError(
            "ecological_state must be 0, 1, or 2."
        )

    if not 0 <= social_state < 3:
        raise ValueError(
            "social_state must be 0, 1, or 2."
        )

    return (
        3
        * ecological_state
        + social_state
    )


# ---------------------------------------------------------------------
# Local/global replacement search
# ---------------------------------------------------------------------


def local_candidates(
    observer: str,
    sources: dict[str, list[str]],
) -> list[str]:
    """
    Legal local replacement candidates.

    Local search looks at the information sources of the observer's
    current information sources.

    Current sources and the observer itself are excluded.
    """

    current = set(
        sources[observer]
    )

    two_hop = {
        candidate
        for source in current
        for candidate in sources[source]
    }

    return sorted(
        two_hop
        - current
        - {
            observer,
        }
    )


def global_candidates(
    observer: str,
    sources: dict[str, list[str]],
) -> list[str]:
    """
    Legal global replacement candidates.

    Any agent not already observed, excluding the observer itself, is
    eligible.
    """

    current = set(
        sources[observer]
    )

    return sorted(
        set(sources)
        - current
        - {
            observer,
        }
    )


def replace_source(
    observer: str,
    drop: str,
    sources: dict[str, list[str]],
    *,
    theta: float,
    rng: np.random.Generator,
    candidate_sources: (
        dict[str, list[str]]
        | None
    ) = None,
) -> dict[str, object] | None:
    """
    Replace one information source while preserving attention capacity.

    Parameters
    ----------
    observer
        Agent changing one attention tie.

    drop
        Existing source to remove.

    sources
        Network that will be modified.

    theta
        Probability of global rather than local search.

        theta = 0
            local search

        theta = 1
            global search

    rng
        Dedicated rewiring RNG.

    candidate_sources
        Optional network snapshot used only to construct the candidate
        pool. This allows all agents at a rewiring checkpoint to search
        the same pre-rewiring network.

    Returns
    -------
    dict or None
        Description of the replacement event, or None if no legal
        replacement exists.

    Notes
    -----
    If local search produces no legal candidate, the function explicitly
    falls back to global search.

    This fallback prevents a locally closed information neighborhood from
    making rewiring impossible solely because there is no legal two-hop
    candidate.
    """

    if not 0.0 <= theta <= 1.0:
        raise ValueError(
            "theta must be between 0 and 1."
        )

    if observer not in sources:
        raise KeyError(
            f"Unknown observer: {observer}"
        )

    if drop not in sources[observer]:
        raise ValueError(
            f"{drop} is not currently observed "
            f"by {observer}."
        )

    search_network = (
        sources
        if candidate_sources is None
        else candidate_sources
    )

    requested_global = (
        rng.random()
        < theta
    )

    requested_scope = (
        "global"
        if requested_global
        else "local"
    )

    if requested_global:
        pool = global_candidates(
            observer,
            search_network,
        )
    else:
        pool = local_candidates(
            observer,
            search_network,
        )

    fallback = False
    used_scope = requested_scope

    if (
        not pool
        and requested_scope == "local"
    ):
        pool = global_candidates(
            observer,
            search_network,
        )

        fallback = True
        used_scope = "global"

    if not pool:
        return None

    replacement = pool[
        int(
            rng.integers(
                len(pool)
            )
        )
    ]

    updated = list(
        sources[observer]
    )

    updated[
        updated.index(
            drop
        )
    ] = replacement

    if len(updated) != len(set(updated)):
        raise RuntimeError(
            "Rewiring produced duplicate sources."
        )

    if observer in updated:
        raise RuntimeError(
            "Rewiring produced a self-link."
        )

    sources[observer] = (
        updated
    )

    return {
        "observer": observer,
        "dropped": drop,
        "added": replacement,
        "requested_scope": (
            requested_scope
        ),
        "used_scope": (
            used_scope
        ),
        "fallback": fallback,
    }


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------


def prediction_errors(
    forecasts: dict[str, float],
    observed: dict[str, float],
) -> dict[str, float]:
    """
    Absolute prediction error for every observer.

        error_i
        =
        |observed_i - forecast_i|
    """

    if set(forecasts) != set(observed):
        raise ValueError(
            "forecasts and observed must have "
            "the same agent names."
        )

    return {
        name: abs(
            float(
                observed[name]
            )
            - float(
                forecasts[name]
            )
        )
        for name in forecasts
    }


def update_forecasts(
    forecasts: dict[str, float],
    observed: dict[str, float],
    *,
    alpha: float,
) -> None:
    """
    Apply the EWMA social forecast update.

        forecast_i(t+1)
        =
        (1-alpha) forecast_i(t)
        + alpha observed_i(t)
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    if set(forecasts) != set(observed):
        raise ValueError(
            "forecasts and observed must have "
            "the same agent names."
        )

    for name in forecasts:
        forecasts[name] = (
            (
                1.0
                - alpha
            )
            * float(
                forecasts[name]
            )
            + alpha
            * float(
                observed[name]
            )
        )


# ---------------------------------------------------------------------
# Rewiring epoch
# ---------------------------------------------------------------------


def rewire_epoch(
    sources: dict[str, list[str]],
    *,
    mode: str,
    theta: float,
    mu: float,
    rng: np.random.Generator,
    prediction_error_values: (
        dict[str, float]
        | None
    ) = None,
    threshold: float = 0.25,
) -> list[dict[str, object]]:
    """
    Apply one decentralized rewiring checkpoint.

    Modes
    -----
    none
        No agent rewires.

    random
        Every observer is eligible. Each rewires independently with
        probability mu.

    prediction_error
        Observer i is eligible when:

            prediction_error_i > threshold

        and then rewires independently with probability mu.

    Every successful event replaces exactly one existing source.

    Candidate search is based on a snapshot of the network at the start
    of the rewiring checkpoint. Therefore agents participating in the
    same checkpoint do not receive an artificial search advantage from
    changes made earlier in the Python iteration order.
    """

    if mode not in REWIRING_MODES:
        raise ValueError(
            f"Unknown rewiring mode: {mode}"
        )

    if not 0.0 <= theta <= 1.0:
        raise ValueError(
            "theta must be between 0 and 1."
        )

    if not 0.0 <= mu <= 1.0:
        raise ValueError(
            "mu must be between 0 and 1."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    if mode == "none":
        return []

    if (
        mode == "prediction_error"
        and prediction_error_values is None
    ):
        raise ValueError(
            "prediction_error rewiring requires "
            "prediction_error_values."
        )

    if (
        prediction_error_values is not None
        and set(prediction_error_values)
        != set(sources)
    ):
        raise ValueError(
            "prediction_error_values must contain "
            "every observer."
        )

    snapshot = copy_sources(
        sources
    )

    events: list[
        dict[str, object]
    ] = []

    for observer in sorted(
        snapshot
    ):
        if mode == "random":
            eligible = True
            error = float(
                "nan"
            )

        else:
            error = float(
                prediction_error_values[
                    observer
                ]
            )

            eligible = (
                error
                > threshold
            )

        if not eligible:
            continue

        if rng.random() >= mu:
            continue

        observer_sources = (
            snapshot[observer]
        )

        if not observer_sources:
            continue

        drop = observer_sources[
            int(
                rng.integers(
                    len(
                        observer_sources
                    )
                )
            )
        ]

        event = replace_source(
            observer,
            drop,
            sources,
            theta=theta,
            rng=rng,
            candidate_sources=(
                snapshot
            ),
        )

        if event is None:
            continue

        event[
            "trigger"
        ] = mode

        event[
            "prediction_error"
        ] = error

        events.append(
            event
        )

    return events


# ---------------------------------------------------------------------
# Social diagnostics
# ---------------------------------------------------------------------


def _gini_nonnegative(
    values,
) -> float:
    values = np.sort(
        np.asarray(
            values,
            dtype=float,
        )
    )

    if len(values) == 0:
        return 0.0

    total = float(
        values.sum()
    )

    if total <= 0.0:
        return 0.0

    n = len(values)

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
            n
            + 1.0
        )
        / n
    )

    return float(
        max(
            0.0,
            value,
        )
    )


def population_low_fraction_excluding(
    observer: str,
    actions: dict[str, int],
) -> float:
    """
    Population low-extraction frequency excluding the focal observer.
    """

    others = [
        action
        for name, action
        in actions.items()
        if name != observer
    ]

    if not others:
        return float(
            "nan"
        )

    return float(
        np.mean(
            [
                action
                == LOW_EXTRACT
                for action in others
            ]
        )
    )


def _majority_label(
    low_fraction: float,
) -> int:
    """
    -1 = high-extraction majority
     0 = exact tie
     1 = low-extraction majority
    """

    if low_fraction > 0.5:
        return 1

    if low_fraction < 0.5:
        return -1

    return 0


def social_metrics(
    sources: dict[str, list[str]],
    actions: dict[str, int] | None,
) -> dict[str, float]:
    """
    Compute inexpensive social-network and perception diagnostics.

    Structural metrics are available without actions.

    Perception metrics compare each observer's information neighborhood
    with the rest of the population excluding that observer.
    """

    visibility = (
        visibility_counts(
            sources
        )
    )

    degrees = np.asarray(
        [
            visibility[name]
            for name in sources
        ],
        dtype=float,
    )

    total_edges = float(
        degrees.sum()
    )

    visibility_gini = (
        _gini_nonnegative(
            degrees
        )
    )

    max_visibility_share = (
        float(
            degrees.max()
            / total_edges
        )
        if total_edges > 0.0
        else 0.0
    )

    zero_visibility_fraction = float(
        np.mean(
            degrees
            == 0.0
        )
    )

    result = {
        "visibility_gini": (
            visibility_gini
        ),
        "max_visibility_share": (
            max_visibility_share
        ),
        "zero_visibility_fraction": (
            zero_visibility_fraction
        ),
        "population_low_fraction": float(
            "nan"
        ),
        "visible_low_fraction": float(
            "nan"
        ),
        "mean_perception_error": float(
            "nan"
        ),
        "signed_perception_bias": float(
            "nan"
        ),
        "majority_mismatch_rate": float(
            "nan"
        ),
        "degree_action_correlation": float(
            "nan"
        ),
    }

    if actions is None:
        return result

    population_low = float(
        np.mean(
            [
                action
                == LOW_EXTRACT
                for action
                in actions.values()
            ]
        )
    )

    visible_actions = [
        actions[source]
        == LOW_EXTRACT
        for observer_sources
        in sources.values()
        for source
        in observer_sources
    ]

    visible_low = (
        float(
            np.mean(
                visible_actions
            )
        )
        if visible_actions
        else float(
            "nan"
        )
    )

    observed = (
        social_observations(
            sources,
            actions,
        )
    )

    absolute_errors = []
    signed_biases = []
    majority_mismatches = []

    for observer in sources:
        actual = (
            population_low_fraction_excluding(
                observer,
                actions,
            )
        )

        if not np.isfinite(
            actual
        ):
            continue

        local = float(
            observed[
                observer
            ]
        )

        absolute_errors.append(
            abs(
                local
                - actual
            )
        )

        signed_biases.append(
            local
            - actual
        )

        local_majority = (
            _majority_label(
                local
            )
        )

        actual_majority = (
            _majority_label(
                actual
            )
        )

        if (
            local_majority != 0
            and actual_majority != 0
        ):
            majority_mismatches.append(
                local_majority
                != actual_majority
            )

    high_actions = np.asarray(
        [
            actions[name]
            == HIGH_EXTRACT
            for name in sources
        ],
        dtype=float,
    )

    if (
        len(degrees) > 1
        and np.std(
            degrees
        ) > 1e-12
        and np.std(
            high_actions
        ) > 1e-12
    ):
        degree_action_correlation = float(
            np.corrcoef(
                degrees,
                high_actions,
            )[
                0,
                1,
            ]
        )

    else:
        degree_action_correlation = float(
            "nan"
        )

    result.update(
        {
            "population_low_fraction": (
                population_low
            ),
            "visible_low_fraction": (
                visible_low
            ),
            "mean_perception_error": (
                float(
                    np.mean(
                        absolute_errors
                    )
                )
                if absolute_errors
                else float(
                    "nan"
                )
            ),
            "signed_perception_bias": (
                float(
                    np.mean(
                        signed_biases
                    )
                )
                if signed_biases
                else float(
                    "nan"
                )
            ),
            "majority_mismatch_rate": (
                float(
                    np.mean(
                        majority_mismatches
                    )
                )
                if majority_mismatches
                else float(
                    "nan"
                )
            ),
            "degree_action_correlation": (
                degree_action_correlation
            ),
        }
    )

    return result