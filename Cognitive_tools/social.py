"""
Social observation and decentralized rewiring primitives for cognitive_tools.

The social layer is separate from the ecological model. The mapping

    sources[observer]

contains the agents whose actions that observer can see. Conceptually,
information flows:

    source -> observer

Two initial social-network families are supported:

``random_k``
    Every observer has exactly k distinct information sources. This is the
    rewiring-compatible architecture used by the S1 and R0-R3 treatments.

``ba``
    A fixed undirected Barabasi-Albert-style network represented as symmetric
    observation lists. Degree, and therefore attention/visibility, is
    heterogeneous. This is the S2 visibility-skew control.

Scientific provenance
---------------------
Restricted network-based social observation is inspired by Schrama, Tilman,
and Vasconcelos (2025). The fixed-k local/global replacement mechanism is
adapted from Oh and Schauf (2025). The prediction-error trigger and the
matched-random-turnover control are project mechanisms.

No source code from either paper is copied here.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .model import HIGH_EXTRACT, LOW_EXTRACT


SOCIAL_STATE_NAMES = (
    "mostly_high",
    "mixed",
    "mostly_low",
)

SOCIAL_NETWORK_MODES = (
    "random_k",
    "ba",
)

REWIRING_MODES = (
    "none",
    "random",
    "random_matched",
    "prediction_error",
)


# ---------------------------------------------------------------------
# Basic network utilities
# ---------------------------------------------------------------------


def copy_sources(
    sources: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Return an independent copy of an attention network."""

    return {
        observer: list(observer_sources)
        for observer, observer_sources in sources.items()
    }


def _validate_agent_names(
    agents: list[str],
) -> list[str]:
    names = list(agents)

    if len(set(names)) != len(names):
        raise ValueError("Agent names must be unique.")

    return names


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

    names = _validate_agent_names(agents)

    if not 1 <= k < len(names):
        raise ValueError(
            "k must satisfy 1 <= k < number of agents."
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


def init_barabasi_albert_attention(
    agents: list[str],
    m: int,
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """
    Create a fixed Barabasi-Albert-style observational network.

    The graph is undirected, but it is returned in the project's standard
    ``sources[observer]`` representation. Therefore every undirected social
    tie appears in both observation lists.

    Construction
    ------------
    1. Start from a complete graph on ``m + 1`` agents.
    2. Add remaining agents sequentially.
    3. Each new agent chooses ``m`` distinct existing agents with probability
       proportional to current degree.

    This produces heterogeneous degree and therefore heterogeneous attention
    and visibility. It is intended as the S2 visibility-skew treatment.

    ``m=2`` gives mean degree close to 4 for moderate/large N and is therefore
    a useful comparison with the default S1 setting ``k=4``. This is a
    cognitive_tools control choice, not a claim that it reproduces the
    numerical network density used by Schrama et al.
    """

    names = _validate_agent_names(agents)
    n_agents = len(names)

    if not 1 <= m < n_agents:
        raise ValueError(
            "m must satisfy 1 <= m < number of agents."
        )

    seed_size = m + 1

    if seed_size > n_agents:
        raise ValueError(
            "Barabasi-Albert initialization requires at least m + 1 agents."
        )

    adjacency: list[set[int]] = [
        set()
        for _ in names
    ]

    # Initial clique.
    for first in range(seed_size):
        for second in range(first + 1, seed_size):
            adjacency[first].add(second)
            adjacency[second].add(first)

    degrees = np.asarray(
        [
            len(neighbors)
            for neighbors in adjacency
        ],
        dtype=float,
    )

    for newcomer in range(seed_size, n_agents):
        existing_degrees = degrees[:newcomer]
        degree_total = float(existing_degrees.sum())

        if degree_total <= 0.0:
            raise RuntimeError(
                "Preferential attachment requires positive existing degree."
            )

        probabilities = existing_degrees / degree_total

        targets = rng.choice(
            newcomer,
            size=m,
            replace=False,
            p=probabilities,
        )

        for raw_target in targets:
            target = int(raw_target)
            adjacency[newcomer].add(target)
            adjacency[target].add(newcomer)
            degrees[target] += 1.0

        degrees[newcomer] = float(m)

    return {
        names[observer_index]: [
            names[source_index]
            for source_index in sorted(adjacency[observer_index])
        ]
        for observer_index in range(n_agents)
    }


def network_edges(
    sources: dict[str, list[str]],
) -> set[tuple[str, str]]:
    """
    Return directed social edges as ``(source, observer)``.

    This orientation matches the conceptual direction of information flow.
    """

    return {
        (source, observer)
        for observer, observer_sources in sources.items()
        for source in observer_sources
    }


def visibility_counts(
    sources: dict[str, list[str]],
) -> dict[str, int]:
    """Count how many observers see each agent."""

    counts = Counter(
        {
            name: 0
            for name in sources
        }
    )

    for observer_sources in sources.values():
        counts.update(observer_sources)

    return dict(counts)


def network_turnover(
    before: dict[str, list[str]],
    after: dict[str, list[str]],
) -> float:
    """
    Jaccard edge turnover between two directed attention networks.

        retention = |E_before intersection E_after| / |E_before union E_after|
        turnover  = 1 - retention
    """

    before_edges = network_edges(before)
    after_edges = network_edges(after)
    union = before_edges | after_edges

    if not union:
        return 0.0

    intersection = before_edges & after_edges

    return float(
        1.0
        - len(intersection) / len(union)
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
    Fraction of an observer's information sources choosing low extraction.

    If no previous action information exists, return ``neutral``.
    """

    if not 0.0 <= neutral <= 1.0:
        raise ValueError(
            "neutral must be between 0 and 1."
        )

    observer_sources = sources[observer]

    if actions is None or not observer_sources:
        return float(neutral)

    return float(
        np.mean(
            [
                actions[source] == LOW_EXTRACT
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
    """Return observed low-extraction fraction for every observer."""

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
    """Discretize a low-extraction fraction into three social states."""

    value = float(low_fraction)

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
    """Combine ecological and social states into a nine-state index."""

    ecological_state = int(ecological_state)
    social_state = int(social_state)

    if not 0 <= ecological_state < 3:
        raise ValueError(
            "ecological_state must be 0, 1, or 2."
        )

    if not 0 <= social_state < 3:
        raise ValueError(
            "social_state must be 0, 1, or 2."
        )

    return 3 * ecological_state + social_state


# ---------------------------------------------------------------------
# Local/global replacement search
# ---------------------------------------------------------------------


def local_candidates(
    observer: str,
    sources: dict[str, list[str]],
) -> list[str]:
    """
    Legal local replacement candidates.

    Local search looks at the information sources of the observer's current
    information sources. Current sources and the observer itself are excluded.
    """

    current = set(sources[observer])

    two_hop = {
        candidate
        for source in current
        for candidate in sources[source]
    }

    return sorted(
        two_hop
        - current
        - {observer}
    )


def global_candidates(
    observer: str,
    sources: dict[str, list[str]],
) -> list[str]:
    """Return legal population-wide replacement candidates."""

    current = set(sources[observer])

    return sorted(
        set(sources)
        - current
        - {observer}
    )


def replace_source(
    observer: str,
    drop: str,
    sources: dict[str, list[str]],
    *,
    theta: float,
    rng: np.random.Generator,
    candidate_sources: dict[str, list[str]] | None = None,
) -> dict[str, object] | None:
    """Replace one information source while preserving attention capacity."""

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
            f"{drop} is not currently observed by {observer}."
        )

    search_network = (
        sources
        if candidate_sources is None
        else candidate_sources
    )

    requested_global = rng.random() < theta
    requested_scope = "global" if requested_global else "local"

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

    if not pool and requested_scope == "local":
        pool = global_candidates(
            observer,
            search_network,
        )
        fallback = True
        used_scope = "global"

    if not pool:
        return None

    replacement = pool[
        int(rng.integers(len(pool)))
    ]

    updated = list(sources[observer])
    updated[updated.index(drop)] = replacement

    if len(updated) != len(set(updated)):
        raise RuntimeError(
            "Rewiring produced duplicate sources."
        )

    if observer in updated:
        raise RuntimeError(
            "Rewiring produced a self-link."
        )

    sources[observer] = updated

    return {
        "observer": observer,
        "dropped": drop,
        "added": replacement,
        "requested_scope": requested_scope,
        "used_scope": used_scope,
        "fallback": fallback,
    }


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------


def prediction_errors(
    forecasts: dict[str, float],
    observed: dict[str, float],
) -> dict[str, float]:
    """Absolute prediction error for every observer."""

    if set(forecasts) != set(observed):
        raise ValueError(
            "forecasts and observed must have the same agent names."
        )

    return {
        name: abs(
            float(observed[name])
            - float(forecasts[name])
        )
        for name in forecasts
    }


def update_forecasts(
    forecasts: dict[str, float],
    observed: dict[str, float],
    *,
    alpha: float,
) -> None:
    """Apply the EWMA social forecast update in place."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    if set(forecasts) != set(observed):
        raise ValueError(
            "forecasts and observed must have the same agent names."
        )

    for name in forecasts:
        forecasts[name] = (
            (1.0 - alpha) * float(forecasts[name])
            + alpha * float(observed[name])
        )


# ---------------------------------------------------------------------
# Rewiring epoch
# ---------------------------------------------------------------------


def _apply_one_rewire(
    observer: str,
    sources: dict[str, list[str]],
    snapshot: dict[str, list[str]],
    *,
    theta: float,
    rng: np.random.Generator,
    trigger: str,
    prediction_error: float,
) -> dict[str, object] | None:
    observer_sources = snapshot[observer]

    if not observer_sources:
        return None

    drop = observer_sources[
        int(rng.integers(len(observer_sources)))
    ]

    event = replace_source(
        observer,
        drop,
        sources,
        theta=theta,
        rng=rng,
        candidate_sources=snapshot,
    )

    if event is None:
        return None

    event["trigger"] = trigger
    event["prediction_error"] = prediction_error

    return event


def _random_matched_observers(
    snapshot: dict[str, list[str]],
    *,
    target_count: int,
    rng: np.random.Generator,
) -> list[str]:
    """
    Select exactly ``target_count`` observers that can legally replace a tie.
    """

    if target_count < 0:
        raise ValueError(
            "target_count must be non-negative."
        )

    eligible = [
        observer
        for observer in sorted(snapshot)
        if snapshot[observer]
        and global_candidates(observer, snapshot)
    ]

    if target_count > len(eligible):
        raise ValueError(
            "Matched rewiring target exceeds the number of observers "
            "with at least one legal replacement."
        )

    if target_count == 0:
        return []

    selected = rng.choice(
        eligible,
        size=target_count,
        replace=False,
    )

    return [
        str(observer)
        for observer in selected
    ]


def rewire_epoch(
    sources: dict[str, list[str]],
    *,
    mode: str,
    theta: float,
    mu: float,
    rng: np.random.Generator,
    prediction_error_values: dict[str, float] | None = None,
    threshold: float = 0.25,
    target_count: int | None = None,
) -> list[dict[str, object]]:
    """
    Apply one decentralized rewiring checkpoint.

    Modes
    -----
    none
        No agent rewires.

    random
        Every observer is eligible and independently rewires with probability
        ``mu``. Retained for exploratory/backward-compatible runs.

    random_matched
        Exactly ``target_count`` legally rewritable observers are sampled
        uniformly without replacement. This is the confirmatory R0 control.
        ``mu`` is not used to determine the event count in this mode.

    prediction_error
        Observer i is eligible when ``prediction_error_i > threshold`` and
        then rewires independently with probability ``mu``.

    All candidate searches use a snapshot of the network at the start of the
    checkpoint. Every successful event replaces exactly one existing source.
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
        if target_count not in (None, 0):
            raise ValueError(
                "target_count is only used by random_matched rewiring."
            )
        return []

    if mode == "prediction_error" and prediction_error_values is None:
        raise ValueError(
            "prediction_error rewiring requires prediction_error_values."
        )

    if (
        prediction_error_values is not None
        and set(prediction_error_values) != set(sources)
    ):
        raise ValueError(
            "prediction_error_values must contain every observer."
        )

    if mode == "random_matched" and target_count is None:
        raise ValueError(
            "random_matched rewiring requires target_count."
        )

    if mode != "random_matched" and target_count is not None:
        raise ValueError(
            "target_count is only used by random_matched rewiring."
        )

    snapshot = copy_sources(sources)
    events: list[dict[str, object]] = []

    if mode == "random_matched":
        observers = _random_matched_observers(
            snapshot,
            target_count=int(target_count),
            rng=rng,
        )

        for observer in observers:
            event = _apply_one_rewire(
                observer,
                sources,
                snapshot,
                theta=theta,
                rng=rng,
                trigger="random_matched",
                prediction_error=float("nan"),
            )

            if event is None:
                raise RuntimeError(
                    "A matched-random observer was selected as rewritable but "
                    "no replacement could be completed."
                )

            events.append(event)

        if len(events) != int(target_count):
            raise RuntimeError(
                "Matched-random rewiring failed to realize the requested "
                "number of events."
            )

        return events

    for observer in sorted(snapshot):
        if mode == "random":
            eligible = True
            error = float("nan")
        else:
            error = float(
                prediction_error_values[observer]
            )
            eligible = error > threshold

        if not eligible:
            continue

        if rng.random() >= mu:
            continue

        event = _apply_one_rewire(
            observer,
            sources,
            snapshot,
            theta=theta,
            rng=rng,
            trigger=mode,
            prediction_error=error,
        )

        if event is not None:
            events.append(event)

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

    total = float(values.sum())

    if total <= 0.0:
        return 0.0

    n = len(values)
    index = np.arange(1, n + 1)

    value = (
        2.0 * np.sum(index * values) / (n * total)
        - (n + 1.0) / n
    )

    return float(max(0.0, value))


def population_low_fraction_excluding(
    observer: str,
    actions: dict[str, int],
) -> float:
    """Population low-extraction frequency excluding the focal observer."""

    others = [
        action
        for name, action in actions.items()
        if name != observer
    ]

    if not others:
        return float("nan")

    return float(
        np.mean(
            [
                action == LOW_EXTRACT
                for action in others
            ]
        )
    )


def _majority_label(
    low_fraction: float,
) -> int:
    """Return -1 for high majority, 0 for tie, 1 for low majority."""

    if low_fraction > 0.5:
        return 1

    if low_fraction < 0.5:
        return -1

    return 0


def network_reciprocity(
    sources: dict[str, list[str]],
) -> float:
    """
    Fraction of directed information edges that have a reverse edge.

    Edges use the project convention ``(source, observer)``. An edge is
    reciprocal when the observer also appears as a source of the original
    source. A symmetric undirected network represented as two directed edges
    therefore has reciprocity 1.
    """

    edges = network_edges(sources)

    if not edges:
        return 0.0

    reciprocated = sum(
        (observer, source) in edges
        for source, observer in edges
    )

    return float(
        reciprocated / len(edges)
    )


def visibility_degree_assortativity(
    sources: dict[str, list[str]],
) -> float:
    """
    Pearson assortativity of visibility degree along information edges.

    For every directed edge ``source -> observer`` we correlate the source's
    visibility degree with the observer's visibility degree. This reduces to
    ordinary degree assortativity for symmetric BA-style observation graphs.

    Return NaN when the correlation is undefined because one endpoint degree
    sequence has effectively zero variance.
    """

    edges = sorted(
        network_edges(sources)
    )

    if len(edges) < 2:
        return float("nan")

    visibility = visibility_counts(sources)

    source_degrees = np.asarray(
        [
            visibility[source]
            for source, _
            in edges
        ],
        dtype=float,
    )

    observer_degrees = np.asarray(
        [
            visibility[observer]
            for _, observer
            in edges
        ],
        dtype=float,
    )

    if (
        np.std(source_degrees) <= 1e-12
        or np.std(observer_degrees) <= 1e-12
    ):
        return float("nan")

    return float(
        np.corrcoef(
            source_degrees,
            observer_degrees,
        )[0, 1]
    )


def observer_social_diagnostics(
    sources: dict[str, list[str]],
    actions: dict[str, int],
) -> dict[str, dict[str, float | bool]]:
    """
    Return focal social-perception diagnostics for every observer.

    The comparison population excludes the focal observer. ``majority_tied``
    is true when either the observer's local sample or the comparison
    population is exactly tied. ``majority_mismatch`` is NaN for those tied
    comparisons and otherwise 0/1.
    """

    if set(actions) != set(sources):
        raise ValueError(
            "actions must contain every social-network agent."
        )

    observed = social_observations(
        sources,
        actions,
    )

    visibility = visibility_counts(
        sources
    )

    diagnostics: dict[
        str,
        dict[str, float | bool],
    ] = {}

    for observer in sources:
        actual = population_low_fraction_excluding(
            observer,
            actions,
        )

        local = float(
            observed[observer]
        )

        if np.isfinite(actual):
            error = abs(
                local - actual
            )
            bias = local - actual

            local_majority = _majority_label(
                local
            )
            actual_majority = _majority_label(
                actual
            )

            tied = (
                local_majority == 0
                or actual_majority == 0
            )

            mismatch = (
                float("nan")
                if tied
                else float(
                    local_majority
                    != actual_majority
                )
            )
        else:
            error = float("nan")
            bias = float("nan")
            tied = False
            mismatch = float("nan")

        diagnostics[observer] = {
            "observed_low_fraction": local,
            "population_low_fraction_excluding": float(actual),
            "perception_error": float(error),
            "signed_perception_bias": float(bias),
            "majority_tied": bool(tied),
            "majority_mismatch": float(mismatch),
            "visibility_degree": float(
                visibility[observer]
            ),
        }

    return diagnostics


def social_metrics(
    sources: dict[str, list[str]],
    actions: dict[str, int] | None,
) -> dict[str, float]:
    """
    Compute social-network and perception diagnostics.

    Perception metrics compare each observer's information neighborhood with
    the rest of the population excluding that observer. Majority mismatch is
    computed only for non-tied comparisons; ``majority_tie_rate`` reports the
    excluded share explicitly.
    """

    visibility = visibility_counts(sources)

    degrees = np.asarray(
        [
            visibility[name]
            for name in sources
        ],
        dtype=float,
    )

    total_edges = float(degrees.sum())

    result = {
        "visibility_gini": _gini_nonnegative(degrees),
        "max_visibility_share": (
            float(degrees.max() / total_edges)
            if total_edges > 0.0
            else 0.0
        ),
        "zero_visibility_fraction": float(
            np.mean(degrees == 0.0)
        ),
        "reciprocity": network_reciprocity(
            sources
        ),
        "degree_assortativity": (
            visibility_degree_assortativity(
                sources
            )
        ),
        "population_low_fraction": float("nan"),
        "visible_low_fraction": float("nan"),
        "visible_population_bias": float("nan"),
        "mean_perception_error": float("nan"),
        "signed_perception_bias": float("nan"),
        "majority_mismatch_rate": float("nan"),
        "majority_tie_rate": float("nan"),
        "degree_action_correlation": float("nan"),
    }

    if actions is None:
        return result

    population_low = float(
        np.mean(
            [
                action == LOW_EXTRACT
                for action in actions.values()
            ]
        )
    )

    visible_actions = [
        actions[source] == LOW_EXTRACT
        for observer_sources in sources.values()
        for source in observer_sources
    ]

    visible_low = (
        float(np.mean(visible_actions))
        if visible_actions
        else float("nan")
    )

    focal = observer_social_diagnostics(
        sources,
        actions,
    )

    absolute_errors = [
        float(row["perception_error"])
        for row in focal.values()
        if np.isfinite(
            float(row["perception_error"])
        )
    ]

    signed_biases = [
        float(row["signed_perception_bias"])
        for row in focal.values()
        if np.isfinite(
            float(row["signed_perception_bias"])
        )
    ]

    majority_ties = [
        bool(row["majority_tied"])
        for row in focal.values()
        if np.isfinite(
            float(
                row["population_low_fraction_excluding"]
            )
        )
    ]

    majority_mismatches = [
        float(row["majority_mismatch"])
        for row in focal.values()
        if np.isfinite(
            float(row["majority_mismatch"])
        )
    ]

    high_actions = np.asarray(
        [
            actions[name] == HIGH_EXTRACT
            for name in sources
        ],
        dtype=float,
    )

    if (
        len(degrees) > 1
        and np.std(degrees) > 1e-12
        and np.std(high_actions) > 1e-12
    ):
        degree_action_correlation = float(
            np.corrcoef(
                degrees,
                high_actions,
            )[0, 1]
        )
    else:
        degree_action_correlation = float("nan")

    result.update(
        {
            "population_low_fraction": population_low,
            "visible_low_fraction": visible_low,
            "visible_population_bias": (
                visible_low - population_low
                if np.isfinite(visible_low)
                else float("nan")
            ),
            "mean_perception_error": (
                float(np.mean(absolute_errors))
                if absolute_errors
                else float("nan")
            ),
            "signed_perception_bias": (
                float(np.mean(signed_biases))
                if signed_biases
                else float("nan")
            ),
            "majority_mismatch_rate": (
                float(np.mean(majority_mismatches))
                if majority_mismatches
                else float("nan")
            ),
            "majority_tie_rate": (
                float(np.mean(majority_ties))
                if majority_ties
                else float("nan")
            ),
            "degree_action_correlation": (
                degree_action_correlation
            ),
        }
    )

    return result
