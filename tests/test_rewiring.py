from __future__ import annotations

import numpy as np
import pytest

from Cognitive_tools.model import (
    HIGH_EXTRACT,
    LOW_EXTRACT,
)
from Cognitive_tools.social import (
    copy_sources,
    global_candidates,
    local_candidates,
    network_edges,
    network_turnover,
    prediction_errors,
    replace_source,
    rewire_epoch,
    social_metrics,
    update_forecasts,
    visibility_counts,
)


# ---------------------------------------------------------------------
# Candidate search
# ---------------------------------------------------------------------


def test_local_candidates_are_sources_of_sources():
    sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_2",
        ],
        "agent_2": [
            "agent_3",
        ],
        "agent_3": [
            "agent_0",
        ],
    }

    assert local_candidates(
        "agent_0",
        sources,
    ) == [
        "agent_2",
    ]


def test_local_candidates_exclude_self_and_current_sources():
    sources = {
        "agent_0": [
            "agent_1",
            "agent_2",
        ],
        "agent_1": [
            "agent_0",
            "agent_2",
            "agent_3",
        ],
        "agent_2": [
            "agent_0",
            "agent_1",
            "agent_3",
        ],
        "agent_3": [
            "agent_0",
            "agent_1",
        ],
    }

    assert local_candidates(
        "agent_0",
        sources,
    ) == [
        "agent_3",
    ]


def test_global_candidates_exclude_self_and_current_sources():
    sources = {
        "agent_0": [
            "agent_1",
            "agent_2",
        ],
        "agent_1": [
            "agent_0",
        ],
        "agent_2": [
            "agent_0",
        ],
        "agent_3": [
            "agent_0",
        ],
        "agent_4": [
            "agent_0",
        ],
    }

    assert global_candidates(
        "agent_0",
        sources,
    ) == [
        "agent_3",
        "agent_4",
    ]


# ---------------------------------------------------------------------
# Source replacement
# ---------------------------------------------------------------------


def test_theta_zero_uses_local_candidate_when_available():
    sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_2",
        ],
        "agent_2": [
            "agent_0",
        ],
        "agent_3": [
            "agent_0",
        ],
    }

    event = replace_source(
        "agent_0",
        "agent_1",
        sources,
        theta=0.0,
        rng=np.random.default_rng(
            1
        ),
    )

    assert event is not None

    assert event[
        "requested_scope"
    ] == "local"

    assert event[
        "used_scope"
    ] == "local"

    assert event[
        "fallback"
    ] is False

    assert sources[
        "agent_0"
    ] == [
        "agent_2",
    ]


def test_theta_one_requests_global_search():
    sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_2",
        ],
        "agent_2": [
            "agent_0",
        ],
        "agent_3": [
            "agent_0",
        ],
    }

    event = replace_source(
        "agent_0",
        "agent_1",
        sources,
        theta=1.0,
        rng=np.random.default_rng(
            2
        ),
    )

    assert event is not None

    assert event[
        "requested_scope"
    ] == "global"

    assert event[
        "used_scope"
    ] == "global"

    assert event[
        "added"
    ] in {
        "agent_2",
        "agent_3",
    }


def test_empty_local_pool_falls_back_to_global():
    sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_0",
        ],
        "agent_2": [
            "agent_0",
        ],
    }

    event = replace_source(
        "agent_0",
        "agent_1",
        sources,
        theta=0.0,
        rng=np.random.default_rng(
            3
        ),
    )

    assert event is not None

    assert event[
        "requested_scope"
    ] == "local"

    assert event[
        "used_scope"
    ] == "global"

    assert event[
        "fallback"
    ] is True

    assert sources[
        "agent_0"
    ] == [
        "agent_2",
    ]


def test_replacement_preserves_attention_capacity():
    sources = {
        "agent_0": [
            "agent_1",
            "agent_2",
        ],
        "agent_1": [
            "agent_0",
            "agent_2",
        ],
        "agent_2": [
            "agent_0",
            "agent_3",
        ],
        "agent_3": [
            "agent_0",
            "agent_1",
        ],
    }

    before_k = len(
        sources[
            "agent_0"
        ]
    )

    event = replace_source(
        "agent_0",
        "agent_1",
        sources,
        theta=1.0,
        rng=np.random.default_rng(
            4
        ),
    )

    assert event is not None

    assert len(
        sources[
            "agent_0"
        ]
    ) == before_k

    assert len(
        set(
            sources[
                "agent_0"
            ]
        )
    ) == before_k

    assert (
        "agent_0"
        not in sources[
            "agent_0"
        ]
    )


def test_invalid_drop_is_rejected():
    sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_0",
        ],
        "agent_2": [
            "agent_0",
        ],
    }

    with pytest.raises(
        ValueError
    ):
        replace_source(
            "agent_0",
            "agent_2",
            sources,
            theta=1.0,
            rng=np.random.default_rng(
                0
            ),
        )


# ---------------------------------------------------------------------
# Network invariants
# ---------------------------------------------------------------------


def test_total_edge_count_is_preserved_after_rewiring():
    agents = [
        f"agent_{index}"
        for index in range(
            12
        )
    ]

    rng = np.random.default_rng(
        42
    )

    from Cognitive_tools.social import (
        init_random_attention,
    )

    sources = init_random_attention(
        agents,
        k=4,
        rng=rng,
    )

    expected_edges = (
        len(
            agents
        )
        * 4
    )

    rewire_rng = (
        np.random.default_rng(
            43
        )
    )

    for _ in range(
        50
    ):
        errors = {
            name: 1.0
            for name
            in agents
        }

        rewire_epoch(
            sources,
            mode="prediction_error",
            theta=0.25,
            mu=1.0,
            rng=rewire_rng,
            prediction_error_values=(
                errors
            ),
            threshold=0.25,
        )

        counts = visibility_counts(
            sources
        )

        assert sum(
            counts.values()
        ) == expected_edges

        assert len(
            network_edges(
                sources
            )
        ) == expected_edges

        for observer in agents:
            assert len(
                sources[
                    observer
                ]
            ) == 4

            assert len(
                set(
                    sources[
                        observer
                    ]
                )
            ) == 4

            assert (
                observer
                not in sources[
                    observer
                ]
            )


def test_network_turnover_identical_network_is_zero():
    sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_0",
        ],
    }

    assert network_turnover(
        sources,
        copy_sources(
            sources
        ),
    ) == pytest.approx(
        0.0
    )


def test_network_turnover_one_replacement():
    before = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_0",
        ],
        "agent_2": [
            "agent_0",
        ],
    }

    after = copy_sources(
        before
    )

    after[
        "agent_0"
    ] = [
        "agent_2",
    ]

    # Before has 3 edges.
    #
    # One removed and one added:
    #
    # intersection = 2
    # union = 4
    # turnover = 1 - 2/4 = 0.5
    assert network_turnover(
        before,
        after,
    ) == pytest.approx(
        0.5
    )


# ---------------------------------------------------------------------
# Rewiring triggers
# ---------------------------------------------------------------------


def test_rewiring_none_changes_nothing():
    sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_2",
        ],
        "agent_2": [
            "agent_0",
        ],
    }

    before = copy_sources(
        sources
    )

    events = rewire_epoch(
        sources,
        mode="none",
        theta=0.25,
        mu=1.0,
        rng=np.random.default_rng(
            0
        ),
    )

    assert events == []

    assert sources == before


def test_mu_zero_changes_nothing():
    sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_2",
        ],
        "agent_2": [
            "agent_0",
        ],
        "agent_3": [
            "agent_0",
        ],
    }

    before = copy_sources(
        sources
    )

    events = rewire_epoch(
        sources,
        mode="random",
        theta=0.25,
        mu=0.0,
        rng=np.random.default_rng(
            0
        ),
    )

    assert events == []

    assert sources == before


def test_random_turnover_can_rewire_all_agents():
    sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_2",
        ],
        "agent_2": [
            "agent_3",
        ],
        "agent_3": [
            "agent_0",
        ],
    }

    events = rewire_epoch(
        sources,
        mode="random",
        theta=1.0,
        mu=1.0,
        rng=np.random.default_rng(
            10
        ),
    )

    assert len(
        events
    ) == 4


def test_prediction_error_requires_threshold_crossing():
    sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_2",
        ],
        "agent_2": [
            "agent_3",
        ],
        "agent_3": [
            "agent_0",
        ],
    }

    errors = {
        "agent_0": 0.50,
        "agent_1": 0.10,
        "agent_2": 0.25,
        "agent_3": 0.00,
    }

    events = rewire_epoch(
        sources,
        mode="prediction_error",
        theta=1.0,
        mu=1.0,
        rng=np.random.default_rng(
            11
        ),
        prediction_error_values=(
            errors
        ),
        threshold=0.25,
    )

    # Strict inequality:
    #
    # error > threshold
    #
    # Only agent_0 qualifies.
    assert len(
        events
    ) == 1

    assert events[
        0
    ][
        "observer"
    ] == "agent_0"


def test_prediction_error_mode_requires_errors():
    sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_0",
        ],
        "agent_2": [
            "agent_0",
        ],
    }

    with pytest.raises(
        ValueError
    ):
        rewire_epoch(
            sources,
            mode="prediction_error",
            theta=0.25,
            mu=0.1,
            rng=np.random.default_rng(
                0
            ),
            prediction_error_values=None,
        )


# ---------------------------------------------------------------------
# Forecast dynamics
# ---------------------------------------------------------------------


def test_prediction_error_is_computed_before_forecast_update():
    forecasts = {
        "agent_0": 0.5,
        "agent_1": 0.5,
    }

    observed = {
        "agent_0": 1.0,
        "agent_1": 0.0,
    }

    errors = prediction_errors(
        forecasts,
        observed,
    )

    assert errors == {
        "agent_0": pytest.approx(
            0.5
        ),
        "agent_1": pytest.approx(
            0.5
        ),
    }

    update_forecasts(
        forecasts,
        observed,
        alpha=0.5,
    )

    assert forecasts[
        "agent_0"
    ] == pytest.approx(
        0.75
    )

    assert forecasts[
        "agent_1"
    ] == pytest.approx(
        0.25
    )


def test_forecast_alpha_zero_preserves_forecast():
    forecasts = {
        "agent_0": 0.5,
    }

    observed = {
        "agent_0": 1.0,
    }

    update_forecasts(
        forecasts,
        observed,
        alpha=0.0,
    )

    assert forecasts[
        "agent_0"
    ] == pytest.approx(
        0.5
    )


def test_forecast_alpha_one_uses_latest_observation():
    forecasts = {
        "agent_0": 0.5,
    }

    observed = {
        "agent_0": 1.0,
    }

    update_forecasts(
        forecasts,
        observed,
        alpha=1.0,
    )

    assert forecasts[
        "agent_0"
    ] == pytest.approx(
        1.0
    )


# ---------------------------------------------------------------------
# Perception diagnostics
# ---------------------------------------------------------------------


def test_social_metrics_detect_known_perception_error():
    sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_0",
        ],
        "agent_2": [
            "agent_0",
        ],
    }

    actions = {
        "agent_0": HIGH_EXTRACT,
        "agent_1": LOW_EXTRACT,
        "agent_2": HIGH_EXTRACT,
    }

    metrics = social_metrics(
        sources,
        actions,
    )

    assert (
        0.0
        <= metrics[
            "visibility_gini"
        ]
        <= 1.0
    )

    assert (
        0.0
        <= metrics[
            "max_visibility_share"
        ]
        <= 1.0
    )

    assert metrics[
        "mean_perception_error"
    ] >= 0.0


# ---------------------------------------------------------------------
# Post-rewire state semantics
# ---------------------------------------------------------------------


def test_post_rewire_network_changes_social_observation():
    from Cognitive_tools.social import (
        observed_low_fraction,
    )

    sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_2",
        ],
        "agent_2": [
            "agent_0",
        ],
    }

    actions = {
        "agent_0": HIGH_EXTRACT,
        "agent_1": LOW_EXTRACT,
        "agent_2": HIGH_EXTRACT,
    }

    before = observed_low_fraction(
        "agent_0",
        sources,
        actions,
    )

    assert before == pytest.approx(
        1.0
    )

    sources[
        "agent_0"
    ] = [
        "agent_2",
    ]

    after = observed_low_fraction(
        "agent_0",
        sources,
        actions,
    )

    assert after == pytest.approx(
        0.0
    )