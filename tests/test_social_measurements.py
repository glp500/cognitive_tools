from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cognitive_tools.model import (
    HIGH_EXTRACT,
    LOW_EXTRACT,
)
from cognitive_tools.social import (
    network_reciprocity,
    observer_social_diagnostics,
    social_metrics,
    visibility_degree_assortativity,
)

import cognitive_tools.experiment as experiment


def test_network_reciprocity_symmetric_network_is_one():
    sources = {
        "agent_0": ["agent_1"],
        "agent_1": ["agent_0"],
    }

    assert network_reciprocity(
        sources
    ) == pytest.approx(
        1.0
    )


def test_network_reciprocity_directed_cycle_is_zero():
    sources = {
        "agent_0": ["agent_1"],
        "agent_1": ["agent_2"],
        "agent_2": ["agent_0"],
    }

    assert network_reciprocity(
        sources
    ) == pytest.approx(
        0.0
    )


def test_visibility_degree_assortativity_symmetric_path():
    # Undirected path 0-1-2-3 represented as symmetric observation lists.
    # Visibility degrees are [1, 2, 2, 1], giving assortativity -0.5.
    sources = {
        "agent_0": ["agent_1"],
        "agent_1": ["agent_0", "agent_2"],
        "agent_2": ["agent_1", "agent_3"],
        "agent_3": ["agent_2"],
    }

    assert visibility_degree_assortativity(
        sources
    ) == pytest.approx(
        -0.5
    )


def test_visibility_degree_assortativity_is_nan_when_degree_constant():
    sources = {
        "agent_0": ["agent_1"],
        "agent_1": ["agent_2"],
        "agent_2": ["agent_0"],
    }

    assert np.isnan(
        visibility_degree_assortativity(
            sources
        )
    )


def test_social_metrics_reports_majority_ties_separately():
    sources = {
        "agent_0": ["agent_1", "agent_2"],
        "agent_1": ["agent_0", "agent_2"],
        "agent_2": ["agent_0", "agent_1"],
    }

    actions = {
        "agent_0": LOW_EXTRACT,
        "agent_1": HIGH_EXTRACT,
        "agent_2": HIGH_EXTRACT,
    }

    metrics = social_metrics(
        sources,
        actions,
    )

    # agent_1 and agent_2 compare against one LOW and one HIGH action,
    # so two of the three majority comparisons are tied and excluded from
    # the mismatch denominator.
    assert metrics[
        "majority_tie_rate"
    ] == pytest.approx(
        2.0 / 3.0
    )

    assert metrics[
        "majority_mismatch_rate"
    ] == pytest.approx(
        0.0
    )


def test_social_metrics_reports_edge_weighted_visibility_bias():
    sources = {
        "agent_0": ["agent_1"],
        "agent_1": ["agent_0"],
        "agent_2": ["agent_0"],
    }

    actions = {
        "agent_0": LOW_EXTRACT,
        "agent_1": HIGH_EXTRACT,
        "agent_2": HIGH_EXTRACT,
    }

    metrics = social_metrics(
        sources,
        actions,
    )

    assert metrics[
        "population_low_fraction"
    ] == pytest.approx(
        1.0 / 3.0
    )

    assert metrics[
        "visible_low_fraction"
    ] == pytest.approx(
        2.0 / 3.0
    )

    assert metrics[
        "visible_population_bias"
    ] == pytest.approx(
        1.0 / 3.0
    )


def test_observer_social_diagnostics_marks_tie_and_mismatch_denominator():
    sources = {
        "agent_0": ["agent_1", "agent_2"],
        "agent_1": ["agent_0", "agent_2"],
        "agent_2": ["agent_0", "agent_1"],
    }

    actions = {
        "agent_0": LOW_EXTRACT,
        "agent_1": HIGH_EXTRACT,
        "agent_2": HIGH_EXTRACT,
    }

    diagnostics = observer_social_diagnostics(
        sources,
        actions,
    )

    assert diagnostics[
        "agent_0"
    ][
        "majority_tied"
    ] is False

    assert diagnostics[
        "agent_0"
    ][
        "majority_mismatch"
    ] == pytest.approx(
        0.0
    )

    assert diagnostics[
        "agent_1"
    ][
        "majority_tied"
    ] is True

    assert np.isnan(
        float(
            diagnostics[
                "agent_1"
            ][
                "majority_mismatch"
            ]
        )
    )


def test_visit_weighted_policy_hamming_ignores_unvisited_states():
    class FakeLearner:
        def __init__(
            self,
            policy,
        ):
            self._policy = tuple(
                policy
            )
            self.q = np.zeros(
                (
                    len(
                        self._policy
                    ),
                    2,
                ),
                dtype=float,
            )

        def greedy_policy(
            self,
        ):
            return self._policy

    learners = {
        "agent_0": FakeLearner(
            (
                LOW_EXTRACT,
                LOW_EXTRACT,
                LOW_EXTRACT,
            )
        ),
        "agent_1": FakeLearner(
            (
                LOW_EXTRACT,
                HIGH_EXTRACT,
                HIGH_EXTRACT,
            )
        ),
    }

    agents = {
        "agent_0": SimpleNamespace(
            position=np.asarray(
                [0, 0]
            )
        ),
        "agent_1": SimpleNamespace(
            position=np.asarray(
                [1, 0]
            )
        ),
    }

    env = SimpleNamespace(
        model=SimpleNamespace(
            by_name=agents
        )
    )

    summary, _ = experiment.policy_diagnostics(
        env,
        np.asarray(
            [["region", "region"]],
            dtype=object,
        ),
        learners,
        "uniform_high",
        population=2,
        replicate=0,
        social_mode="none",
        rewiring="none",
        sources=None,
        rewire_counts={
            "agent_0": 0,
            "agent_1": 0,
        },
        mean_prediction_errors={
            "agent_0": float("nan"),
            "agent_1": float("nan"),
        },
        state_visit_counts=np.asarray(
            [100, 0, 0],
            dtype=int,
        ),
    )

    assert summary[
        "policy_hamming_mean"
    ] == pytest.approx(
        2.0 / 3.0
    )

    assert summary[
        "policy_hamming_visit_weighted_mean"
    ] == pytest.approx(
        0.0
    )

    assert summary[
        "visited_state_fraction"
    ] == pytest.approx(
        1.0 / 3.0
    )

    assert summary[
        "training_visit_fraction_scarce"
    ] == pytest.approx(
        1.0
    )


def test_network_snapshot_rows_use_source_to_observer_direction():
    args = SimpleNamespace(
        social_mode="fixed",
        social_network="random_k",
        rewiring="none",
        rewire_theta=0.25,
    )

    sources = {
        "agent_0": ["agent_1"],
        "agent_1": ["agent_0"],
    }

    rows = experiment.network_snapshot_rows(
        scenario_name="uniform_high",
        population=2,
        replicate=0,
        args=args,
        checkpoint="initial",
        time=0,
        sources=sources,
    )

    edges = {
        (
            row["source"],
            row["observer"],
        )
        for row in rows
    }

    assert edges == {
        ("agent_0", "agent_1"),
        ("agent_1", "agent_0"),
    }


def test_agent_social_accumulator_excludes_ties_from_mismatch_denominator():
    sources = {
        "agent_0": ["agent_1", "agent_2"],
        "agent_1": ["agent_0", "agent_2"],
        "agent_2": ["agent_0", "agent_1"],
    }

    actions = {
        "agent_0": LOW_EXTRACT,
        "agent_1": HIGH_EXTRACT,
        "agent_2": HIGH_EXTRACT,
    }

    focal = observer_social_diagnostics(
        sources,
        actions,
    )

    accumulators = experiment.make_agent_social_accumulators(
        sources
    )

    experiment.update_agent_social_accumulators(
        accumulators,
        focal,
    )

    assert accumulators[
        "agent_0"
    ][
        "majority_valid_count"
    ] == 1

    assert accumulators[
        "agent_0"
    ][
        "majority_mismatch_count"
    ] == 0

    assert accumulators[
        "agent_1"
    ][
        "majority_tie_count"
    ] == 1

    assert accumulators[
        "agent_1"
    ][
        "majority_valid_count"
    ] == 0
