from types import SimpleNamespace

import numpy as np
import pytest

from cognitive_tools.scenarios import BALANCED_SCENARIOS
from cognitive_tools.visibility import PROFILES
from cognitive_tools.visibility_hypotheses import _association, _h1_checkpoints, _pooled_contrast


def test_h1_centers_within_treatment_before_association():
    # Large differences between treatment means must not determine H1's sign.
    gini = np.array([[0.10, 0.20, 0.30], [0.70, 0.80, 0.90]])
    error = np.array([[0.50, 0.40, 0.30], [0.90, 0.80, 0.70]])
    assert _association(gini, error) == pytest.approx(-1.0)


def test_h1_checkpoint_association_tracks_within_run_visibility_change():
    runs = []
    for profile in PROFILES:
        rows = []
        for scenario in BALANCED_SCENARIOS:
            for time, gini in ((1, 0.1), (2, 0.2), (3, 0.1)):
                rows.append(
                    dict(
                        scenario=scenario,
                        population=8,
                        replicate=0,
                        time=time,
                        visibility_gini=gini,
                        social_perception_error=0.5 - gini,
                    )
                )
        runs.append(
            SimpleNamespace(
                config=dict(network_dynamics="adaptive_bounded", visibility_profile=profile),
                tables={"training_timeseries": rows},
            )
        )
    pairs, summary = _h1_checkpoints(runs, bootstrap_reps=10, bootstrap_seed=1)
    assert len(pairs) == 45
    assert {row["measure"] for row in summary} == {"within_run", "linear_time_adjusted"}
    assert all(row["estimate"] == pytest.approx(-1.0) for row in summary)
    assert all(row["n_treatment_cells"] == 15 for row in summary)


def test_pooled_h2_averages_cells_within_replicate_before_interval():
    cells = {("equal", "fixed"), ("random", "fixed")}
    rows = []
    for replicate, values in ((0, (0.1, 0.3)), (1, (0.2, 0.4))):
        for (profile, dynamics), value in zip(sorted(cells), values):
            rows.append(
                dict(
                    comparison="balanced_segregated-balanced_dispersed",
                    metric="social_perception_error",
                    profile=profile,
                    dynamics=dynamics,
                    population=8,
                    replicate=replicate,
                    value=value,
                )
            )
    replicates, summary = _pooled_contrast(
        rows,
        hypothesis="H2",
        comparison="balanced_segregated-balanced_dispersed",
        expected_cells=cells,
        bootstrap_reps=100,
        bootstrap_seed=9,
    )
    assert [row["value"] for row in replicates] == pytest.approx([0.2, 0.3])
    assert summary["mean"] == pytest.approx(0.25)
    assert summary["n_replicates"] == 2
    assert summary["n_cells"] == 2
    assert summary["population"] == 8


def test_pooled_hypothesis_rejects_missing_cell():
    with pytest.raises(ValueError, match="Incomplete H2"):
        _pooled_contrast(
            [
                dict(
                    comparison="balanced_segregated-balanced_dispersed",
                    metric="social_perception_error",
                    profile="equal",
                    dynamics="fixed",
                    population=8,
                    replicate=0,
                    value=0.1,
                )
            ],
            hypothesis="H2",
            comparison="balanced_segregated-balanced_dispersed",
            expected_cells={("equal", "fixed"), ("random", "fixed")},
            bootstrap_reps=10,
            bootstrap_seed=1,
        )
