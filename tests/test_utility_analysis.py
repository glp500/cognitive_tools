from types import SimpleNamespace

import pytest

from cognitive_tools.utility_analysis import METRICS, build_utility_summaries


def run(mode="capped_harvest"):
    rows = [
        dict(
            strategy="q_learning",
            evaluation_mode="fresh_reset",
            scenario="uniform_high",
            population=8,
            replicate=i,
            **{metric: value for metric in METRICS},
        )
        for i, value in enumerate((1.0, 3.0))
    ]
    return SimpleNamespace(
        config={"reward_mode": mode},
        tables={"evaluation_summary": rows},
        run_id="test",
        path="test",
        treatment="B0",
        label="test",
        theta=0,
        mu=0,
    )


def test_utility_summary_uses_independent_replicates():
    records, summaries = build_utility_summaries([run()], bootstrap_reps=100, bootstrap_seed=42)
    assert len(records) == 8
    assert len(summaries) == 4
    assert all(row["mean"] == 2 and row["n_replicates"] == 2 for row in summaries)


def test_missing_capped_accounts_rejected_but_legacy_skipped():
    data = run()
    for row in data.tables["evaluation_summary"]:
        for metric in METRICS:
            del row[metric]
    with pytest.raises(ValueError, match="Missing"):
        build_utility_summaries([data], bootstrap_reps=10, bootstrap_seed=42)
    data.config = {}
    assert build_utility_summaries([data], bootstrap_reps=10, bootstrap_seed=42) == ([], [])


def test_duplicate_or_nonfinite_utility_rejected():
    data = run()
    data.tables["evaluation_summary"][1]["replicate"] = 0
    with pytest.raises(ValueError, match="Duplicate"):
        build_utility_summaries([data], bootstrap_reps=10, bootstrap_seed=42)
    data = run()
    data.tables["evaluation_summary"][0][METRICS[0]] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        build_utility_summaries([data], bootstrap_reps=10, bootstrap_seed=42)


def test_partial_harvest_and_empty_capped_tables_rejected():
    data = run("harvest")
    for metric in METRICS:
        del data.tables["evaluation_summary"][0][metric]
    with pytest.raises(ValueError, match="Missing"):
        build_utility_summaries([data], bootstrap_reps=10, bootstrap_seed=42)
    data = run()
    data.tables["evaluation_summary"] = []
    with pytest.raises(ValueError, match="Missing primary"):
        build_utility_summaries([data], bootstrap_reps=10, bootstrap_seed=42)
    data = run()
    data.config.update(scenarios=["uniform_high"], populations=[8], replicates=3)
    with pytest.raises(ValueError, match="Incomplete primary"):
        build_utility_summaries([data], bootstrap_reps=10, bootstrap_seed=42)


@pytest.mark.parametrize("empty", [False, True])
def test_new_harvest_cannot_masquerade_as_legacy(empty):
    data = run("harvest")
    for row in data.tables["evaluation_summary"]:
        for metric in METRICS:
            del row[metric]
    if empty:
        data.tables["evaluation_summary"] = []
    with pytest.raises(ValueError, match="Missing"):
        build_utility_summaries([data], bootstrap_reps=10, bootstrap_seed=42)
