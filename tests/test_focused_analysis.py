from types import SimpleNamespace

import pytest

from cognitive_tools.focused_analysis import build_contrasts, build_window_rows


def test_window_excludes_boundary_and_preserves_undefined_mismatch():
    rows = [
        dict(
            scenario="a",
            population=8,
            replicate=0,
            time=t,
            social_perception_error=value,
            majority_mismatch_rate="nan",
            majority_tie_rate=1,
            visibility_gini=0.2,
            low_extraction_rate=0.5,
        )
        for t, value in [(4000, 99), (4050, 0.2), (5000, 0.4)]
    ]
    run = SimpleNamespace(
        run_id="s1",
        label="S1",
        treatment="S1",
        config={"training_steps": 5000},
        tables={"training_timeseries": rows},
    )
    result = {r["metric"]: r for r in build_window_rows([run])}
    assert result["social_perception_error"]["value"] == pytest.approx(0.3)
    assert result["social_perception_error"]["n_checkpoints"] == 2
    assert result["majority_mismatch_rate"]["n_valid"] == 0
    assert result["majority_tie_rate"]["value"] == 1
    run.treatment = "B0"
    assert {r["metric"] for r in build_window_rows([run])} == {"low_extraction_rate"}


def test_contrasts_pair_replicates_not_checkpoint_or_agent_rows():
    runs = [
        SimpleNamespace(run_id="r", treatment="R1", theta=0, mu=0.1, paired_adaptive_run_id=None),
        SimpleNamespace(
            run_id="control", treatment="R0", theta=0, mu=0.1, paired_adaptive_run_id="r"
        ),
    ]
    rows = [
        dict(
            run_id=run,
            scenario="a",
            population=8,
            replicate=rep,
            metric="social_perception_error",
            value=value,
        )
        for run, values in [("r", [0.2, 0.8]), ("control", [0.1, 0.6])]
        for rep, value in enumerate(values)
    ]
    differences, summary = build_contrasts(rows, runs, bootstrap_reps=50, bootstrap_seed=1729)
    assert len(differences) == 2
    assert summary[0]["n"] == 2
    assert summary[0]["mean"] == pytest.approx(0.15)
    rows.pop()
    with pytest.raises(ValueError, match="replicate"):
        build_contrasts(rows, runs, bootstrap_reps=50, bootstrap_seed=1729)
