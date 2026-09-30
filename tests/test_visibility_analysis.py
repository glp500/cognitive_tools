from types import SimpleNamespace

import pytest

from cognitive_tools import experiment
from cognitive_tools.visibility_analysis import run_visibility_analysis


def _run(profile, dynamics, scenario="uniform_high"):
    from cognitive_tools.visibility import load_visibility_spec

    _, sha = load_visibility_spec()
    args = experiment.build_parser().parse_args(
        [
            "--study-protocol",
            "visibility_bounded_search_v1",
            "--reward-mode",
            "capped_harvest",
            "--social-mode",
            "fixed",
            "--social-k",
            "4",
            "--attention-k",
            "4",
            "--visibility-profile",
            profile,
            "--network-dynamics",
            dynamics,
            "--rewiring",
            "prediction_error" if dynamics == "adaptive_bounded" else "none",
            "--training-steps",
            "4",
            "--evaluation-steps",
            "2",
            "--record-every",
            "2",
            "--record-network-every",
            "2",
            "--scenarios",
            scenario,
            "--populations",
            "8",
            "--replicates",
            "1",
        ]
    )
    args.visibility_profile_spec_sha256 = sha
    result = experiment.run_condition(scenario, 8, 0, args)
    return SimpleNamespace(
        config=dict(vars(args), social_measurement_schema="stage5_visibility_v1"),
        run_id=f"{profile}_{dynamics}_{scenario}",
        tables=result,
    )


def test_visibility_analysis_builds_paired_tables_and_figures(tmp_path):
    fixed = _run("equal", "fixed")
    adaptive = _run("equal", "adaptive_bounded")
    data, figures = tmp_path / "data", tmp_path / "figures"
    data.mkdir()
    figures.mkdir()
    tables = run_visibility_analysis(
        [fixed, adaptive], data, figures, bootstrap_reps=20, bootstrap_seed=1
    )
    assert len(tables["initial_visibility_summary"]) == 4
    assert {row["metric"] for row in tables["adaptive_fixed_summary"]} == {
        "social_perception_error",
        "visibility_gini",
        "low_extraction_rate",
        "resource_fraction",
        "reserve_welfare",
        "final_wealth_gini",
    }
    assert len(list(figures.glob("0*.png"))) == 5
    assert (data / "adaptive_fixed_replicates.csv").exists()


def test_visibility_analysis_rejects_unpaired_initial_graph(tmp_path):
    fixed = _run("random", "fixed")
    adaptive = _run("random", "adaptive_bounded")
    adaptive.tables["initial_visibility"][0]["initial_network_sha256"] = "corrupt"
    with pytest.raises(ValueError, match="Inconsistent"):
        run_visibility_analysis(
            [fixed, adaptive], tmp_path, tmp_path, bootstrap_reps=2, bootstrap_seed=1
        )
