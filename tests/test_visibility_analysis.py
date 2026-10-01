from types import SimpleNamespace

import pytest

from cognitive_tools import experiment
from cognitive_tools.scenarios import BALANCED_SCENARIOS
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


def test_balanced_analysis_uses_matched_ecologies_and_weighted_stock(tmp_path):
    runs = []
    for dynamics in ("fixed", "adaptive_bounded"):
        template = _run("equal", dynamics, "balanced_uniform")
        template.config["environment_design"] = "balanced_capacity_v1"
        template.config["scenarios"] = list(BALANCED_SCENARIOS)
        args = experiment.build_parser().parse_args(
            [
                "--study-protocol",
                "visibility_bounded_search_v1",
                "--environment-design",
                "balanced_capacity_v1",
                "--reward-mode",
                "capped_harvest",
                "--social-mode",
                "fixed",
                "--social-k",
                "4",
                "--attention-k",
                "4",
                "--visibility-profile",
                "equal",
                "--network-dynamics",
                dynamics,
                "--rewiring",
                "prediction_error" if dynamics == "adaptive_bounded" else "none",
                "--scenarios",
                *BALANCED_SCENARIOS,
                "--populations",
                "8",
                "--replicates",
                "1",
                "--training-steps",
                "4",
                "--evaluation-steps",
                "2",
                "--record-every",
                "2",
                "--record-network-every",
                "2",
            ]
        )
        args.visibility_profile_spec_sha256 = template.config["visibility_profile_spec_sha256"]
        merged = {
            key: list(value) for key, value in template.tables.items() if isinstance(value, list)
        }
        for scenario in BALANCED_SCENARIOS[1:]:
            result = experiment.run_condition(scenario, 8, 0, args)
            for key, value in result.items():
                if isinstance(value, list):
                    merged.setdefault(key, []).extend(value)
        template.tables = merged
        runs.append(template)
    tables = run_visibility_analysis(
        runs, tmp_path / "data", tmp_path / "figures", bootstrap_reps=20, bootstrap_seed=1
    )
    assert {row["comparison"] for row in tables["ecology_contrast_summary"]} == {
        "balanced_dispersed-balanced_uniform",
        "balanced_segregated-balanced_dispersed",
        "balanced_segregated-balanced_uniform",
    }
    assert {
        "low_extraction_rate",
        "visibility_gini",
        "resource_fraction",
        "reserve_welfare",
        "final_wealth_gini",
    } <= {row["metric"] for row in tables["ecology_contrast_summary"]}
    assert {
        "low_extraction_rate",
        "visibility_gini",
        "resource_fraction",
        "reserve_welfare",
        "final_wealth_gini",
    } <= {row["metric"] for row in tables["adaptive_fixed_summary"]}
    assert "local_resource_fraction" in {row["metric"] for row in tables["outcome_summary"]}
    outcomes = {
        (row["scenario"], row["dynamics"], row["metric"]): float(row["value"])
        for row in tables["outcome_replicates"]
    }
    for scenario in BALANCED_SCENARIOS:
        for dynamics in ("fixed", "adaptive_bounded"):
            assert outcomes[(scenario, dynamics, "resource_fraction")] == pytest.approx(
                outcomes[(scenario, dynamics, "resource_stock")]
                / outcomes[(scenario, dynamics, "total_capacity")]
            )
    assert len(list((tmp_path / "figures").glob("0*.png"))) == 5


def test_visibility_analysis_rejects_unpaired_initial_graph(tmp_path):
    fixed = _run("random", "fixed")
    adaptive = _run("random", "adaptive_bounded")
    adaptive.tables["initial_visibility"][0]["initial_network_sha256"] = "corrupt"
    with pytest.raises(ValueError, match="Inconsistent"):
        run_visibility_analysis(
            [fixed, adaptive], tmp_path, tmp_path, bootstrap_reps=2, bootstrap_seed=1
        )


def test_visibility_contrasts_follow_frozen_comparison_directions(tmp_path):
    from cognitive_tools.visibility import PROFILES

    runs = [_run(profile, "fixed") for profile in PROFILES]
    tables = run_visibility_analysis(runs, tmp_path, tmp_path, bootstrap_reps=20, bootstrap_seed=1)
    contrasts = tables["visibility_profile_contrasts"]
    assert {row["comparison"] for row in contrasts} == {
        "random-equal",
        "normal_centered-random",
        "low_propensity_majority-random",
        "high_propensity_majority-random",
    }
    values = {
        row["profile"]: float(row["value"])
        for row in tables["primary_window_replicates"]
        if row["metric"] == "social_perception_error"
    }
    random_equal = next(row for row in contrasts if row["comparison"] == "random-equal")
    assert float(random_equal["value"]) == pytest.approx(values["random"] - values["equal"])


def test_stage5_analysis_rejects_incompatible_profile_specs(tmp_path):
    from cognitive_tools.analysis import RunData, validate_compatibility

    config = dict(
        study_protocol="visibility_bounded_search_v1",
        reward_mode="capped_harvest",
        metabolism=0.002,
        visibility_profile_spec_sha256="a",
    )

    def item(name, metadata):
        return RunData(
            path=tmp_path / name,
            config=metadata,
            run_id=name,
            treatment=name,
            theta=0.25,
            mu=0.1,
            config_sha256="",
            schedule_sha256=None,
            matched_schedule_sha256=None,
        )

    first = item("first", config)
    second = item("second", dict(config, visibility_profile_spec_sha256="b"))
    with pytest.raises(ValueError, match="spec hashes"):
        validate_compatibility([first, second])


def test_undefined_majority_mismatch_is_not_zero_in_summary():
    from cognitive_tools.visibility_analysis import _summary

    rows = [dict(profile="equal", metric="majority_mismatch_rate", value=float("nan"))]
    assert _summary(rows, ("profile", "metric"), reps=20, seed=1) == []
