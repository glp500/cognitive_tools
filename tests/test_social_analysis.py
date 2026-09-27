from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import social_experiment_analysis as analysis


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)


def base_config(
    *,
    treatment: str,
    rewiring: str,
    theta: float = 0.25,
    mu: float = 0.10,
) -> dict:
    return {
        "social_measurement_schema": "stage4_v1",
        "treatment": treatment,
        "rewiring": rewiring,
        "rewire_theta": theta,
        "rewire_mu": mu,
        "network_eval": "frozen",
        "scenarios": [
            "uniform_high",
        ],
        "populations": [
            8,
        ],
        "replicates": 2,
        "seed": 42,
        "width": 4,
        "height": 4,
        "coupling": 0.10,
        "low_harvest": 0.002,
        "high_harvest": 0.020,
        "metabolism": 0.002,
        "initial_energy": 1.0,
        "energy_capacity": 1.0,
        "initial_resource_fraction": 0.50,
        "alpha": 0.10,
        "gamma": 0.95,
        "epsilon": 0.20,
        "epsilon_min": 0.02,
        "epsilon_decay": 0.9995,
        "training_steps": 100,
        "evaluation_steps": 20,
        "record_every": 10,
        "record_network_every": 10,
        "social_k": 4,
        "rewire_every": 10,
        "rewire_threshold": 0.25,
        "forecast_alpha": 0.50,
        "run_metadata": {
            "git_commit_sha": "a" * 40,
            "git_worktree_dirty": False,
        },
        "matched_rewire_schedule_sha256": None,
    }


def evaluation_rows(
    resource_values: tuple[float, float],
) -> list[dict]:
    rows = []

    for replicate, value in enumerate(
        resource_values
    ):
        common = {
            "scenario": "uniform_high",
            "population": 8,
            "replicate": replicate,
            "social_mode": "fixed",
            "social_network": "random_k",
            "rewiring": "prediction_error",
            "strategy": "q_learning",
            "network_adaptive": False,
            "steps_evaluated": 20,
            "eval_mean_mean_resource_fraction": value,
            "final_mean_resource_fraction": value,
            "eval_mean_mean_reserve_welfare": 0.7,
            "eval_mean_mean_need_satisfaction": 0.8,
            "eval_mean_deprivation_rate": 0.1,
            "final_wealth_gini": 0.2,
            "eval_mean_social_perception_error": 0.15,
            "eval_mean_visibility_gini": 0.25,
            "eval_mean_majority_mismatch_rate": 0.20,
        }

        rows.append(
            {
                **common,
                "evaluation_mode": "fresh_reset",
                "network_start": "terminal",
            }
        )
        rows.append(
            {
                **common,
                "evaluation_mode": "fresh_reset_network",
                "network_start": "initial",
                "eval_mean_mean_resource_fraction": value - 0.05,
                "final_mean_resource_fraction": value - 0.05,
            }
        )

    return rows


def training_rows() -> list[dict]:
    rows = []

    for replicate in range(2):
        for time in (
            10,
            100,
        ):
            rows.append(
                {
                    "scenario": "uniform_high",
                    "population": 8,
                    "replicate": replicate,
                    "time": time,
                    "mean_resource_fraction": 0.5 + replicate * 0.02,
                    "wealth_gini": 0.2 + replicate * 0.01,
                    "visibility_gini": 0.3 + replicate * 0.02,
                    "social_perception_error": 0.15,
                    "degree_action_correlation": 0.1,
                }
            )

    return rows


def network_rows() -> list[dict]:
    rows = []

    for replicate in range(2):
        for time in (
            0,
            10,
            100,
        ):
            rows.append(
                {
                    "scenario": "uniform_high",
                    "population": 8,
                    "replicate": replicate,
                    "time": time,
                    "visibility_gini": 0.2 + 0.001 * time,
                    "mean_perception_error": (
                        "nan"
                        if time == 0
                        else 0.15
                    ),
                    "degree_action_correlation": (
                        "nan"
                        if time == 0
                        else 0.1
                    ),
                    "majority_mismatch_rate": (
                        "nan"
                        if time == 0
                        else 0.2
                    ),
                    "majority_tie_rate": (
                        "nan"
                        if time == 0
                        else 0.1
                    ),
                    "reciprocity": 0.3,
                    "degree_assortativity": 0.05,
                    "edge_turnover": 0.0 if time == 0 else 0.1,
                    "cumulative_rewires": 0 if time == 0 else time // 10,
                }
            )

    return rows


def policy_rows() -> list[dict]:
    rows = []

    for replicate in range(2):
        row = {
            "scenario": "uniform_high",
            "population": 8,
            "replicate": replicate,
        }

        for ecological in analysis.ECOLOGICAL_STATES:
            for social in analysis.SOCIAL_STATES:
                row[
                    f"policy_low_{ecological}_{social}"
                ] = 0.5
                row[
                    f"training_visit_fraction_{ecological}_{social}"
                ] = 1.0 / 9.0

        rows.append(row)

    return rows


def agent_rows() -> list[dict]:
    rows = []

    for replicate in range(2):
        for agent in range(2):
            rows.append(
                {
                    "scenario": "uniform_high",
                    "population": 8,
                    "replicate": replicate,
                    "agent": f"agent_{agent}",
                    "mean_population_low_fraction_excluding": 0.5,
                    "mean_observed_low_fraction": 0.55,
                }
            )

    return rows


def make_run(
    root: Path,
    name: str,
    *,
    treatment: str,
    rewiring: str,
    theta: float = 0.25,
    mu: float = 0.10,
    resources: tuple[float, float] = (
        0.50,
        0.60,
    ),
    schedule_payload: str | None = None,
    matched_hash: str | None = None,
) -> Path:
    run_dir = root / name
    data_dir = run_dir / "data"
    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = base_config(
        treatment=treatment,
        rewiring=rewiring,
        theta=theta,
        mu=mu,
    )
    config[
        "matched_rewire_schedule_sha256"
    ] = matched_hash

    with (
        run_dir
        / "config.json"
    ).open(
        "w"
    ) as file:
        json.dump(
            config,
            file,
        )

    write_csv(
        data_dir
        / "evaluation_summary.csv",
        evaluation_rows(
            resources
        ),
    )
    write_csv(
        data_dir
        / "training_timeseries.csv",
        training_rows(),
    )
    write_csv(
        data_dir
        / "policy_summary.csv",
        policy_rows(),
    )
    write_csv(
        data_dir
        / "agent_social_summary.csv",
        agent_rows(),
    )
    write_csv(
        data_dir
        / "network_timeseries.csv",
        network_rows(),
    )

    if schedule_payload is not None:
        (
            data_dir
            / "rewiring_schedule.csv"
        ).write_text(
            schedule_payload
        )

    return run_dir


def test_bootstrap_mean_ci_is_deterministic():
    first = analysis.bootstrap_mean_ci(
        [
            1.0,
            2.0,
            3.0,
        ],
        bootstrap_reps=100,
        seed=123,
    )
    second = analysis.bootstrap_mean_ci(
        [
            1.0,
            2.0,
            3.0,
        ],
        bootstrap_reps=100,
        seed=123,
    )

    assert first == second
    assert first[0] == pytest.approx(
        2.0
    )


def test_resource_regime_boundaries():
    assert analysis.resource_regime(
        0.2,
        low_threshold=1.0 / 3.0,
        high_threshold=2.0 / 3.0,
    ) == "low"

    assert analysis.resource_regime(
        0.5,
        low_threshold=1.0 / 3.0,
        high_threshold=2.0 / 3.0,
    ) == "middle"

    assert analysis.resource_regime(
        2.0 / 3.0,
        low_threshold=1.0 / 3.0,
        high_threshold=2.0 / 3.0,
    ) == "high"


def test_load_run_rejects_pre_stage4_schema(
    tmp_path,
):
    run_dir = make_run(
        tmp_path,
        "bad",
        treatment="R2",
        rewiring="prediction_error",
    )

    path = run_dir / "config.json"
    config = json.loads(
        path.read_text()
    )
    config[
        "social_measurement_schema"
    ] = "stage3"
    path.write_text(
        json.dumps(config)
    )

    with pytest.raises(
        ValueError,
        match="Stage 4",
    ):
        analysis.load_run(
            run_dir
        )


def test_r0_pairing_prefers_schedule_hash(
    tmp_path,
):
    adaptive_dir = make_run(
        tmp_path,
        "adaptive",
        treatment="R2",
        rewiring="prediction_error",
        schedule_payload="schedule-a\n",
    )
    adaptive = analysis.load_run(
        adaptive_dir
    )

    r0_dir = make_run(
        tmp_path,
        "r0",
        treatment="R0",
        rewiring="random_matched",
        matched_hash=(
            adaptive.schedule_sha256
        ),
    )
    r0 = analysis.load_run(
        r0_dir
    )

    runs = [
        adaptive,
        r0,
    ]
    analysis.assign_run_labels(
        runs
    )
    warnings = analysis.resolve_r0_pairs(
        runs
    )

    assert warnings == []
    assert (
        r0.paired_adaptive_run_id
        == adaptive.run_id
    )


def test_paired_effect_is_adaptive_minus_r0(
    tmp_path,
):
    adaptive_dir = make_run(
        tmp_path,
        "adaptive",
        treatment="R2",
        rewiring="prediction_error",
        resources=(
            0.7,
            0.8,
        ),
        schedule_payload="schedule-b\n",
    )
    adaptive = analysis.load_run(
        adaptive_dir
    )

    r0_dir = make_run(
        tmp_path,
        "r0",
        treatment="R0",
        rewiring="random_matched",
        resources=(
            0.5,
            0.6,
        ),
        matched_hash=(
            adaptive.schedule_sha256
        ),
    )
    r0 = analysis.load_run(
        r0_dir
    )

    runs = [
        adaptive,
        r0,
    ]
    analysis.assign_run_labels(
        runs
    )
    analysis.resolve_r0_pairs(
        runs
    )

    rows, summary = analysis.build_paired_effects(
        runs,
        bootstrap_reps=100,
        bootstrap_seed=1,
    )

    resource_rows = [
        row
        for row in rows
        if row[
            "metric"
        ]
        == "resource_fraction"
    ]

    assert len(resource_rows) == 2
    assert all(
        row[
            "adaptive_minus_r0"
        ]
        == pytest.approx(
            0.2
        )
        for row in resource_rows
    )

    resource_summary = [
        row
        for row in summary
        if row[
            "metric"
        ]
        == "resource_fraction"
    ]

    assert len(resource_summary) == 1
    assert resource_summary[0][
        "mean_adaptive_minus_r0"
    ] == pytest.approx(
        0.2
    )


def test_fixed_network_memory_check_is_zero(
    tmp_path,
):
    run_dir = make_run(
        tmp_path,
        "s1",
        treatment="S1",
        rewiring="none",
        resources=(
            0.5,
            0.6,
        ),
    )

    # For a fixed graph the carried and reset-network rows must be identical.
    evaluation_path = (
        run_dir
        / "data"
        / "evaluation_summary.csv"
    )
    rows = list(
        csv.DictReader(
            evaluation_path.open()
        )
    )

    carried_by_rep = {
        row[
            "replicate"
        ]: row
        for row in rows
        if row[
            "evaluation_mode"
        ]
        == "fresh_reset"
    }

    for row in rows:
        if (
            row[
                "evaluation_mode"
            ]
            == "fresh_reset_network"
        ):
            source = carried_by_rep[
                row[
                    "replicate"
                ]
            ]

            for key in (
                "eval_mean_mean_resource_fraction",
                "final_mean_resource_fraction",
                "eval_mean_mean_reserve_welfare",
                "eval_mean_mean_need_satisfaction",
                "eval_mean_deprivation_rate",
                "final_wealth_gini",
                "eval_mean_social_perception_error",
                "eval_mean_visibility_gini",
                "eval_mean_majority_mismatch_rate",
            ):
                row[key] = source[key]

    write_csv(
        evaluation_path,
        rows,
    )

    run = analysis.load_run(
        run_dir
    )
    analysis.assign_run_labels(
        [
            run,
        ]
    )

    _, summary, warnings = (
        analysis.build_network_memory_effects(
            [
                run,
            ],
            bootstrap_reps=10,
            bootstrap_seed=1,
        )
    )

    assert warnings == []

    resource = [
        row
        for row in summary
        if row[
            "metric"
        ]
        == "resource_fraction"
    ]

    assert resource[0][
        "mean_carried_minus_reset"
    ] == pytest.approx(
        0.0
    )


def test_end_to_end_analysis_writes_core_outputs(
    tmp_path,
):
    adaptive_dir = make_run(
        tmp_path,
        "adaptive",
        treatment="R2",
        rewiring="prediction_error",
        resources=(
            0.7,
            0.8,
        ),
        schedule_payload="schedule-c\n",
    )
    adaptive = analysis.load_run(
        adaptive_dir
    )

    r0_dir = make_run(
        tmp_path,
        "r0",
        treatment="R0",
        rewiring="random_matched",
        resources=(
            0.5,
            0.6,
        ),
        matched_hash=(
            adaptive.schedule_sha256
        ),
    )

    output_dir = (
        tmp_path
        / "analysis"
    )

    args = SimpleNamespace(
        run=[
            str(adaptive_dir),
            str(r0_dir),
        ],
        analysis_name="test",
        output=str(output_dir),
        bootstrap_reps=10,
        bootstrap_seed=123,
        resource_low_threshold=(
            1.0 / 3.0
        ),
        resource_high_threshold=(
            2.0 / 3.0
        ),
    )

    result = analysis.run_analysis(
        args
    )

    assert result == output_dir.resolve()
    assert (
        output_dir
        / "analysis_manifest.json"
    ).is_file()
    assert (
        output_dir
        / "data"
        / "resource_distribution.csv"
    ).is_file()
    assert (
        output_dir
        / "data"
        / "paired_adaptive_minus_r0.csv"
    ).is_file()
    assert (
        output_dir
        / "figures"
        / "01_resource_outcome_distributions.png"
    ).is_file()
    assert (
        output_dir
        / "figures"
        / "04_visibility_gini_trajectories.png"
    ).is_file()



def test_validate_compatibility_rejects_mechanical_mismatch(
    tmp_path,
):
    first_dir = make_run(
        tmp_path,
        "first",
        treatment="S1",
        rewiring="none",
    )
    second_dir = make_run(
        tmp_path,
        "second",
        treatment="S2",
        rewiring="none",
    )

    config_path = second_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["training_steps"] = 101
    config_path.write_text(json.dumps(config))

    runs = [
        analysis.load_run(first_dir),
        analysis.load_run(second_dir),
    ]
    analysis.assign_run_labels(runs)

    with pytest.raises(
        ValueError,
        match="training_steps",
    ):
        analysis.validate_compatibility(runs)


def test_resource_summary_reports_regime_probabilities(
    tmp_path,
):
    run_dir = make_run(
        tmp_path,
        "adaptive",
        treatment="R2",
        rewiring="prediction_error",
        resources=(
            0.2,
            0.8,
        ),
    )
    run = analysis.load_run(run_dir)
    analysis.assign_run_labels([run])

    _, summary = analysis.build_resource_distribution(
        [run],
        low_threshold=1.0 / 3.0,
        high_threshold=2.0 / 3.0,
        bootstrap_reps=20,
        bootstrap_seed=1,
    )

    assert len(summary) == 1
    assert summary[0]["prob_low_resource"] == pytest.approx(0.5)
    assert summary[0]["prob_high_resource"] == pytest.approx(0.5)
    assert summary[0]["prob_middle_resource"] == pytest.approx(0.0)


def test_phase_summary_retains_theta_and_mu(
    tmp_path,
):
    run_dir = make_run(
        tmp_path,
        "adaptive",
        treatment="R2",
        rewiring="prediction_error",
        theta=0.25,
        mu=0.20,
    )
    run = analysis.load_run(run_dir)
    analysis.assign_run_labels([run])

    resource_rows, _ = analysis.build_resource_distribution(
        [run],
        low_threshold=1.0 / 3.0,
        high_threshold=2.0 / 3.0,
        bootstrap_reps=10,
        bootstrap_seed=1,
    )
    terminal = analysis.build_terminal_training_rows(
        [run],
        resource_rows,
    )
    phase = analysis.build_phase_summary(
        [run],
        resource_rows,
        terminal,
        bootstrap_reps=10,
        bootstrap_seed=1,
    )

    assert {
        row["metric"]
        for row in phase
    } == {
        "resource_fraction",
        "visibility_gini",
    }
    assert all(row["rewire_theta"] == pytest.approx(0.25) for row in phase)
    assert all(row["rewire_mu"] == pytest.approx(0.20) for row in phase)
