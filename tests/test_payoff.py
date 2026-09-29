from dataclasses import replace

import numpy as np
import pytest

from cognitive_tools.payoff import PayoffConfig, assignment, make_env, policy_actions, rollout


def small(**kwargs):
    return PayoffConfig(
        population=4,
        width=2,
        height=2,
        horizons=(3,),
        compositions=(0, 1, 3),
        focal_count=2,
        replicates=2,
        **kwargs,
    )


def test_paired_resets_and_coplayer_assignments_are_identical():
    config = small()
    a, oa, _ = make_env(config, "uniform_high", 0)
    b, ob, _ = make_env(config, "uniform_high", 0)
    assert a.model is not b.model
    np.testing.assert_array_equal(a.model.resource, b.model.resource)
    for name in oa:
        np.testing.assert_array_equal(oa[name]["self"], ob[name]["self"])
    c = assignment(4, 2, [3, 0, 1], 1, "C")
    d = assignment(4, 2, [3, 0, 1], 1, "D")
    assert c == ("D", "D", "C", "C")
    assert d == ("D", "D", "D", "C")
    a.step({name: 1 for name in oa})
    assert not np.array_equal(a.model.resource, b.model.resource)


def test_short_rollout_matches_hand_calculated_constant_harvest():
    config = small()
    rows, summaries = rollout(config, "uniform_high", 0, ("C",) * 4)
    for row in rows:
        assert row["return_sum"] == pytest.approx(0.006)
        assert row["return_discounted"] == pytest.approx(0.002 * (1 + 0.95 + 0.95**2))
        assert row["wealth_delta"] == pytest.approx(row["return_sum"])
    assert summaries[0]["mean_return_sum"] == pytest.approx(0.006)
    assert (rows, summaries) == rollout(config, "uniform_high", 0, ("C",) * 4)


def test_policy_uses_existing_observation_bins():
    observations = {
        f"agent_{i}": {"local": np.array([r, 0, 1])} for i, r in enumerate([0.1, 0.5, 0.9])
    }
    assert policy_actions(observations, ("C",) * 3, "010", "111") == {
        "agent_0": 0,
        "agent_1": 1,
        "agent_2": 0,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"gamma": float("nan")},
        {"gamma": 1.1},
        {"population": 1},
        {"focal_count": 5},
        {"horizons": (0,)},
        {"compositions": (1,)},
        {"policy_c": "012"},
        {"policy_d": "000"},
        {"replicates": 0},
    ],
)
def test_invalid_protocol_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(small(), **changes).validate()


def test_sweep_writes_complete_pairs_and_refuses_overwrite(tmp_path):
    import csv
    import json

    from cognitive_tools.payoff import run_experiment

    destination = tmp_path / "payoff"
    config = replace(small(), scenarios=("uniform_high",))
    run_experiment(config, destination, workers=1)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    pairs = list(csv.DictReader((destination / "paired_returns.csv").open()))
    assert len(pairs) == 2 * 2 * 3
    assert all(
        float(r["delta_sum"]) == pytest.approx(float(r["d_sum"]) - float(r["c_sum"])) for r in pairs
    )
    with pytest.raises(FileExistsError):
        run_experiment(config, destination, workers=1)


def test_multiple_horizons_and_gamma_zero_use_correct_timing():
    config = replace(small(), horizons=(1, 3, 5), gamma=0.0, late_window=2)
    rows, _ = rollout(config, "uniform_high", 0, ("C",) * 4)
    for row in rows:
        assert row["return_sum"] == pytest.approx(0.002 * row["horizon"])
        assert row["return_discounted"] == pytest.approx(0.002)
        assert row["late_harvest_rate"] == pytest.approx(0.002)


@pytest.mark.parametrize("reward_mode", ["harvest", "capped_harvest"])
def test_parallel_and_serial_runs_have_identical_outputs(tmp_path, reward_mode):
    from cognitive_tools.payoff import run_experiment

    config = replace(small(), scenarios=("uniform_high",))
    config = replace(config, reward_mode=reward_mode)
    a = run_experiment(config, tmp_path / "serial", workers=1)
    b = run_experiment(config, tmp_path / "parallel", workers=2)
    assert a["output_sha256"] == b["output_sha256"]


def test_1000_step_endpoints_reproduce_saved_campaign():
    import csv
    from pathlib import Path

    saved = Path(
        "results/q_learning_baseline/experiments/ecology_perception_parallel_v1_s1/data/evaluation_summary.csv"
    )
    if not saved.exists():
        pytest.skip("Saved campaign is not available in this checkout")
    with saved.open() as handle:
        expected = {
            (r["scenario"], r["strategy"]): float(r["final_mean_wealth"])
            for r in csv.DictReader(handle)
            if r["replicate"] == "0" and r["strategy"] in ("always_low", "always_high")
        }
    config = PayoffConfig()
    for scenario in config.scenarios:
        for branch, strategy in [("C", "always_low"), ("D", "always_high")]:
            _, summaries = rollout(config, scenario, 0, (branch,) * 64)
            assert summaries[0]["mean_return_sum"] == pytest.approx(
                expected[(scenario, strategy)], abs=1e-12
            )


def test_capped_rollout_separates_utility_and_harvest():
    config = small(reward_mode="capped_harvest")
    rows, summaries = rollout(config, "uniform_high", 0, ("D",) * 4)
    for row in rows:
        assert row["return_sum"] == pytest.approx(0.006)
        assert row["harvest_sum"] == pytest.approx(0.06)
        assert row["return_discounted"] == pytest.approx(0.002 * (1 + 0.95 + 0.95**2))
        for metric in ("sum", "discounted"):
            assert row[f"return_{metric}"] + row[f"uncredited_harvest_{metric}"] == pytest.approx(
                row[f"harvest_{metric}"]
            )
        assert row["wealth_delta"] == pytest.approx(row["harvest_sum"])
        assert row["late_harvest_rate"] == pytest.approx(0.02)
        assert row["late_utility_rate"] == pytest.approx(0.002)
    assert summaries[0]["mean_harvest_sum"] == pytest.approx(0.06)


def test_cap_is_applied_per_step_not_to_aggregate_harvest():
    config = replace(
        small(),
        population=2,
        width=1,
        height=1,
        horizons=(2,),
        compositions=(0, 1),
        initial_resource_fraction=0.02,
        reward_mode="capped_harvest",
    )
    rows, _ = rollout(config, "uniform_high", 0, ("D", "D"))
    for row in rows:
        assert row["return_sum"] == pytest.approx(0.002)
        assert min(row["harvest_sum"], 2 * config.metabolism) > row["return_sum"]
