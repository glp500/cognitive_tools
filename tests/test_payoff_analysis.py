import csv

import numpy as np
import pytest

from cognitive_tools.payoff import PayoffConfig, run_experiment
from cognitive_tools.payoff_analysis import analyze, classify, simultaneous_intervals


def test_simultaneous_intervals_use_replicate_blocks():
    values = np.array([[1, 2], [2, 4], [3, 6], [4, 8]], dtype=float)
    mean, low, high = simultaneous_intervals(values, resamples=1000, seed=1)
    np.testing.assert_allclose(mean, [2.5, 5])
    assert low[1] == pytest.approx(2 * low[0])
    assert high[1] == pytest.approx(2 * high[0])
    with pytest.raises(ValueError):
        simultaneous_intervals(values[:1], resamples=1000, seed=1)


@pytest.mark.parametrize(
    "bounds,expected",
    [
        ([(1, 2), (1, 2), (1, 2), (-2, -1)], "supported"),
        ([(-2, -1), (1, 2), (1, 2), (1, 2)], "contradicted"),
        ([(1, 2), (1, 2), (-2, -1), (-2, -1)], "contradicted"),
        ([(1, 2), (1, 2), (-1, 1), (-2, -1)], "inconclusive"),
        ([(0, 0), (1, 2), (1, 2), (1, 2)], "contradicted"),
    ],
)
def test_verdicts_respect_and_or_logic(bounds, expected):
    assert classify(bounds) == expected


def test_end_to_end_analysis_rejects_tampered_or_incomplete_data(tmp_path):
    config = PayoffConfig(
        scenarios=("uniform_high",),
        population=3,
        width=2,
        height=2,
        replicates=3,
        focal_count=1,
        compositions=(0, 1, 2),
        horizons=(3,),
    )
    run = tmp_path / "run"
    run_experiment(config, run)
    analysis = tmp_path / "analysis"
    analyze(run, analysis, resamples=200, figures=False)
    rows = list(csv.DictReader((analysis / "contrasts.csv").open()))
    assert len(rows) == 8
    collective = next(r for r in rows if r["contrast"] == "collective" and r["return"] == "sum")
    assert float(collective["mean"]) == pytest.approx(0.006 - 0.060)
    with pytest.raises(FileExistsError):
        analyze(run, analysis, resamples=200, figures=False)
    with (run / "paired_returns.csv").open("a") as f:
        f.write("corruption\n")
    with pytest.raises(ValueError, match="hash"):
        analyze(run, tmp_path / "bad", resamples=200, figures=False)


def test_audit_rejects_missing_pairs_even_with_updated_hash(tmp_path):
    import json

    from cognitive_tools.payoff import sha256
    from cognitive_tools.payoff_analysis import load_run

    config = PayoffConfig(
        scenarios=("uniform_high",),
        population=3,
        width=2,
        height=2,
        replicates=2,
        focal_count=1,
        compositions=(0, 2),
        horizons=(2,),
    )
    run = tmp_path / "run"
    run_experiment(config, run)
    pairs = run / "paired_returns.csv"
    lines = pairs.read_text().splitlines()
    pairs.write_text("\n".join(lines[:-1]) + "\n")
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_sha256"]["paired_returns.csv"] = sha256(pairs)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Incomplete pair coverage"):
        load_run(run)


def test_replication_within_block_does_not_increase_precision():
    from cognitive_tools.payoff_analysis import summarize

    config = PayoffConfig(
        scenarios=("uniform_high",),
        population=2,
        compositions=(0, 1),
        replicates=3,
        focal_count=1,
        horizons=(1,),
    )
    pairs, episodes = [], []
    for rep in range(3):
        for k in (0, 1):
            pairs.append(
                dict(
                    scenario="uniform_high",
                    replicate=rep,
                    horizon=1,
                    k=k,
                    c_sum=1 + rep + k,
                    d_sum=2 + rep + k,
                    c_discounted=1 + rep + k,
                    d_discounted=2 + rep + k,
                )
            )
        for branch in ("C", "D"):
            episodes.append(
                dict(
                    scenario="uniform_high",
                    replicate=rep,
                    horizon=1,
                    branch=branch,
                    kind="endpoint",
                    mean_return_sum=(4 + rep if branch == "C" else 2 + rep),
                    mean_return_discounted=(4 + rep if branch == "C" else 2 + rep),
                    mean_resource_fraction=0.5,
                    final_resource_fraction=0.5,
                    mean_reserve_welfare=0.5,
                    late_harvest_rate=0.1,
                )
            )
    original = summarize(config, pairs, episodes, resamples=200, seed=1)
    duplicated = summarize(config, pairs * 10, episodes, resamples=200, seed=1)
    assert original == duplicated


def test_numerical_tie_is_not_positive_conflict():
    assert classify([(1, 2), (1, 2), (1e-16, 2e-16), (0, 0)]) == "contradicted"


def test_capped_audit_and_legacy_schema(tmp_path):
    import hashlib
    import json
    from dataclasses import replace

    from cognitive_tools.payoff import sha256, write_csv
    from cognitive_tools.payoff_analysis import load_run

    cfg = PayoffConfig(
        scenarios=("uniform_high",),
        population=2,
        width=2,
        height=2,
        replicates=2,
        focal_count=1,
        compositions=(0, 1),
        horizons=(2,),
        reward_mode="capped_harvest",
    )
    run = tmp_path / "capped"
    run_experiment(cfg, run)
    load_run(run)
    analyze(run, tmp_path / "capped_analysis", resamples=200, figures=False)
    normalized = list(
        csv.DictReader((tmp_path / "capped_analysis" / "normalized_contrasts.csv").open())
    )
    assert float(normalized[0]["maximum_utility"]) == pytest.approx(0.004)
    path = run / "agent_returns.csv"
    rows = list(csv.DictReader(path.open()))
    rows[0]["uncredited_harvest_sum"] = "100"
    write_csv(path, rows)
    m = json.loads((run / "manifest.json").read_text())
    m["output_sha256"]["agent_returns.csv"] = sha256(path)
    (run / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="accounting"):
        load_run(run)
    legacy = tmp_path / "legacy"
    run_experiment(replace(cfg, reward_mode="harvest"), legacy)
    path = legacy / "manifest.json"
    m = json.loads(path.read_text())
    m["schema"] = "population_payoff_v1"
    del m["reward_definition"]
    del m["config"]["reward_mode"]
    m["protocol_sha256"] = hashlib.sha256(
        json.dumps(m["config"], sort_keys=True).encode()
    ).hexdigest()
    path.write_text(json.dumps(m))
    assert load_run(legacy)[1].reward_mode == "harvest"


def test_v2_hash_covers_reward_definition_and_requires_explicit_mode(tmp_path):
    import hashlib
    import json

    from cognitive_tools.payoff_analysis import load_run

    cfg = PayoffConfig(
        scenarios=("uniform_high",),
        population=2,
        width=1,
        height=1,
        replicates=2,
        focal_count=1,
        compositions=(0, 1),
        horizons=(1,),
    )
    run_experiment(cfg, tmp_path / "run")
    path = tmp_path / "run" / "manifest.json"
    m = json.loads(path.read_text())

    def protocol_hash():
        payload = {key: m[key] for key in ("config", "reward_definition")}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    assert m["protocol_sha256"] == protocol_hash()
    del m["config"]["reward_mode"]
    m["protocol_sha256"] = protocol_hash()
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="explicit"):
        load_run(path.parent)
