from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pytest

import baseline_validation_experiment as experiment
from Cognitive_tools.social import (
    init_barabasi_albert_attention,
    init_random_attention,
    network_edges,
    rewire_epoch,
    visibility_counts,
)


def make_args(**overrides):
    values = {
        "width": 4,
        "height": 4,
        "seed": 123,
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
        "social_mode": "fixed",
        "social_network": "random_k",
        "social_k": 2,
        "ba_m": 1,
        "rewiring": "none",
        "rewire_theta": 0.25,
        "rewire_mu": 0.10,
        "rewire_every": 2,
        "rewire_threshold": 0.25,
        "forecast_alpha": 0.50,
        "record_every": 2,
        "record_network_every": 2,
        "training_steps": 4,
        "evaluation_steps": 3,
        "matched_rewire_schedule": None,
        "scenarios": [
            "uniform_high",
        ],
        "populations": [
            6,
        ],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_ba_attention_is_symmetric_and_has_no_self_links():
    agents = [
        f"agent_{index}"
        for index in range(32)
    ]

    sources = init_barabasi_albert_attention(
        agents,
        m=2,
        rng=np.random.default_rng(42),
    )

    for observer, observer_sources in sources.items():
        assert observer not in observer_sources
        assert len(observer_sources) == len(set(observer_sources))

        for source in observer_sources:
            assert observer in sources[source]


def test_ba_attention_has_heterogeneous_visibility():
    agents = [
        f"agent_{index}"
        for index in range(64)
    ]

    sources = init_barabasi_albert_attention(
        agents,
        m=2,
        rng=np.random.default_rng(7),
    )

    visibility = visibility_counts(sources)

    assert len(set(visibility.values())) > 1
    assert max(visibility.values()) > min(visibility.values())


def test_ba_m_two_has_mean_degree_close_to_four():
    agents = [
        f"agent_{index}"
        for index in range(64)
    ]

    sources = init_barabasi_albert_attention(
        agents,
        m=2,
        rng=np.random.default_rng(9),
    )

    mean_degree = float(
        np.mean(
            [
                len(observer_sources)
                for observer_sources in sources.values()
            ]
        )
    )

    assert 3.5 < mean_degree < 4.0


def test_ba_attention_is_deterministic_for_same_seed():
    agents = [
        f"agent_{index}"
        for index in range(32)
    ]

    first = init_barabasi_albert_attention(
        agents,
        m=2,
        rng=np.random.default_rng(100),
    )

    second = init_barabasi_albert_attention(
        agents,
        m=2,
        rng=np.random.default_rng(100),
    )

    assert first == second


def test_random_matched_rewiring_realizes_exact_target_count():
    agents = [
        f"agent_{index}"
        for index in range(12)
    ]

    sources = init_random_attention(
        agents,
        k=4,
        rng=np.random.default_rng(1),
    )

    edge_count_before = len(network_edges(sources))

    events = rewire_epoch(
        sources,
        mode="random_matched",
        theta=0.25,
        mu=0.10,
        rng=np.random.default_rng(2),
        target_count=5,
    )

    assert len(events) == 5
    assert len(network_edges(sources)) == edge_count_before
    assert len(
        {
            event["observer"]
            for event in events
        }
    ) == 5


def test_random_matched_zero_target_changes_nothing():
    agents = [
        f"agent_{index}"
        for index in range(8)
    ]

    sources = init_random_attention(
        agents,
        k=2,
        rng=np.random.default_rng(3),
    )

    before = {
        observer: list(observer_sources)
        for observer, observer_sources in sources.items()
    }

    events = rewire_epoch(
        sources,
        mode="random_matched",
        theta=1.0,
        mu=0.10,
        rng=np.random.default_rng(4),
        target_count=0,
    )

    assert events == []
    assert sources == before


def test_random_matched_requires_target_count():
    agents = [
        f"agent_{index}"
        for index in range(8)
    ]

    sources = init_random_attention(
        agents,
        k=2,
        rng=np.random.default_rng(5),
    )

    with pytest.raises(ValueError):
        rewire_epoch(
            sources,
            mode="random_matched",
            theta=0.25,
            mu=0.10,
            rng=np.random.default_rng(6),
        )


def test_make_social_sources_supports_ba():
    args = make_args(
        social_network="ba",
        ba_m=2,
        rewiring="none",
    )

    env, _, _ = experiment.make_environment(
        "uniform_high",
        population=8,
        replicate=0,
        max_steps=2,
        args=args,
    )

    sources = experiment.make_social_sources(
        env,
        replicate=0,
        args=args,
    )

    assert sources is not None
    assert len(set(visibility_counts(sources).values())) > 1


def test_validate_configuration_rejects_ba_rewiring():
    args = make_args(
        social_network="ba",
        ba_m=2,
        rewiring="prediction_error",
    )

    with pytest.raises(
        ValueError,
        match="BA",
    ):
        experiment.validate_configuration(args)


def test_validate_configuration_requires_schedule_for_matched_random():
    args = make_args(
        rewiring="random_matched",
        matched_rewire_schedule=None,
    )

    with pytest.raises(
        ValueError,
        match="matched",
    ):
        experiment.validate_configuration(args)


def test_load_matched_schedule_and_lookup(tmp_path):
    schedule_path = tmp_path / "rewiring_schedule.csv"

    with schedule_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "scenario",
                "population",
                "replicate",
                "time",
                "rewiring",
                "rewire_theta",
                "rewire_every",
                "base_seed",
                "social_network",
                "social_k",
                "training_steps",
                "target_rewires",
                "successful_rewires",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scenario": "uniform_high",
                "population": 6,
                "replicate": 0,
                "time": 2,
                "rewiring": "prediction_error",
                "rewire_theta": 0.25,
                "rewire_every": 2,
                "base_seed": 123,
                "social_network": "random_k",
                "social_k": 2,
                "training_steps": 4,
                "target_rewires": "",
                "successful_rewires": 3,
            }
        )

    args = make_args(
        rewiring="random_matched",
        matched_rewire_schedule=str(schedule_path),
        rewire_theta=0.25,
    )

    target = experiment.matched_rewire_target(
        args,
        scenario_name="uniform_high",
        population=6,
        replicate=0,
        time=2,
    )

    assert target == 3


def test_matched_schedule_theta_must_match(tmp_path):
    schedule_path = tmp_path / "rewiring_schedule.csv"

    with schedule_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "scenario",
                "population",
                "replicate",
                "time",
                "rewiring",
                "rewire_theta",
                "rewire_every",
                "base_seed",
                "social_network",
                "social_k",
                "training_steps",
                "target_rewires",
                "successful_rewires",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scenario": "uniform_high",
                "population": 6,
                "replicate": 0,
                "time": 2,
                "rewiring": "prediction_error",
                "rewire_theta": 0.0,
                "rewire_every": 2,
                "base_seed": 123,
                "social_network": "random_k",
                "social_k": 2,
                "training_steps": 4,
                "target_rewires": "",
                "successful_rewires": 2,
            }
        )

    args = make_args(
        rewiring="random_matched",
        matched_rewire_schedule=str(schedule_path),
        rewire_theta=0.25,
    )

    with pytest.raises(
        ValueError,
        match="theta",
    ):
        experiment.matched_rewire_target(
            args,
            scenario_name="uniform_high",
            population=6,
            replicate=0,
            time=2,
        )


def test_matched_training_uses_schedule_event_count(tmp_path):
    schedule_path = tmp_path / "rewiring_schedule.csv"

    with schedule_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "scenario",
                "population",
                "replicate",
                "time",
                "rewiring",
                "rewire_theta",
                "rewire_every",
                "base_seed",
                "social_network",
                "social_k",
                "training_steps",
                "target_rewires",
                "successful_rewires",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scenario": "uniform_high",
                "population": 6,
                "replicate": 0,
                "time": 2,
                "rewiring": "prediction_error",
                "rewire_theta": 0.25,
                "rewire_every": 2,
                "base_seed": 123,
                "social_network": "random_k",
                "social_k": 2,
                "training_steps": 2,
                "target_rewires": "",
                "successful_rewires": 2,
            }
        )

    args = make_args(
        rewiring="random_matched",
        matched_rewire_schedule=str(schedule_path),
        training_steps=2,
        evaluation_steps=1,
        rewire_every=2,
        record_every=2,
        record_network_every=2,
    )

    output = experiment.train_q_learning(
        "uniform_high",
        population=6,
        replicate=0,
        args=args,
    )

    rewire_counts = output[8]
    schedule_rows = output[10]

    assert sum(rewire_counts.values()) == 2
    assert len(schedule_rows) == 1
    assert schedule_rows[0]["target_rewires"] == 2
    assert schedule_rows[0]["successful_rewires"] == 2
