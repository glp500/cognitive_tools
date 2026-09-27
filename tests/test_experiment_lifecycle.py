from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from Cognitive_tools.model import (
    HIGH_EXTRACT,
    LOW_EXTRACT,
)
from Cognitive_tools.social import (
    copy_sources,
)

import baseline_validation_experiment as experiment


def make_args(
    **overrides,
):
    """Small deterministic configuration for lifecycle tests."""

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
        "network_eval": "frozen",
        "scenarios": [
            "uniform_high",
        ],
        "populations": [
            6,
        ],
    }

    values.update(
        overrides
    )

    return SimpleNamespace(
        **values
    )


def test_run_metadata_has_reproducibility_fields():
    metadata = experiment.build_run_metadata()

    assert metadata["schema_version"] == 1
    assert metadata["created_at_utc"]
    assert metadata["python_version"]
    assert metadata["platform"]
    assert metadata["command"]
    assert isinstance(metadata["package_versions"], dict)

    expected_packages = {
        "numpy",
        "mesa",
        "pettingzoo",
        "gymnasium",
        "matplotlib",
        "networkx",
        "pytest",
    }

    assert set(metadata["package_versions"]) == expected_packages

    commit = metadata["git_commit_sha"]

    if commit is not None:
        assert len(commit) == 40

    dirty = metadata["git_worktree_dirty"]
    assert dirty is None or isinstance(dirty, bool)


def test_initial_random_k_graph_is_matched_across_scenarios():
    args = make_args(
        social_network="random_k",
        social_k=2,
    )

    uniform_env, _, _ = experiment.make_environment(
        "uniform_high",
        population=6,
        replicate=2,
        max_steps=2,
        args=args,
    )

    patchy_env, _, _ = experiment.make_environment(
        "patchy_high",
        population=6,
        replicate=2,
        max_steps=2,
        args=args,
    )

    uniform_sources = experiment.make_social_sources(
        uniform_env,
        replicate=2,
        args=args,
    )

    patchy_sources = experiment.make_social_sources(
        patchy_env,
        replicate=2,
        args=args,
    )

    assert uniform_sources == patchy_sources


def test_initial_random_k_graph_is_independent_of_rewiring_mode():
    fixed_args = make_args(
        rewiring="none",
    )

    adaptive_args = make_args(
        rewiring="prediction_error",
    )

    env, _, _ = experiment.make_environment(
        "uniform_high",
        population=6,
        replicate=1,
        max_steps=2,
        args=fixed_args,
    )

    fixed_sources = experiment.make_social_sources(
        env,
        replicate=1,
        args=fixed_args,
    )

    adaptive_sources = experiment.make_social_sources(
        env,
        replicate=1,
        args=adaptive_args,
    )

    assert fixed_sources == adaptive_sources


def test_frozen_evaluation_does_not_mutate_social_network():
    args = make_args(
        social_mode="fixed",
        social_network="random_k",
        social_k=2,
        rewiring="prediction_error",
        evaluation_steps=4,
    )

    env, observations, _ = experiment.make_environment(
        "uniform_high",
        population=6,
        replicate=0,
        max_steps=args.evaluation_steps,
        args=args,
    )

    sources = experiment.make_social_sources(
        env,
        replicate=0,
        args=args,
    )

    assert sources is not None

    before = copy_sources(sources)

    summary, _ = experiment.evaluate_policy(
        env,
        observations,
        scenario_name="uniform_high",
        population=6,
        replicate=0,
        strategy="always_low",
        evaluation_mode="test_frozen",
        learners=None,
        sources=sources,
        previous_actions=None,
        args=args,
        network_start="terminal",
        adaptive_network=False,
    )

    assert sources == before
    assert summary["network_adaptive"] is False
    assert summary["evaluation_total_rewires"] == 0


def test_adaptive_evaluation_rewires_copy_not_input_graph():
    args = make_args(
        social_mode="fixed",
        social_network="random_k",
        social_k=2,
        rewiring="prediction_error",
        rewire_theta=1.0,
        rewire_mu=1.0,
        rewire_every=1,
        rewire_threshold=0.0,
        evaluation_steps=2,
        record_every=1,
        network_eval="adaptive",
    )

    env, observations, _ = experiment.make_environment(
        "uniform_high",
        population=6,
        replicate=0,
        max_steps=args.evaluation_steps,
        args=args,
    )

    sources = experiment.make_social_sources(
        env,
        replicate=0,
        args=args,
    )

    assert sources is not None

    before = copy_sources(sources)
    forecasts = {
        name: 0.5
        for name in sources
    }

    summary, timeseries = experiment.evaluate_policy(
        env,
        observations,
        scenario_name="uniform_high",
        population=6,
        replicate=0,
        strategy="always_low",
        evaluation_mode="fresh_adaptive_network",
        learners=None,
        sources=sources,
        previous_actions=None,
        args=args,
        network_start="terminal",
        adaptive_network=True,
        forecasts=forecasts,
    )

    assert sources == before
    assert forecasts == {
        name: 0.5
        for name in sources
    }
    assert summary["network_adaptive"] is True
    assert summary["evaluation_total_rewires"] > 0
    assert timeseries[-1]["evaluation_rewires_cumulative"] > 0


def test_run_condition_decomposes_carried_and_reset_networks(
    monkeypatch,
):
    terminal_sources = {
        "agent_0": [
            "agent_2",
        ],
        "agent_1": [
            "agent_0",
        ],
        "agent_2": [
            "agent_1",
        ],
    }

    initial_sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_2",
        ],
        "agent_2": [
            "agent_0",
        ],
    }

    final_forecasts = {
        "agent_0": 0.2,
        "agent_1": 0.5,
        "agent_2": 0.8,
    }

    def fake_train(
        scenario_name,
        population,
        replicate,
        args,
    ):
        return (
            object(),
            {},
            None,
            {},
            [],
            [],
            terminal_sources,
            None,
            {
                "agent_0": 0,
                "agent_1": 0,
                "agent_2": 0,
            },
            {
                "agent_0": 0.0,
                "agent_1": 0.0,
                "agent_2": 0.0,
            },
            [],
            initial_sources,
            final_forecasts,
        )

    def fake_policy_diagnostics(
        *args,
        **kwargs,
    ):
        return (
            {
                "scenario": "uniform_high",
            },
            [],
        )

    def fake_make_environment(
        *args,
        **kwargs,
    ):
        return (
            object(),
            {},
            None,
        )

    calls = []

    def fake_evaluate_policy(
        *args,
        **kwargs,
    ):
        calls.append(
            {
                "strategy": kwargs["strategy"],
                "evaluation_mode": kwargs["evaluation_mode"],
                "sources": kwargs["sources"],
                "network_start": kwargs["network_start"],
                "adaptive_network": kwargs["adaptive_network"],
                "forecasts": kwargs.get("forecasts"),
            }
        )

        return (
            {
                "strategy": kwargs["strategy"],
                "evaluation_mode": kwargs["evaluation_mode"],
            },
            [],
        )

    monkeypatch.setattr(
        experiment,
        "train_q_learning",
        fake_train,
    )
    monkeypatch.setattr(
        experiment,
        "policy_diagnostics",
        fake_policy_diagnostics,
    )
    monkeypatch.setattr(
        experiment,
        "make_environment",
        fake_make_environment,
    )
    monkeypatch.setattr(
        experiment,
        "evaluate_policy",
        fake_evaluate_policy,
    )

    output = experiment.run_condition(
        "uniform_high",
        population=3,
        replicate=0,
        args=make_args(
            social_mode="fixed",
            social_network="random_k",
            social_k=1,
            network_eval="frozen",
            populations=[3],
        ),
    )

    assert len(calls) == 6

    assert [
        call["evaluation_mode"]
        for call in calls
    ] == [
        "continuation",
        "fresh_reset",
        "fresh_reset_network",
        "fresh_reset",
        "fresh_reset",
        "fresh_reset",
    ]

    assert calls[0]["sources"] is terminal_sources
    assert calls[1]["sources"] is terminal_sources
    assert calls[2]["sources"] is initial_sources

    for call in calls[3:]:
        assert call["sources"] is terminal_sources

    assert calls[0]["network_start"] == "terminal"
    assert calls[1]["network_start"] == "terminal"
    assert calls[2]["network_start"] == "initial"
    assert all(
        call["adaptive_network"] is False
        for call in calls
    )

    assert output["rewiring_schedule"] == []


def test_run_condition_adds_adaptive_robustness_evaluation(
    monkeypatch,
):
    terminal_sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_2",
        ],
        "agent_2": [
            "agent_0",
        ],
    }
    initial_sources = copy_sources(terminal_sources)
    final_forecasts = {
        "agent_0": 0.3,
        "agent_1": 0.4,
        "agent_2": 0.5,
    }

    def fake_train(
        *args,
        **kwargs,
    ):
        return (
            object(),
            {},
            None,
            {},
            [],
            [],
            terminal_sources,
            None,
            {
                name: 0
                for name in terminal_sources
            },
            {
                name: 0.0
                for name in terminal_sources
            },
            [],
            initial_sources,
            final_forecasts,
        )

    monkeypatch.setattr(
        experiment,
        "train_q_learning",
        fake_train,
    )
    monkeypatch.setattr(
        experiment,
        "policy_diagnostics",
        lambda *args, **kwargs: ({}, []),
    )
    monkeypatch.setattr(
        experiment,
        "make_environment",
        lambda *args, **kwargs: (object(), {}, None),
    )

    calls = []

    def fake_evaluate_policy(
        *args,
        **kwargs,
    ):
        calls.append(kwargs)
        return ({}, [])

    monkeypatch.setattr(
        experiment,
        "evaluate_policy",
        fake_evaluate_policy,
    )

    experiment.run_condition(
        "uniform_high",
        population=3,
        replicate=0,
        args=make_args(
            rewiring="prediction_error",
            network_eval="adaptive",
            social_k=1,
            populations=[3],
        ),
    )

    adaptive_calls = [
        call
        for call in calls
        if call["evaluation_mode"] == "fresh_adaptive_network"
    ]

    assert len(adaptive_calls) == 1
    adaptive_call = adaptive_calls[0]
    assert adaptive_call["sources"] is terminal_sources
    assert adaptive_call["network_start"] == "terminal"
    assert adaptive_call["adaptive_network"] is True
    assert adaptive_call["forecasts"] is final_forecasts


def test_q_update_uses_post_rewire_social_state(
    monkeypatch,
):
    """
    A rewiring event between a_t and the Q update must affect s_(t+1),
    while Stage 3 preserves an independent snapshot of the initial graph.
    """

    args = make_args(
        social_mode="fixed",
        social_network="random_k",
        social_k=1,
        rewiring="prediction_error",
        rewire_every=1,
        rewire_mu=1.0,
        rewire_threshold=0.0,
        training_steps=1,
        evaluation_steps=1,
        record_every=10,
        record_network_every=10,
        populations=[3],
    )

    initial_sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_2",
        ],
        "agent_2": [
            "agent_0",
        ],
    }

    class FakeLearner:
        def __init__(self, action: int):
            self.action = action
            self.epsilon = 0.20
            self.updates = []

        def choose_action(
            self,
            state,
            *,
            explore=True,
        ):
            return self.action

        def update(
            self,
            state,
            action,
            reward,
            next_state,
            *,
            done=False,
        ):
            self.updates.append(
                {
                    "state": int(state),
                    "action": int(action),
                    "next_state": int(next_state),
                    "done": bool(done),
                }
            )

        def decay_exploration(self):
            return None

    learners = {
        "agent_0": FakeLearner(HIGH_EXTRACT),
        "agent_1": FakeLearner(LOW_EXTRACT),
        "agent_2": FakeLearner(HIGH_EXTRACT),
    }

    def fake_make_social_sources(
        env,
        replicate,
        args,
    ):
        return copy_sources(initial_sources)

    def fake_make_learners(
        env,
        replicate,
        args,
    ):
        return learners

    def fake_rewire_epoch(
        sources,
        **kwargs,
    ):
        sources["agent_0"] = [
            "agent_2",
        ]

        return [
            {
                "observer": "agent_0",
                "dropped": "agent_1",
                "added": "agent_2",
                "requested_scope": "global",
                "used_scope": "global",
                "fallback": False,
                "trigger": "prediction_error",
                "prediction_error": 0.5,
            }
        ]

    monkeypatch.setattr(
        experiment,
        "make_social_sources",
        fake_make_social_sources,
    )
    monkeypatch.setattr(
        experiment,
        "make_learners",
        fake_make_learners,
    )
    monkeypatch.setattr(
        experiment,
        "rewire_epoch",
        fake_rewire_epoch,
    )

    output = experiment.train_q_learning(
        "uniform_high",
        population=3,
        replicate=0,
        args=args,
    )

    returned_learners = output[3]
    terminal_sources = output[6]
    rewire_counts = output[8]
    returned_initial_sources = output[11]
    final_forecasts = output[12]

    update = returned_learners[
        "agent_0"
    ].updates[0]

    # First decision: neutral/mixed social state.
    assert update["state"] % 3 == 1

    # Post-rewire target: agent_0 now sees agent_2's HIGH action.
    assert update["next_state"] % 3 == 0
    assert rewire_counts["agent_0"] == 1

    # The training endpoint changed, but the Stage 3 initial snapshot did not.
    assert terminal_sources["agent_0"] == [
        "agent_2",
    ]
    assert returned_initial_sources == initial_sources
    assert returned_initial_sources is not terminal_sources

    assert final_forecasts is not None
    assert set(final_forecasts) == set(initial_sources)


def test_adaptive_network_eval_rejects_random_matched():
    args = make_args(
        rewiring="random_matched",
        matched_rewire_schedule="placeholder.csv",
        network_eval="adaptive",
    )

    # Avoid schedule file loading: the adaptive-evaluation incompatibility
    # is the behavior under test, so substitute a harmless existing path.
    args.matched_rewire_schedule = __file__

    with pytest.raises(
        ValueError,
        match="random_matched",
    ):
        experiment.validate_configuration(args)


def test_b0_condition_is_deterministic_for_same_seed():
    args = make_args(
        social_mode="none",
        social_network="random_k",
        rewiring="none",
        network_eval="frozen",
        training_steps=4,
        evaluation_steps=3,
        record_every=2,
        populations=[3],
    )

    first = experiment.run_condition(
        "uniform_high",
        population=3,
        replicate=0,
        args=args,
    )

    second = experiment.run_condition(
        "uniform_high",
        population=3,
        replicate=0,
        args=args,
    )

    first_serialized = json.dumps(
        first,
        sort_keys=True,
        allow_nan=True,
    )

    second_serialized = json.dumps(
        second,
        sort_keys=True,
        allow_nan=True,
    )

    assert first_serialized == second_serialized
