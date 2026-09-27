from __future__ import annotations

import json
from types import SimpleNamespace

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

    experiment.evaluate_policy(
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
    )

    assert sources == before


def test_run_condition_carries_terminal_network_into_current_evaluations(
    monkeypatch,
):
    """
    Stage 2 still carries the terminal graph into all current evaluations.

    Stage 3 will deliberately add reset-network and adaptive-network modes.
    """

    terminal_sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_0",
        ],
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
            },
            {
                "agent_0": 0.0,
                "agent_1": 0.0,
            },
            [],
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
        population=2,
        replicate=0,
        args=make_args(
            social_mode="fixed",
            social_network="random_k",
            social_k=1,
        ),
    )

    assert len(calls) == 5
    assert [
        call["evaluation_mode"]
        for call in calls
    ] == [
        "continuation",
        "fresh_reset",
        "fresh_reset",
        "fresh_reset",
        "fresh_reset",
    ]

    assert all(
        call["sources"] is terminal_sources
        for call in calls
    )

    assert output["rewiring_schedule"] == []


def test_q_update_uses_post_rewire_social_state(
    monkeypatch,
):
    """
    A rewiring event between a_t and the Q update must affect s_(t+1).
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
    rewire_counts = output[8]

    update = returned_learners[
        "agent_0"
    ].updates[0]

    # First decision: neutral/mixed social state.
    assert update["state"] % 3 == 1

    # Post-rewire target: agent_0 now sees agent_2's HIGH action.
    assert update["next_state"] % 3 == 0
    assert rewire_counts["agent_0"] == 1


def test_b0_condition_is_deterministic_for_same_seed():
    args = make_args(
        social_mode="none",
        social_network="random_k",
        rewiring="none",
        training_steps=4,
        evaluation_steps=3,
        record_every=2,
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
