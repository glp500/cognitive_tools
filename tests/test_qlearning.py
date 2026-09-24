from __future__ import annotations

import numpy as np
import pytest

from Cognitive_tools.qlearning import (
    QLearningPolicy,
    resource_state,
    resource_state_from_fraction,
)


# ---------------------------------------------------------------------
# Baseline compatibility
# ---------------------------------------------------------------------


def test_default_shape_preserves_baseline():
    """
    The default learner must retain the current ecological baseline:
    three states and two actions.
    """

    learner = QLearningPolicy(
        seed=42
    )

    assert learner.n_states == 3
    assert learner.n_actions == 2

    assert learner.q.shape == (
        3,
        2,
    )


def test_default_initialization_matches_pre_refactor_rng_call():
    """
    The generic learner must reproduce the same seeded Q-table
    initialization as the previous fixed 3 x 2 implementation.
    """

    seed = 42

    expected = np.random.default_rng(
        seed
    ).normal(
        loc=0.0,
        scale=1e-8,
        size=(
            3,
            2,
        ),
    )

    learner = QLearningPolicy(
        seed=seed
    )

    np.testing.assert_array_equal(
        learner.q,
        expected,
    )


# ---------------------------------------------------------------------
# Generic table dimensions
# ---------------------------------------------------------------------


def test_generic_nine_state_two_action_table():
    """
    The future ecology x social representation will require nine states.

    This test establishes that the learning algorithm itself does not
    need to change when that representation is introduced.
    """

    learner = QLearningPolicy(
        n_states=9,
        n_actions=2,
        seed=1,
    )

    assert learner.n_states == 9
    assert learner.n_actions == 2

    assert learner.q.shape == (
        9,
        2,
    )

    assert len(
        learner.greedy_policy()
    ) == 9


def test_generic_action_count_controls_exploration():
    """
    Exploration must use n_actions rather than assuming two actions.
    """

    learner = QLearningPolicy(
        n_states=4,
        n_actions=5,
        epsilon=1.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        seed=7,
    )

    actions = {
        learner.choose_action(
            state=0,
            explore=True,
        )
        for _ in range(
            200
        )
    }

    assert actions <= set(
        range(5)
    )

    # If action selection were still hard-coded to two actions,
    # this would fail.
    assert len(
        actions
    ) > 2


# ---------------------------------------------------------------------
# Q-learning mathematics
# ---------------------------------------------------------------------


def test_q_update_matches_hand_calculation():
    """
    Check the ordinary non-terminal Q-learning update against a
    hand-calculated result.
    """

    learner = QLearningPolicy(
        n_states=3,
        n_actions=2,
        alpha=0.5,
        gamma=0.9,
        epsilon=0.0,
        epsilon_min=0.0,
        seed=0,
    )

    learner.q[:] = 0.0

    learner.q[
        1
    ] = [
        2.0,
        4.0,
    ]

    learner.update(
        state=0,
        action=1,
        reward=1.0,
        next_state=1,
        done=False,
    )

    # target
    # =
    # reward + gamma * max_a Q(next_state, a)
    # =
    # 1 + 0.9 * 4
    expected_target = (
        1.0
        + 0.9 * 4.0
    )

    # updated Q
    # =
    # old + alpha * (target - old)
    expected = (
        0.0
        + 0.5
        * expected_target
    )

    assert learner.q[
        0,
        1,
    ] == pytest.approx(
        expected
    )


def test_terminal_update_uses_reward_only():
    """
    Terminal transitions must not bootstrap from next-state Q values.
    """

    learner = QLearningPolicy(
        alpha=0.25,
        gamma=0.99,
        epsilon=0.0,
        epsilon_min=0.0,
        seed=0,
    )

    learner.q[:] = 10.0

    learner.q[
        0,
        0,
    ] = 2.0

    learner.update(
        state=0,
        action=0,
        reward=6.0,
        next_state=1,
        done=True,
    )

    expected = (
        2.0
        + 0.25
        * (
            6.0
            - 2.0
        )
    )

    assert learner.q[
        0,
        0,
    ] == pytest.approx(
        expected
    )


# ---------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------


def test_greedy_action_uses_state_specific_q_values():
    learner = QLearningPolicy(
        n_states=2,
        n_actions=3,
        epsilon=0.0,
        epsilon_min=0.0,
        seed=0,
    )

    learner.q[:] = [
        [
            0.0,
            3.0,
            1.0,
        ],
        [
            5.0,
            1.0,
            2.0,
        ],
    ]

    assert learner.choose_action(
        state=0,
        explore=False,
    ) == 1

    assert learner.choose_action(
        state=1,
        explore=False,
    ) == 0

    assert learner.greedy_policy() == (
        1,
        0,
    )


# ---------------------------------------------------------------------
# Ecological state encoding
# ---------------------------------------------------------------------


def test_resource_state_boundaries():
    assert resource_state_from_fraction(
        0.0
    ) == 0

    assert resource_state_from_fraction(
        1.0 / 3.0
        - 1e-12
    ) == 0

    assert resource_state_from_fraction(
        1.0 / 3.0
    ) == 1

    assert resource_state_from_fraction(
        2.0 / 3.0
        - 1e-12
    ) == 1

    assert resource_state_from_fraction(
        2.0 / 3.0
    ) == 2

    assert resource_state_from_fraction(
        1.0
    ) == 2


def test_resource_state_observation_adapter():
    """
    The existing environment adapter must continue using local R/K.

    Wealth and energy are deliberately irrelevant to the ecological
    state encoder.
    """

    observation = {
        "local": np.array(
            [
                0.75,
                100.0,
                50.0,
            ],
            dtype=float,
        )
    }

    assert resource_state(
        observation
    ) == 2


# ---------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "n_states": 0,
        },
        {
            "n_actions": 0,
        },
        {
            "alpha": -0.1,
        },
        {
            "alpha": 1.1,
        },
        {
            "gamma": -0.1,
        },
        {
            "gamma": 1.1,
        },
        {
            "epsilon": -0.1,
        },
        {
            "epsilon": 1.1,
        },
        {
            "epsilon_min": -0.1,
        },
        {
            "epsilon_min": 0.5,
            "epsilon": 0.2,
        },
        {
            "epsilon_decay": -0.1,
        },
        {
            "epsilon_decay": 1.1,
        },
    ],
)
def test_invalid_configuration_is_rejected(
    kwargs,
):
    with pytest.raises(
        ValueError
    ):
        QLearningPolicy(
            **kwargs
        )


def test_invalid_state_index_is_rejected():
    learner = QLearningPolicy(
        n_states=3,
        n_actions=2,
        seed=0,
    )

    with pytest.raises(
        IndexError
    ):
        learner.choose_action(
            state=3
        )


def test_negative_state_index_is_rejected():
    learner = QLearningPolicy(
        n_states=3,
        n_actions=2,
        seed=0,
    )

    with pytest.raises(
        IndexError
    ):
        learner.choose_action(
            state=-1
        )


def test_invalid_action_index_is_rejected():
    learner = QLearningPolicy(
        n_states=3,
        n_actions=2,
        seed=0,
    )

    with pytest.raises(
        IndexError
    ):
        learner.update(
            state=0,
            action=2,
            reward=0.0,
            next_state=1,
            done=False,
        )


def test_invalid_next_state_is_rejected():
    learner = QLearningPolicy(
        n_states=3,
        n_actions=2,
        seed=0,
    )

    with pytest.raises(
        IndexError
    ):
        learner.update(
            state=0,
            action=0,
            reward=0.0,
            next_state=3,
            done=False,
        )


# ---------------------------------------------------------------------
# Exploration decay
# ---------------------------------------------------------------------


def test_epsilon_decay():
    learner = QLearningPolicy(
        epsilon=0.20,
        epsilon_min=0.02,
        epsilon_decay=0.50,
        seed=0,
    )

    learner.decay_exploration()

    assert learner.epsilon == pytest.approx(
        0.10
    )

    learner.decay_exploration()

    assert learner.epsilon == pytest.approx(
        0.05
    )

    learner.decay_exploration()

    assert learner.epsilon == pytest.approx(
        0.025
    )

    learner.decay_exploration()

    assert learner.epsilon == pytest.approx(
        0.02
    )

    learner.decay_exploration()

    assert learner.epsilon == pytest.approx(
        0.02
    )