from __future__ import annotations

import numpy as np

from Cognitive_tools.model import (
    HIGH_EXTRACT,
    LOW_EXTRACT,
)


def test_qlearning_experiment_imports():
    """
    The legacy Q-learning experiment must remain importable while it is
    still used by the canonical validation experiment.
    """

    import qlearning_experiment

    assert (
        qlearning_experiment
        is not None
    )


def test_baseline_validation_experiment_imports():
    """
    The canonical baseline experiment must import successfully.
    """

    import baseline_validation_experiment

    assert (
        baseline_validation_experiment
        is not None
    )


def test_baseline_always_low_action_rule():
    """
    The validation runner must use the physical low-extraction action.
    """

    from baseline_validation_experiment import (
        action_rule,
    )

    observations = {
        "agent_0": {
            "local": np.array(
                [
                    0.5,
                ],
                dtype=float,
            )
        }
    }

    rng = np.random.default_rng(
        0
    )

    (
        states,
        actions,
    ) = action_rule(
        strategy="always_low",
        observations=observations,
        learners=None,
        rng=rng,
    )

    assert states == {
        "agent_0": 1,
    }

    assert actions == {
        "agent_0": LOW_EXTRACT,
    }


def test_baseline_always_high_action_rule():
    """
    The validation runner must use the physical high-extraction action.
    """

    from baseline_validation_experiment import (
        action_rule,
    )

    observations = {
        "agent_0": {
            "local": np.array(
                [
                    0.5,
                ],
                dtype=float,
            )
        }
    }

    rng = np.random.default_rng(
        0
    )

    (
        states,
        actions,
    ) = action_rule(
        strategy="always_high",
        observations=observations,
        learners=None,
        rng=rng,
    )

    assert states == {
        "agent_0": 1,
    }

    assert actions == {
        "agent_0": HIGH_EXTRACT,
    }