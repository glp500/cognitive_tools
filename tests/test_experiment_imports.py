from __future__ import annotations

import numpy as np

from cognitive_tools.model import (
    HIGH_EXTRACT,
    LOW_EXTRACT,
)


def test_baseline_always_low_action_rule():
    """
    The validation runner must use the physical low-extraction action.
    """

    from cognitive_tools.experiment import (
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

    from cognitive_tools.experiment import (
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

def test_package_imports():
    import cognitive_tools
    import cognitive_tools.experiment
    import cognitive_tools.analysis

    assert cognitive_tools.EcoEnv is not None
    assert callable(cognitive_tools.experiment.main)
    assert callable(cognitive_tools.analysis.main)


def test_canonical_module_clis():
    import subprocess
    import sys

    for module, option in [('cognitive_tools.experiment', '--rewiring'), ('cognitive_tools.analysis', '--run')]:
        completed = subprocess.run([sys.executable, '-m', module, '--help'], check=True, capture_output=True, text=True)
        assert option in completed.stdout
