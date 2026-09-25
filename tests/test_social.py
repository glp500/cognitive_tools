from __future__ import annotations

import numpy as np
import pytest

from Cognitive_tools.model import (
    HIGH_EXTRACT,
    LOW_EXTRACT,
)
from Cognitive_tools.social import (
    init_random_attention,
    joint_state,
    observed_low_fraction,
    social_bin,
    visibility_counts,
)


# ---------------------------------------------------------------------
# Fixed attention network
# ---------------------------------------------------------------------


def test_random_attention_has_fixed_k():
    agents = [
        f"agent_{index}"
        for index in range(
            10
        )
    ]

    rng = np.random.default_rng(
        42
    )

    sources = init_random_attention(
        agents,
        k=4,
        rng=rng,
    )

    assert set(
        sources
    ) == set(
        agents
    )

    for observer in agents:
        assert len(
            sources[
                observer
            ]
        ) == 4

        assert len(
            set(
                sources[
                    observer
                ]
            )
        ) == 4


def test_random_attention_has_no_self_links():
    agents = [
        f"agent_{index}"
        for index in range(
            8
        )
    ]

    sources = init_random_attention(
        agents,
        k=3,
        rng=np.random.default_rng(
            1
        ),
    )

    for observer in agents:
        assert (
            observer
            not in sources[
                observer
            ]
        )


def test_random_attention_is_seeded():
    agents = [
        f"agent_{index}"
        for index in range(
            8
        )
    ]

    first = init_random_attention(
        agents,
        k=4,
        rng=np.random.default_rng(
            123
        ),
    )

    second = init_random_attention(
        agents,
        k=4,
        rng=np.random.default_rng(
            123
        ),
    )

    assert first == second


@pytest.mark.parametrize(
    "k",
    [
        0,
        4,
        5,
    ],
)
def test_invalid_attention_capacity_is_rejected(
    k,
):
    agents = [
        "agent_0",
        "agent_1",
        "agent_2",
        "agent_3",
    ]

    with pytest.raises(
        ValueError
    ):
        init_random_attention(
            agents,
            k=k,
            rng=np.random.default_rng(
                0
            ),
        )


def test_duplicate_agent_names_are_rejected():
    agents = [
        "agent_0",
        "agent_0",
        "agent_1",
    ]

    with pytest.raises(
        ValueError
    ):
        init_random_attention(
            agents,
            k=1,
            rng=np.random.default_rng(
                0
            ),
        )


# ---------------------------------------------------------------------
# Social observation
# ---------------------------------------------------------------------


def test_observed_low_fraction():
    sources = {
        "agent_0": [
            "agent_1",
            "agent_2",
            "agent_3",
            "agent_4",
        ],
        "agent_1": [
            "agent_0",
        ],
        "agent_2": [
            "agent_0",
        ],
        "agent_3": [
            "agent_0",
        ],
        "agent_4": [
            "agent_0",
        ],
    }

    actions = {
        "agent_0": HIGH_EXTRACT,
        "agent_1": LOW_EXTRACT,
        "agent_2": LOW_EXTRACT,
        "agent_3": HIGH_EXTRACT,
        "agent_4": LOW_EXTRACT,
    }

    value = observed_low_fraction(
        "agent_0",
        sources,
        actions,
    )

    assert value == pytest.approx(
        0.75
    )


def test_first_social_observation_is_neutral():
    sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_0",
        ],
    }

    value = observed_low_fraction(
        "agent_0",
        sources,
        None,
    )

    assert value == pytest.approx(
        0.5
    )


def test_custom_neutral_social_observation():
    sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_0",
        ],
    }

    value = observed_low_fraction(
        "agent_0",
        sources,
        None,
        neutral=0.25,
    )

    assert value == pytest.approx(
        0.25
    )


def test_invalid_neutral_value_is_rejected():
    sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_0",
        ],
    }

    with pytest.raises(
        ValueError
    ):
        observed_low_fraction(
            "agent_0",
            sources,
            None,
            neutral=1.1,
        )


# ---------------------------------------------------------------------
# Social-state encoding
# ---------------------------------------------------------------------


def test_social_bin_boundaries():
    assert social_bin(
        0.0
    ) == 0

    assert social_bin(
        1.0 / 3.0
        - 1e-12
    ) == 0

    assert social_bin(
        1.0 / 3.0
    ) == 1

    assert social_bin(
        0.5
    ) == 1

    assert social_bin(
        2.0 / 3.0
        - 1e-12
    ) == 1

    assert social_bin(
        2.0 / 3.0
    ) == 2

    assert social_bin(
        1.0
    ) == 2


@pytest.mark.parametrize(
    "value",
    [
        -0.01,
        1.01,
    ],
)
def test_invalid_social_fraction_is_rejected(
    value,
):
    with pytest.raises(
        ValueError
    ):
        social_bin(
            value
        )


def test_joint_state_covers_zero_through_eight():
    states = {
        joint_state(
            ecological_state,
            social_state,
        )
        for ecological_state
        in range(
            3
        )
        for social_state
        in range(
            3
        )
    }

    assert states == set(
        range(
            9
        )
    )


def test_joint_state_order():
    assert joint_state(
        0,
        0,
    ) == 0

    assert joint_state(
        0,
        2,
    ) == 2

    assert joint_state(
        1,
        0,
    ) == 3

    assert joint_state(
        1,
        1,
    ) == 4

    assert joint_state(
        2,
        2,
    ) == 8


@pytest.mark.parametrize(
    "ecological_state,social_state",
    [
        (
            -1,
            0,
        ),
        (
            3,
            0,
        ),
        (
            0,
            -1,
        ),
        (
            0,
            3,
        ),
    ],
)
def test_invalid_joint_state_is_rejected(
    ecological_state,
    social_state,
):
    with pytest.raises(
        ValueError
    ):
        joint_state(
            ecological_state,
            social_state,
        )


# ---------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------


def test_visibility_counts_sum_to_n_times_k():
    agents = [
        f"agent_{index}"
        for index in range(
            12
        )
    ]

    k = 4

    sources = init_random_attention(
        agents,
        k=k,
        rng=np.random.default_rng(
            10
        ),
    )

    counts = visibility_counts(
        sources
    )

    assert set(
        counts
    ) == set(
        agents
    )

    assert sum(
        counts.values()
    ) == (
        len(
            agents
        )
        * k
    )


# ---------------------------------------------------------------------
# Baseline integration
# ---------------------------------------------------------------------


def test_baseline_state_encoding_is_unchanged():
    from baseline_validation_experiment import (
        encode_states,
    )

    observations = {
        "agent_0": {
            "local": np.array(
                [
                    0.5,
                    0.0,
                    0.0,
                ],
                dtype=float,
            )
        }
    }

    states = encode_states(
        observations,
        social_mode="none",
        sources=None,
        previous_actions=None,
    )

    assert states == {
        "agent_0": 1,
    }


def test_fixed_social_state_starts_neutral():
    from baseline_validation_experiment import (
        encode_states,
    )

    observations = {
        "agent_0": {
            "local": np.array(
                [
                    0.5,
                    0.0,
                    0.0,
                ],
                dtype=float,
            )
        },
        "agent_1": {
            "local": np.array(
                [
                    0.5,
                    0.0,
                    0.0,
                ],
                dtype=float,
            )
        },
    }

    sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_0",
        ],
    }

    states = encode_states(
        observations,
        social_mode="fixed",
        sources=sources,
        previous_actions=None,
    )

    # Ecological state 1 + neutral/mixed social state 1:
    #
    #     3 * 1 + 1 = 4
    assert states == {
        "agent_0": 4,
        "agent_1": 4,
    }


def test_fixed_social_state_uses_previous_actions():
    from baseline_validation_experiment import (
        encode_states,
    )

    observations = {
        "agent_0": {
            "local": np.array(
                [
                    0.5,
                    0.0,
                    0.0,
                ],
                dtype=float,
            )
        },
        "agent_1": {
            "local": np.array(
                [
                    0.5,
                    0.0,
                    0.0,
                ],
                dtype=float,
            )
        },
    }

    sources = {
        "agent_0": [
            "agent_1",
        ],
        "agent_1": [
            "agent_0",
        ],
    }

    previous_actions = {
        "agent_0": HIGH_EXTRACT,
        "agent_1": LOW_EXTRACT,
    }

    states = encode_states(
        observations,
        social_mode="fixed",
        sources=sources,
        previous_actions=(
            previous_actions
        ),
    )

    # agent_0 observes LOW:
    #
    # ecological = 1
    # social = 2
    # joint = 5
    assert states[
        "agent_0"
    ] == 5

    # agent_1 observes HIGH:
    #
    # ecological = 1
    # social = 0
    # joint = 3
    assert states[
        "agent_1"
    ] == 3