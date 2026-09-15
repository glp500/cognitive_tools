from __future__ import annotations

import numpy as np


COOPERATE = 0
DEFECT = 1

STATE_NAMES = (
    "scarce",
    "moderate",
    "abundant",
)


def resource_state(
    observation: dict,
) -> int:
    """
    Convert local resource condition R/K into one of three states.

        0: scarce    R/K < 1/3
        1: moderate  1/3 <= R/K < 2/3
        2: abundant  R/K >= 2/3
    """

    resource_fraction = float(
        observation["local"][0]
    )

    if resource_fraction < 1.0 / 3.0:
        return 0

    if resource_fraction < 2.0 / 3.0:
        return 1

    return 2


class QLearningPolicy:
    """Minimal independent tabular Q-learning policy."""

    def __init__(
        self,
        *,
        alpha: float = 0.10,
        gamma: float = 0.95,
        epsilon: float = 0.20,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.9995,
        seed: int | None = None,
    ):
        self.alpha = float(alpha)
        self.gamma = float(gamma)

        self.epsilon = float(epsilon)
        self.epsilon_min = float(
            epsilon_min
        )
        self.epsilon_decay = float(
            epsilon_decay
        )

        self.rng = np.random.default_rng(
            seed
        )

        # Tiny random initialization removes a deterministic
        # cooperate-first tie bias while remaining effectively zero.
        self.q = self.rng.normal(
            loc=0.0,
            scale=1e-8,
            size=(3, 2),
        )

    def choose_action(
        self,
        state: int,
        *,
        explore: bool = True,
    ) -> int:
        if (
            explore
            and self.rng.random()
            < self.epsilon
        ):
            return int(
                self.rng.integers(2)
            )

        return int(
            np.argmax(
                self.q[state]
            )
        )

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        *,
        done: bool,
    ) -> None:
        current = self.q[
            state,
            action,
        ]

        if done:
            target = reward
        else:
            target = (
                reward
                + self.gamma
                * float(
                    np.max(
                        self.q[
                            next_state
                        ]
                    )
                )
            )

        self.q[
            state,
            action,
        ] = (
            current
            + self.alpha
            * (
                target
                - current
            )
        )

    def decay_exploration(
        self,
    ) -> None:
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon
            * self.epsilon_decay,
        )

    def greedy_policy(
        self,
    ) -> tuple[int, int, int]:
        return tuple(
            int(
                np.argmax(
                    self.q[state]
                )
            )
            for state in range(3)
        )
