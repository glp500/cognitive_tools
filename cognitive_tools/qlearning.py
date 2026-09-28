"""
Generic independent tabular Q-learning used by cognitive_tools.

The learner in this module is deliberately agnostic to ecology, social
networks, and the meaning of actions. It receives integer state/action
indices and scalar rewards only.

Scientific provenance
---------------------
The update rule is standard one-step Q-learning:

    Q(s, a) <- Q(s, a)
               + alpha [r + gamma max_a' Q(s', a') - Q(s, a)]

with terminal target equal to the immediate reward.

Reference:

    Watkins, C. J. C. H., & Dayan, P. (1992).
    "Q-learning."
    Machine Learning, 8, 279-292.
    https://doi.org/10.1007/BF00992698

No source code from Watkins & Dayan is copied here. This is an
independent implementation of the published algorithm.

The ecological state encoder remains in this module for now because the
current baseline uses a small tabular ecological state representation.
QLearningPolicy itself does not inspect observations or know what a
state means.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------
# Current ecological baseline states
# ---------------------------------------------------------------------

STATE_NAMES = (
    "scarce",
    "moderate",
    "abundant",
)


def resource_state_from_fraction(
    resource_fraction: float,
) -> int:
    """
    Discretize local resource condition R/K into three ecological states.

        0: scarce
           R/K < 1/3

        1: moderate
           1/3 <= R/K < 2/3

        2: abundant
           R/K >= 2/3

    This function contains the ecological interpretation.

    QLearningPolicy itself remains independent of this encoding.
    """

    value = float(
        resource_fraction
    )

    if value < 1.0 / 3.0:
        return 0

    if value < 2.0 / 3.0:
        return 1

    return 2


def resource_state(
    observation: dict,
) -> int:
    """
    Convert the current EcoEnv observation into the baseline ecological
    state.

    Only local R/K is used by the Q-learning baseline.

    This small adapter is retained so existing experiment scripts do not
    change behavior during this refactor.
    """

    return resource_state_from_fraction(
        observation["local"][0]
    )


# ---------------------------------------------------------------------
# Generic tabular learner
# ---------------------------------------------------------------------


class QLearningPolicy:
    """
    Minimal independent tabular Q-learning policy.

    Parameters
    ----------
    n_states
        Number of discrete learner states.

    n_actions
        Number of discrete actions.

    alpha
        Learning rate.

    gamma
        Discount factor.

    epsilon
        Initial epsilon-greedy exploration probability.

    epsilon_min
        Minimum exploration probability.

    epsilon_decay
        Multiplicative epsilon decay after each update period.

    seed
        Random seed for this learner.

    Notes
    -----
    The defaults n_states=3 and n_actions=2 exactly match the current
    ecological baseline.

    The class deliberately has no knowledge of:

    - resources,
    - scarcity,
    - extraction,
    - cooperation,
    - agents,
    - social networks.

    Those meanings belong to the experiment/state-encoding layer.
    """

    def __init__(
        self,
        *,
        n_states: int = 3,
        n_actions: int = 2,
        alpha: float = 0.10,
        gamma: float = 0.95,
        epsilon: float = 0.20,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.9995,
        seed: int | None = None,
    ):
        # -------------------------------------------------------------
        # Validate table dimensions
        # -------------------------------------------------------------

        if n_states <= 0:
            raise ValueError(
                "n_states must be positive."
            )

        if n_actions <= 0:
            raise ValueError(
                "n_actions must be positive."
            )

        # -------------------------------------------------------------
        # Validate learning parameters
        # -------------------------------------------------------------

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "alpha must be between 0 and 1."
            )

        if not 0.0 <= gamma <= 1.0:
            raise ValueError(
                "gamma must be between 0 and 1."
            )

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "epsilon must be between 0 and 1."
            )

        if not 0.0 <= epsilon_min <= 1.0:
            raise ValueError(
                "epsilon_min must be between 0 and 1."
            )

        if epsilon_min > epsilon:
            raise ValueError(
                "epsilon_min must be <= epsilon."
            )

        if not 0.0 <= epsilon_decay <= 1.0:
            raise ValueError(
                "epsilon_decay must be between 0 and 1."
            )

        # -------------------------------------------------------------
        # Store configuration
        # -------------------------------------------------------------

        self.n_states = int(
            n_states
        )

        self.n_actions = int(
            n_actions
        )

        self.alpha = float(
            alpha
        )

        self.gamma = float(
            gamma
        )

        self.epsilon = float(
            epsilon
        )

        self.epsilon_min = float(
            epsilon_min
        )

        self.epsilon_decay = float(
            epsilon_decay
        )

        self.rng = np.random.default_rng(
            seed
        )

        # Tiny random initialization removes deterministic tie-breaking
        # toward action 0 while remaining effectively zero.
        #
        # For the default 3 x 2 baseline this is the same RNG call and
        # shape used before this refactor.
        self.q = self.rng.normal(
            loc=0.0,
            scale=1e-8,
            size=(
                self.n_states,
                self.n_actions,
            ),
        )

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def _check_state(
        self,
        state: int,
    ) -> int:
        state = int(
            state
        )

        if not (
            0
            <= state
            < self.n_states
        ):
            raise IndexError(
                f"state {state} outside "
                f"[0, {self.n_states - 1}]."
            )

        return state

    def _check_action(
        self,
        action: int,
    ) -> int:
        action = int(
            action
        )

        if not (
            0
            <= action
            < self.n_actions
        ):
            raise IndexError(
                f"action {action} outside "
                f"[0, {self.n_actions - 1}]."
            )

        return action

    # -----------------------------------------------------------------
    # Action selection
    # -----------------------------------------------------------------

    def choose_action(
        self,
        state: int,
        *,
        explore: bool = True,
    ) -> int:
        """
        Choose an epsilon-greedy action.
        """

        state = self._check_state(
            state
        )

        if (
            explore
            and self.rng.random()
            < self.epsilon
        ):
            return int(
                self.rng.integers(
                    self.n_actions
                )
            )

        return int(
            np.argmax(
                self.q[
                    state
                ]
            )
        )

    # -----------------------------------------------------------------
    # Learning
    # -----------------------------------------------------------------

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        *,
        done: bool,
    ) -> None:
        """
        Apply one standard Q-learning update.
        """

        state = self._check_state(
            state
        )

        action = self._check_action(
            action
        )

        next_state = self._check_state(
            next_state
        )

        current = self.q[
            state,
            action,
        ]

        if done:
            target = float(
                reward
            )

        else:
            target = (
                float(reward)
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

    # -----------------------------------------------------------------
    # Exploration
    # -----------------------------------------------------------------

    def decay_exploration(
        self,
    ) -> None:
        """
        Apply multiplicative epsilon decay with a lower bound.
        """

        self.epsilon = max(
            self.epsilon_min,
            (
                self.epsilon
                * self.epsilon_decay
            ),
        )

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def greedy_policy(
        self,
    ) -> tuple[int, ...]:
        """
        Return the greedy action for every discrete state.
        """

        return tuple(
            int(
                np.argmax(
                    self.q[
                        state
                    ]
                )
            )
            for state in range(
                self.n_states
            )
        )