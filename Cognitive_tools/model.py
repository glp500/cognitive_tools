from __future__ import annotations

import mesa
import numpy as np


# Actions
STAY = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
HARVEST = 5


class EcoAgent(mesa.Agent):
    """Agent with position, cumulative wealth, and energy."""

    def __init__(
        self,
        model: "EcoModel",
        name: str,
        position: tuple[int, int],
        initial_energy: float,
    ):
        super().__init__(model)

        self.name = name
        self.position = np.array(position, dtype=int)

        # Cumulative amount ever harvested.
        self.wealth = 0.0

        # Current material reserve.
        self.energy = initial_energy

class EcoModel(mesa.Model):
    """Minimal renewable common-pool resource model."""

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        n_agents: int = 4,
        regeneration_rate: float = 0.05,
        harvest_amount: float = 0.25,
        metabolism_rate: float = 0.05,
        initial_energy: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__(rng=seed)

        self.width = width
        self.height = height
        self.regeneration_rate = regeneration_rate
        self.harvest_amount = harvest_amount
        self.metabolism_rate = metabolism_rate
        self.initial_energy = initial_energy

        # Joint actions supplied by the PettingZoo wrapper before each Mesa step.
        self.actions: dict[str, int] = {}

        # Ecological state.
        self.capacity = np.ones((height, width), dtype=float)
        self.resource = 0.5 * self.capacity

        # Agents.
        self.by_name: dict[str, EcoAgent] = {}

        for i in range(n_agents):
            name = f"agent_{i}"

            x = int(self.rng.integers(width))
            y = int(self.rng.integers(height))

            agent = EcoAgent(
                model=self,
                name=name,
                position=(x, y),
                initial_energy=self.initial_energy,
            )

            self.by_name[name] = agent

    def step(self) -> dict[str, float]:
        """
        Advance the ecological model one timestep using the current joint actions.

        Returns the amount harvested by each agent.
        """

        actions = self.actions

        self._move_agents(actions)

        harvest = self._harvest(actions)

        self._metabolize()

        self._regenerate()

        return harvest

    def _move_agents(self, actions: dict[str, int]) -> None:
        """Resolve all movement actions."""

        moves = {
            UP: (0, -1),
            DOWN: (0, 1),
            LEFT: (-1, 0),
            RIGHT: (1, 0),
        }

        for name, action in actions.items():
            if action not in moves:
                continue

            agent = self.by_name[name]

            dx, dy = moves[action]

            x = np.clip(
                agent.position[0] + dx,
                0,
                self.width - 1,
            )
            y = np.clip(
                agent.position[1] + dy,
                0,
                self.height - 1,
            )

            agent.position[:] = (x, y)

    def _harvest(
        self,
        actions: dict[str, int],
    ) -> dict[str, float]:
        """
        Resolve harvesting simultaneously.

        Agents sharing a cell divide the available resource equally.
        """

        harvested = {
            name: 0.0
            for name in self.by_name
        }

        harvesters: dict[tuple[int, int], list[str]] = {}

        for name, action in actions.items():
            if action != HARVEST:
                continue

            agent = self.by_name[name]

            x, y = agent.position

            harvesters.setdefault((x, y), []).append(name)

        for (x, y), names in harvesters.items():

            available = self.resource[y, x]

            requested = self.harvest_amount * len(names)

            total_harvest = min(
                available,
                requested,
            )

            share = total_harvest / len(names)

            self.resource[y, x] -= total_harvest

            for name in names:
                agent = self.by_name[name]
                agent.wealth += share
                agent.energy += share
                harvested[name] = share

        return harvested

    def _regenerate(self) -> None:
        """Logistic resource regeneration."""

        growth = (
            self.regeneration_rate
            * self.resource
            * (1.0 - self.resource / self.capacity)
        )

        self.resource += growth

        self.resource = np.clip(
            self.resource,
            0.0,
            self.capacity,
        )

    def _metabolize(self) -> None:
        """Consume each agent's metabolic requirement."""

        for agent in self.by_name.values():

            agent.energy -= self.metabolism_rate

            agent.energy = max(
                0.0,
                agent.energy,
            )