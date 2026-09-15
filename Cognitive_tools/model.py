from __future__ import annotations

import mesa
import numpy as np

STAY = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
HARVEST = 5


class EcoAgent(mesa.Agent):
    """Agent with position, cumulative wealth, and current energy."""

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
        self.wealth = 0.0
        self.energy = float(initial_energy)


class EcoModel(mesa.Model):
    """
    Renewable common-pool resource model.

    Defaults reproduce the earlier model. Supplying a capacity map,
    equilibrium fraction, and coupling rate places the same random-action
    agents into heterogeneous reaction-diffusion environments.
    """

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        n_agents: int = 4,
        regeneration_rate: float = 0.05,
        harvest_amount: float = 0.25,
        metabolism_rate: float = 0.05,
        initial_energy: float = 1.0,
        equilibrium_fraction: float = 1.0,
        coupling_rate: float = 0.0,
        initial_resource_fraction: float = 0.5,
        capacity: np.ndarray | None = None,
        seed: int | None = None,
    ):
        super().__init__(rng=seed)

        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive.")
        if n_agents <= 0:
            raise ValueError("n_agents must be positive.")
        if regeneration_rate < 0:
            raise ValueError("regeneration_rate must be non-negative.")
        if harvest_amount < 0:
            raise ValueError("harvest_amount must be non-negative.")
        if metabolism_rate < 0:
            raise ValueError("metabolism_rate must be non-negative.")
        if initial_energy < 0:
            raise ValueError("initial_energy must be non-negative.")
        if not 0 <= equilibrium_fraction <= 1:
            raise ValueError("equilibrium_fraction must be between 0 and 1.")
        if not 0 <= coupling_rate <= 1:
            raise ValueError("coupling_rate must be between 0 and 1.")
        if not 0 <= initial_resource_fraction <= 1:
            raise ValueError("initial_resource_fraction must be between 0 and 1.")

        self.width = width
        self.height = height
        self.regeneration_rate = float(regeneration_rate)
        self.harvest_amount = float(harvest_amount)
        self.metabolism_rate = float(metabolism_rate)
        self.initial_energy = float(initial_energy)
        self.equilibrium_fraction = float(equilibrium_fraction)
        self.coupling_rate = float(coupling_rate)
        self.initial_resource_fraction = float(initial_resource_fraction)

        # Gives an isolated tile the positive equilibrium R* = qK.
        self.depletion_rate = self.regeneration_rate * (1 - self.equilibrium_fraction)

        self.actions: dict[str, int] = {}

        if capacity is None:
            self.capacity = np.ones((height, width), dtype=float)
        else:
            capacity = np.asarray(capacity, dtype=float)
            if capacity.shape != (height, width):
                raise ValueError(
                    f"capacity must have shape ({height}, {width}), got {capacity.shape}."
                )
            if np.any(capacity <= 0) or np.any(capacity > 1):
                raise ValueError("capacity values must be in the interval (0, 1].")
            self.capacity = capacity.copy()

        self.resource = self.initial_resource_fraction * self.capacity

        self.by_name: dict[str, EcoAgent] = {}
        for i in range(n_agents):
            name = f"agent_{i}"
            x = int(self.rng.integers(width))
            y = int(self.rng.integers(height))
            self.by_name[name] = EcoAgent(
                model=self,
                name=name,
                position=(x, y),
                initial_energy=self.initial_energy,
            )

    def step(self) -> dict[str, float]:
        """movement -> harvest -> metabolism -> ecology"""

        self._move_agents(self.actions)
        harvested = self._harvest(self.actions)
        self._metabolize()
        self._update_ecology()
        return harvested

    def _move_agents(self, actions: dict[str, int]) -> None:
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

            x = np.clip(agent.position[0] + dx, 0, self.width - 1)
            y = np.clip(agent.position[1] + dy, 0, self.height - 1)
            agent.position[:] = (x, y)

    def _harvest(self, actions: dict[str, int]) -> dict[str, float]:
        """Resolve harvest requests simultaneously within each occupied cell."""

        harvested = {name: 0.0 for name in self.by_name}
        harvesters: dict[tuple[int, int], list[str]] = {}

        for name, action in actions.items():
            if action != HARVEST:
                continue

            x, y = self.by_name[name].position
            harvesters.setdefault((int(x), int(y)), []).append(name)

        for (x, y), names in harvesters.items():
            available = float(self.resource[y, x])
            requested = self.harvest_amount * len(names)
            total_harvest = min(available, requested)
            share = total_harvest / len(names)

            self.resource[y, x] -= total_harvest

            for name in names:
                agent = self.by_name[name]
                agent.wealth += share
                agent.energy += share
                harvested[name] = share

        return harvested

    def _metabolize(self) -> None:
        for agent in self.by_name.values():
            agent.energy = max(0.0, agent.energy - self.metabolism_rate)

    def _update_ecology(self) -> None:
        """
        Spatial logistic reaction-diffusion update.

        Local reaction:
            rR(1 - R/K) - dR

        Spatial exchange:
            c(mean_neighbor_resource - R)
        """

        growth = (
            self.regeneration_rate
            * self.resource
            * (1 - self.resource / self.capacity)
        )
        depletion = self.depletion_rate * self.resource
        neighbor_mean = self._neighbor_mean(self.resource)
        diffusion = self.coupling_rate * (neighbor_mean - self.resource)

        self.resource += growth - depletion + diffusion
        self.resource = np.clip(self.resource, 0.0, self.capacity)

    @staticmethod
    def _neighbor_mean(field: np.ndarray) -> np.ndarray:
        """Mean of the four adjacent cells, with no wraparound."""

        padded = np.pad(field, 1, mode="edge")
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        return (up + down + left + right) / 4.0
