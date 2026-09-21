from __future__ import annotations

import mesa
import numpy as np


LOW_EXTRACT = 0
HIGH_EXTRACT = 1

# Backward-compatible names used elsewhere in the project.
COOPERATE = LOW_EXTRACT
DEFECT = HIGH_EXTRACT


class EcoAgent(mesa.Agent):
    """Stationary agent with wealth, an energy reserve, and welfare records."""

    def __init__(
        self,
        model: "EcoModel",
        name: str,
        position: tuple[int, int],
        initial_energy: float,
        energy_capacity: float,
    ):
        super().__init__(model)

        self.name = name
        self.position = np.array(position, dtype=int)

        self.wealth = 0.0

        self.energy_capacity = float(energy_capacity)
        self.energy = min(float(initial_energy), self.energy_capacity)

        # Welfare accounting. These do not enter the Q-learning reward.
        self.metabolic_consumption = 0.0
        self.metabolic_shortfall = 0.0
        self.need_satisfaction = 1.0
        self.cumulative_shortfall = 0.0
        self.deprivation_steps = 0

    @property
    def reserve_welfare(self) -> float:
        """Bounded energy-reserve adequacy in [0, 1]."""

        if self.energy_capacity <= 0.0:
            return 0.0

        return float(
            np.clip(
                self.energy / self.energy_capacity,
                0.0,
                1.0,
            )
        )


class EcoModel(mesa.Model):
    """
    Stationary-agent common-pool resource model.

    Agents do not move. At each timestep they choose one of two extraction
    intensities:

        LOW_EXTRACT  = low extraction / cooperation proxy
        HIGH_EXTRACT = high extraction / defection proxy

    Welfare and wealth are intentionally separate:

    - wealth = cumulative resource appropriated;
    - energy = bounded reserve used to satisfy metabolism;
    - need_satisfaction = fraction of current metabolic need met;
    - cumulative_shortfall = cumulative unmet metabolic need.

    The Q-learning reward remains realized harvest, so welfare is an outcome,
    not an objective supplied to the learner.
    """

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        n_agents: int = 16,
        regeneration_rate: float | np.ndarray = 0.05,
        equilibrium_fraction: float | np.ndarray = 0.70,
        coupling_rate: float = 0.10,
        cooperative_harvest_amount: float = 0.002,
        defective_harvest_amount: float = 0.020,
        metabolism_rate: float = 0.002,
        initial_energy: float = 1.0,
        energy_capacity: float = 1.0,
        initial_resource_fraction: float = 0.50,
        capacity: np.ndarray | None = None,
        seed: int | None = None,
    ):
        super().__init__(rng=seed)

        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive.")
        if n_agents <= 0:
            raise ValueError("n_agents must be positive.")
        if not 0.0 <= coupling_rate <= 1.0:
            raise ValueError("coupling_rate must be between 0 and 1.")
        if cooperative_harvest_amount < 0.0:
            raise ValueError("cooperative_harvest_amount must be non-negative.")
        if defective_harvest_amount < cooperative_harvest_amount:
            raise ValueError(
                "defective_harvest_amount must be >= cooperative_harvest_amount."
            )
        if metabolism_rate < 0.0:
            raise ValueError("metabolism_rate must be non-negative.")
        if energy_capacity <= 0.0:
            raise ValueError("energy_capacity must be positive.")
        if initial_energy < 0.0 or initial_energy > energy_capacity:
            raise ValueError(
                "initial_energy must be between 0 and energy_capacity."
            )
        if not 0.0 <= initial_resource_fraction <= 1.0:
            raise ValueError("initial_resource_fraction must be between 0 and 1.")

        self.width = width
        self.height = height
        self.n_agents = n_agents

        self.coupling_rate = float(coupling_rate)
        self.cooperative_harvest_amount = float(cooperative_harvest_amount)
        self.defective_harvest_amount = float(defective_harvest_amount)
        self.metabolism_rate = float(metabolism_rate)
        self.initial_energy = float(initial_energy)
        self.energy_capacity = float(energy_capacity)
        self.initial_resource_fraction = float(initial_resource_fraction)

        self.capacity = self._capacity_field(capacity)

        self.regeneration_rate = regeneration_rate
        self.equilibrium_fraction = equilibrium_fraction

        self.regeneration_map = self._parameter_field(
            regeneration_rate,
            name="regeneration_rate",
            minimum=0.0,
            maximum=None,
        )

        self.equilibrium_map = self._parameter_field(
            equilibrium_fraction,
            name="equilibrium_fraction",
            minimum=0.0,
            maximum=1.0,
        )

        # Gives an isolated tile the positive equilibrium R* = qK.
        self.depletion_map = (
            self.regeneration_map
            * (1.0 - self.equilibrium_map)
        )

        self.resource = (
            self.initial_resource_fraction
            * self.capacity
        )

        self.actions: dict[str, int] = {}

        # Positions are random at initialization and fixed thereafter.
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
                energy_capacity=self.energy_capacity,
            )

    def _capacity_field(
        self,
        capacity: np.ndarray | None,
    ) -> np.ndarray:
        if capacity is None:
            return np.ones(
                (self.height, self.width),
                dtype=float,
            )

        field = np.asarray(
            capacity,
            dtype=float,
        )

        if field.shape != (self.height, self.width):
            raise ValueError(
                "capacity must have shape "
                f"({self.height}, {self.width}), got {field.shape}."
            )

        if np.any(field <= 0.0) or np.any(field > 1.0):
            raise ValueError(
                "capacity values must be in the interval (0, 1]."
            )

        return field.copy()

    def _parameter_field(
        self,
        value: float | np.ndarray,
        *,
        name: str,
        minimum: float,
        maximum: float | None,
    ) -> np.ndarray:
        if np.isscalar(value):
            field = np.full(
                (self.height, self.width),
                float(value),
                dtype=float,
            )
        else:
            field = np.asarray(
                value,
                dtype=float,
            )

            if field.shape != (self.height, self.width):
                raise ValueError(
                    f"{name} must have shape "
                    f"({self.height}, {self.width}), got {field.shape}."
                )

            field = field.copy()

        if np.any(field < minimum):
            raise ValueError(
                f"{name} must be >= {minimum}."
            )

        if maximum is not None and np.any(field > maximum):
            raise ValueError(
                f"{name} must be <= {maximum}."
            )

        return field

    def step(self) -> None:
        """Harvest, satisfy metabolism, then update the ecology."""

        self._harvest(self.actions)
        self._metabolize()
        self._update_ecology()

    def _harvest(
        self,
        actions: dict[str, int],
    ) -> dict[str, float]:
        """
        Resolve simultaneous extraction.

        If a tile cannot satisfy all requests, every request on that tile is
        reduced proportionally. Wealth receives the full realized harvest;
        energy is a bounded reserve and cannot exceed energy_capacity.
        """

        harvested = {
            name: 0.0
            for name in self.by_name
        }

        requests_by_tile: dict[
            tuple[int, int],
            list[tuple[str, float]],
        ] = {}

        for name, agent in self.by_name.items():
            action = actions.get(
                name,
                LOW_EXTRACT,
            )

            if action == LOW_EXTRACT:
                requested = self.cooperative_harvest_amount
            elif action == HIGH_EXTRACT:
                requested = self.defective_harvest_amount
            else:
                raise ValueError(
                    f"Unknown action {action}. Use 0=low or 1=high extraction."
                )

            x, y = agent.position

            requests_by_tile.setdefault(
                (int(x), int(y)),
                [],
            ).append((name, requested))

        for (x, y), requests in requests_by_tile.items():
            available = float(self.resource[y, x])

            total_requested = sum(
                amount
                for _, amount in requests
            )

            if total_requested <= 0.0:
                continue

            scale = min(
                1.0,
                available / total_requested,
            )

            total_harvest = 0.0

            for name, requested in requests:
                realized = requested * scale
                agent = self.by_name[name]

                agent.wealth += realized
                agent.energy = min(
                    agent.energy_capacity,
                    agent.energy + realized,
                )

                harvested[name] = realized
                total_harvest += realized

            self.resource[y, x] -= total_harvest

        return harvested

    def _metabolize(self) -> None:
        """Consume energy and record unmet metabolic need."""

        for agent in self.by_name.values():
            need = self.metabolism_rate

            if need <= 0.0:
                agent.metabolic_consumption = 0.0
                agent.metabolic_shortfall = 0.0
                agent.need_satisfaction = 1.0
                continue

            consumed = min(
                agent.energy,
                need,
            )

            shortfall = need - consumed

            agent.energy -= consumed
            agent.metabolic_consumption = consumed
            agent.metabolic_shortfall = shortfall
            agent.need_satisfaction = consumed / need
            agent.cumulative_shortfall += shortfall

            if shortfall > 1e-12:
                agent.deprivation_steps += 1

    def _update_ecology(self) -> None:
        """Spatial logistic reaction-diffusion update."""

        growth = (
            self.regeneration_map
            * self.resource
            * (
                1.0
                - self.resource / self.capacity
            )
        )

        depletion = (
            self.depletion_map
            * self.resource
        )

        neighbour_mean = self._neighbor_mean(
            self.resource
        )

        diffusion = (
            self.coupling_rate
            * (
                neighbour_mean
                - self.resource
            )
        )

        self.resource += (
            growth
            - depletion
            + diffusion
        )

        self.resource = np.clip(
            self.resource,
            0.0,
            self.capacity,
        )

    @staticmethod
    def _neighbor_mean(
        field: np.ndarray,
    ) -> np.ndarray:
        padded = np.pad(
            field,
            1,
            mode="edge",
        )

        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]

        return (
            up
            + down
            + left
            + right
        ) / 4.0
