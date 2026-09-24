"""
Stationary-agent common-pool-resource model.

`EcoModel` owns simulation state:

- resource stock;
- stationary agents;
- extraction;
- wealth;
- energy/welfare bookkeeping.

The actual renewable-resource equation is implemented once, in
`Cognitive_tools.ecology.resource_step`.

Scientific provenance
---------------------
The renewable-resource feedback is conceptually related to:

    Tilman, A. R., Plotkin, J. B., & Akcay, E. (2020).
    "Evolutionary games with environmental feedbacks."
    Nature Communications, 11, 915.
    https://doi.org/10.1038/s41467-020-14531-6

and:

    Tu, C., Wu, Y., Chen, R., Fan, Y., & Yang, Y. (2025).
    "Balancing Resource and Strategy: Coevolution for Sustainable
    Common-Pool Resource Management."
    Earth Systems and Environment, 9, 1529-1542.
    https://doi.org/10.1007/s41748-024-00489-8

This source file is not copied from either work. See PROVENANCE.md.
"""

from __future__ import annotations

import mesa
import numpy as np

from .ecology import (
    coerce_capacity_map,
    coerce_parameter_map,
    depletion_from_equilibrium,
    resource_step,
)


LOW_EXTRACT = 0
HIGH_EXTRACT = 1

# Temporary backward-compatible aliases.
#
# These remain during the ecological refactor so that this commit changes
# no behavioral semantics. A later terminology-cleanup commit should remove
# them from the mechanical model and use LOW_EXTRACT/HIGH_EXTRACT directly.
COOPERATE = LOW_EXTRACT
DEFECT = HIGH_EXTRACT


class EcoAgent(mesa.Agent):
    """
    Stationary resource user.

    Wealth is cumulative realized extraction.

    Energy and metabolic variables are diagnostic welfare quantities and
    do not enter the current Q-learning reward.
    """

    def __init__(
        self,
        model: "EcoModel",
        name: str,
        position: tuple[int, int],
        initial_energy: float,
        energy_capacity: float,
    ):
        super().__init__(
            model
        )

        self.name = name

        self.position = np.array(
            position,
            dtype=int,
        )

        self.wealth = 0.0

        self.energy_capacity = float(
            energy_capacity
        )

        self.energy = min(
            float(initial_energy),
            self.energy_capacity,
        )

        # Welfare accounting.
        self.metabolic_consumption = 0.0
        self.metabolic_shortfall = 0.0
        self.need_satisfaction = 1.0
        self.cumulative_shortfall = 0.0
        self.deprivation_steps = 0

    @property
    def reserve_welfare(
        self,
    ) -> float:
        """
        Energy reserve as a fraction of maximum reserve capacity.
        """

        if self.energy_capacity <= 0.0:
            return 0.0

        return float(
            np.clip(
                (
                    self.energy
                    / self.energy_capacity
                ),
                0.0,
                1.0,
            )
        )


class EcoModel(mesa.Model):
    """
    Spatial common-pool-resource model with stationary users.

    Timestep ordering
    -----------------
    Each timestep preserves the current baseline ordering:

        1. requested extraction is resolved;
        2. realized harvest is removed from the resource;
        3. realized harvest enters wealth and energy;
        4. metabolism is applied;
        5. the remaining resource undergoes ecological dynamics.

    Agents do not move.

    Actions
    -------
    LOW_EXTRACT
        Small requested harvest.

    HIGH_EXTRACT
        Larger requested harvest.

    Reward
    ------
    The PettingZoo wrapper uses realized harvest as the Q-learning reward.

    Welfare and ecological sustainability are outcomes, not reward terms.
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
        super().__init__(
            rng=seed
        )

        # -------------------------------------------------------------
        # Validate scalar model settings
        # -------------------------------------------------------------

        if width <= 0 or height <= 0:
            raise ValueError(
                "width and height must be positive."
            )

        if n_agents <= 0:
            raise ValueError(
                "n_agents must be positive."
            )

        if not 0.0 <= coupling_rate <= 1.0:
            raise ValueError(
                "coupling_rate must be between 0 and 1."
            )

        if cooperative_harvest_amount < 0.0:
            raise ValueError(
                "cooperative_harvest_amount "
                "must be non-negative."
            )

        if (
            defective_harvest_amount
            < cooperative_harvest_amount
        ):
            raise ValueError(
                "defective_harvest_amount must be "
                ">= cooperative_harvest_amount."
            )

        if metabolism_rate < 0.0:
            raise ValueError(
                "metabolism_rate must be non-negative."
            )

        if energy_capacity <= 0.0:
            raise ValueError(
                "energy_capacity must be positive."
            )

        if (
            initial_energy < 0.0
            or initial_energy > energy_capacity
        ):
            raise ValueError(
                "initial_energy must be between "
                "0 and energy_capacity."
            )

        if not (
            0.0
            <= initial_resource_fraction
            <= 1.0
        ):
            raise ValueError(
                "initial_resource_fraction "
                "must be between 0 and 1."
            )

        # -------------------------------------------------------------
        # Model settings
        # -------------------------------------------------------------

        self.width = int(
            width
        )

        self.height = int(
            height
        )

        self.n_agents = int(
            n_agents
        )

        self.coupling_rate = float(
            coupling_rate
        )

        self.cooperative_harvest_amount = float(
            cooperative_harvest_amount
        )

        self.defective_harvest_amount = float(
            defective_harvest_amount
        )

        self.metabolism_rate = float(
            metabolism_rate
        )

        self.initial_energy = float(
            initial_energy
        )

        self.energy_capacity = float(
            energy_capacity
        )

        self.initial_resource_fraction = float(
            initial_resource_fraction
        )

        # -------------------------------------------------------------
        # Ecological fields
        # -------------------------------------------------------------

        self.capacity = coerce_capacity_map(
            capacity,
            width=self.width,
            height=self.height,
        )

        self.regeneration_rate = (
            regeneration_rate
        )

        self.equilibrium_fraction = (
            equilibrium_fraction
        )

        self.regeneration_map = (
            coerce_parameter_map(
                regeneration_rate,
                width=self.width,
                height=self.height,
                name="regeneration_rate",
                minimum=0.0,
            )
        )

        self.equilibrium_map = (
            coerce_parameter_map(
                equilibrium_fraction,
                width=self.width,
                height=self.height,
                name="equilibrium_fraction",
                minimum=0.0,
                maximum=1.0,
            )
        )

        self.depletion_map = (
            depletion_from_equilibrium(
                self.regeneration_map,
                self.equilibrium_map,
            )
        )

        self.resource = (
            self.initial_resource_fraction
            * self.capacity
        )

        # -------------------------------------------------------------
        # Agent state
        # -------------------------------------------------------------

        self.actions: dict[
            str,
            int,
        ] = {}

        self.by_name: dict[
            str,
            EcoAgent,
        ] = {}

        # Agent positions are sampled once and then remain fixed.
        for index in range(
            self.n_agents
        ):
            name = (
                f"agent_{index}"
            )

            x = int(
                self.rng.integers(
                    self.width
                )
            )

            y = int(
                self.rng.integers(
                    self.height
                )
            )

            self.by_name[
                name
            ] = EcoAgent(
                model=self,
                name=name,
                position=(
                    x,
                    y,
                ),
                initial_energy=(
                    self.initial_energy
                ),
                energy_capacity=(
                    self.energy_capacity
                ),
            )

    # -----------------------------------------------------------------
    # Main timestep
    # -----------------------------------------------------------------

    def step(
        self,
    ) -> None:
        """
        Advance the coupled agent-resource system by one timestep.
        """

        self._harvest(
            self.actions
        )

        self._metabolize()

        self._update_ecology()

    # -----------------------------------------------------------------
    # Extraction
    # -----------------------------------------------------------------

    def _harvest(
        self,
        actions: dict[
            str,
            int,
        ],
    ) -> dict[
        str,
        float,
    ]:
        """
        Resolve simultaneous extraction.

        If agents on the same tile collectively request more resource than
        the tile contains, every request on that tile is scaled by the same
        factor.

        Wealth receives the full realized harvest.

        Energy is a bounded reserve and cannot exceed energy_capacity.
        """

        harvested = {
            name: 0.0
            for name in self.by_name
        }

        requests_by_tile: dict[
            tuple[
                int,
                int,
            ],
            list[
                tuple[
                    str,
                    float,
                ]
            ],
        ] = {}

        for (
            name,
            agent,
        ) in self.by_name.items():
            action = actions.get(
                name,
                LOW_EXTRACT,
            )

            if (
                action
                == LOW_EXTRACT
            ):
                requested = (
                    self.cooperative_harvest_amount
                )

            elif (
                action
                == HIGH_EXTRACT
            ):
                requested = (
                    self.defective_harvest_amount
                )

            else:
                raise ValueError(
                    f"Unknown action {action}. "
                    "Use 0=low or 1=high extraction."
                )

            x, y = (
                agent.position
            )

            tile = (
                int(x),
                int(y),
            )

            requests_by_tile.setdefault(
                tile,
                [],
            ).append(
                (
                    name,
                    requested,
                )
            )

        for (
            x,
            y,
        ), requests in (
            requests_by_tile.items()
        ):
            available = float(
                self.resource[
                    y,
                    x,
                ]
            )

            total_requested = sum(
                amount
                for _, amount
                in requests
            )

            if (
                total_requested
                <= 0.0
            ):
                continue

            scale = min(
                1.0,
                (
                    available
                    / total_requested
                ),
            )

            total_harvest = 0.0

            for (
                name,
                requested,
            ) in requests:
                realized = (
                    requested
                    * scale
                )

                agent = (
                    self.by_name[
                        name
                    ]
                )

                agent.wealth += (
                    realized
                )

                agent.energy = min(
                    agent.energy_capacity,
                    (
                        agent.energy
                        + realized
                    ),
                )

                harvested[
                    name
                ] = realized

                total_harvest += (
                    realized
                )

            self.resource[
                y,
                x,
            ] -= total_harvest

        return harvested

    # -----------------------------------------------------------------
    # Welfare accounting
    # -----------------------------------------------------------------

    def _metabolize(
        self,
    ) -> None:
        """
        Consume stored energy and record unmet metabolic need.
        """

        for agent in (
            self.by_name.values()
        ):
            need = (
                self.metabolism_rate
            )

            if need <= 0.0:
                agent.metabolic_consumption = 0.0
                agent.metabolic_shortfall = 0.0
                agent.need_satisfaction = 1.0
                continue

            consumed = min(
                agent.energy,
                need,
            )

            shortfall = (
                need
                - consumed
            )

            agent.energy -= (
                consumed
            )

            agent.metabolic_consumption = (
                consumed
            )

            agent.metabolic_shortfall = (
                shortfall
            )

            agent.need_satisfaction = (
                consumed
                / need
            )

            agent.cumulative_shortfall += (
                shortfall
            )

            if shortfall > 1e-12:
                agent.deprivation_steps += 1

    # -----------------------------------------------------------------
    # Ecology
    # -----------------------------------------------------------------

    def _update_ecology(
        self,
    ) -> None:
        """
        Advance the post-harvest resource state.

        All ecological mathematics lives in `ecology.resource_step`.
        """

        self.resource = resource_step(
            resource=self.resource,
            capacity=self.capacity,
            regeneration_rate=(
                self.regeneration_map
            ),
            depletion_rate=(
                self.depletion_map
            ),
            coupling_rate=(
                self.coupling_rate
            ),
        )