from __future__ import annotations

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from . import model


class EcoEnv(ParallelEnv):
    """
    PettingZoo wrapper for the stationary common-pool resource model.

    Action 0: low extraction / cooperation proxy
    Action 1: high extraction / defection proxy

    The environment exposes a richer observation for compatibility and
    diagnostics, but the current tabular Q-learner uses only local R/K.
    """

    metadata = {
        "name": "eco_commons_qlearning_v1",
    }

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        n_agents: int = 16,
        max_steps: int = 6_000,
        regeneration_rate: float | np.ndarray = 0.05,
        equilibrium_fraction: float | np.ndarray = 0.70,
        coupling_rate: float = 0.10,
        cooperative_harvest_amount: float = 0.002,
        defective_harvest_amount: float = 0.020,
        harvest_amount: float | None = None,
        metabolism_rate: float = 0.002,
        initial_energy: float = 1.0,
        energy_capacity: float = 1.0,
        initial_resource_fraction: float = 0.50,
        capacity_map: np.ndarray | None = None,
    ):
        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.max_steps = max_steps

        self.regeneration_rate = regeneration_rate
        self.equilibrium_fraction = equilibrium_fraction
        self.coupling_rate = coupling_rate

        self.cooperative_harvest_amount = cooperative_harvest_amount

        # Backward-compatible alias for older experiment scripts.
        self.defective_harvest_amount = (
            defective_harvest_amount
            if harvest_amount is None
            else harvest_amount
        )

        self.metabolism_rate = metabolism_rate
        self.initial_energy = initial_energy
        self.energy_capacity = energy_capacity
        self.initial_resource_fraction = initial_resource_fraction

        self.capacity_map = (
            None
            if capacity_map is None
            else np.asarray(
                capacity_map,
                dtype=float,
            ).copy()
        )

        self.possible_agents = [
            f"agent_{i}"
            for i in range(n_agents)
        ]

        self.action_spaces = {
            agent: spaces.Discrete(2)
            for agent in self.possible_agents
        }

        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "resources": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(3, 3),
                        dtype=np.float32,
                    ),
                    "capacities": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(3, 3),
                        dtype=np.float32,
                    ),
                    # [x_norm, y_norm, wealth, energy]
                    "self": spaces.Box(
                        low=0.0,
                        high=np.inf,
                        shape=(4,),
                        dtype=np.float32,
                    ),
                    # [R/K, wealth, energy]
                    "local": spaces.Box(
                        low=0.0,
                        high=np.inf,
                        shape=(3,),
                        dtype=np.float32,
                    ),
                }
            )
            for agent in self.possible_agents
        }

        self.model: model.EcoModel | None = None

    def reset(
        self,
        seed: int | None = None,
        options=None,
    ):
        self.model = model.EcoModel(
            width=self.width,
            height=self.height,
            n_agents=self.n_agents,
            regeneration_rate=self.regeneration_rate,
            equilibrium_fraction=self.equilibrium_fraction,
            coupling_rate=self.coupling_rate,
            cooperative_harvest_amount=self.cooperative_harvest_amount,
            defective_harvest_amount=self.defective_harvest_amount,
            metabolism_rate=self.metabolism_rate,
            initial_energy=self.initial_energy,
            energy_capacity=self.energy_capacity,
            initial_resource_fraction=self.initial_resource_fraction,
            capacity=self.capacity_map,
            seed=seed,
        )

        self.agents = self.possible_agents.copy()

        if seed is not None:
            for i, agent in enumerate(
                self.possible_agents
            ):
                self.action_spaces[agent].seed(
                    seed + i
                )

        observations = {
            agent: self.observe(agent)
            for agent in self.agents
        }

        infos = {
            agent: {}
            for agent in self.agents
        }

        return observations, infos

    def step(self, actions):
        assert self.model is not None

        current_agents = self.agents.copy()

        pre_state = {}

        for name in current_agents:
            agent = self.model.by_name[name]
            x, y = agent.position

            pre_state[name] = {
                "resource_before": float(
                    self.model.resource[y, x]
                ),
                "capacity": float(
                    self.model.capacity[y, x]
                ),
            }

        wealth_before = {
            name: self.model.by_name[name].wealth
            for name in current_agents
        }

        self.model.actions = actions
        self.model.step()

        rewards = {
            name: (
                self.model.by_name[name].wealth
                - wealth_before[name]
            )
            for name in current_agents
        }

        terminations = {
            name: False
            for name in current_agents
        }

        finished = (
            self.model.steps
            >= self.max_steps
        )

        truncations = {
            name: finished
            for name in current_agents
        }

        infos = {}

        for name in current_agents:
            resource_before = pre_state[name][
                "resource_before"
            ]
            capacity = pre_state[name][
                "capacity"
            ]
            agent = self.model.by_name[name]

            infos[name] = {
                "action_name": (
                    "low"
                    if actions[name]
                    == model.LOW_EXTRACT
                    else "high"
                ),
                "harvested": float(
                    rewards[name]
                ),
                "resource_before": resource_before,
                "capacity": capacity,
                "resource_fraction_before": (
                    resource_before
                    / max(capacity, 1e-12)
                ),
                "energy": float(agent.energy),
                "reserve_welfare": float(
                    agent.reserve_welfare
                ),
                "need_satisfaction": float(
                    agent.need_satisfaction
                ),
                "metabolic_shortfall": float(
                    agent.metabolic_shortfall
                ),
                "cumulative_shortfall": float(
                    agent.cumulative_shortfall
                ),
            }

        if finished:
            self.agents = []
            observations = {}
        else:
            observations = {
                name: self.observe(name)
                for name in current_agents
            }

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )

    def observe(self, name: str):
        assert self.model is not None

        agent = self.model.by_name[name]
        x, y = agent.position

        resources = np.zeros(
            (3, 3),
            dtype=np.float32,
        )

        capacities = np.zeros(
            (3, 3),
            dtype=np.float32,
        )

        for row, dy in enumerate((-1, 0, 1)):
            for col, dx in enumerate((-1, 0, 1)):
                xx = int(x + dx)
                yy = int(y + dy)

                if (
                    0 <= xx < self.width
                    and 0 <= yy < self.height
                ):
                    resources[row, col] = (
                        self.model.resource[yy, xx]
                    )
                    capacities[row, col] = (
                        self.model.capacity[yy, xx]
                    )

        local_resource = float(
            self.model.resource[y, x]
        )
        local_capacity = float(
            self.model.capacity[y, x]
        )

        resource_fraction = (
            local_resource
            / max(local_capacity, 1e-12)
        )

        own_state = np.array(
            [
                x / max(self.width - 1, 1),
                y / max(self.height - 1, 1),
                agent.wealth,
                agent.energy,
            ],
            dtype=np.float32,
        )

        local_state = np.array(
            [
                resource_fraction,
                agent.wealth,
                agent.energy,
            ],
            dtype=np.float32,
        )

        return {
            "resources": resources,
            "capacities": capacities,
            "self": own_state,
            "local": local_state,
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        pass

    def close(self):
        pass
