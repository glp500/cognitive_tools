from __future__ import annotations

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from . import model


class EcoEnv(ParallelEnv):
    """PettingZoo wrapper around EcoModel."""

    metadata = {"name": "eco_commons_v0"}

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        n_agents: int = 4,
        max_steps: int = 1_000,
        regeneration_rate: float = 0.05,
        metabolism_rate: float = 0.05,
        initial_energy: float = 1.0,
        harvest_amount: float = 0.25,
        equilibrium_fraction: float = 1.0,
        coupling_rate: float = 0.0,
        initial_resource_fraction: float = 0.5,
        capacity_map: np.ndarray | None = None,
    ):
        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.max_steps = max_steps

        self.regeneration_rate = regeneration_rate
        self.metabolism_rate = metabolism_rate
        self.initial_energy = initial_energy
        self.harvest_amount = harvest_amount
        self.equilibrium_fraction = equilibrium_fraction
        self.coupling_rate = coupling_rate
        self.initial_resource_fraction = initial_resource_fraction
        self.capacity_map = (
            None
            if capacity_map is None
            else np.asarray(capacity_map, dtype=float).copy()
        )

        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]

        self.action_spaces = {
            agent: spaces.Discrete(6)
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
                    "self": spaces.Box(
                        low=0.0,
                        high=np.inf,
                        shape=(4,),
                        dtype=np.float32,
                    ),
                }
            )
            for agent in self.possible_agents
        }

        self.model: model.EcoModel | None = None

    def reset(self, seed: int | None = None, options=None):
        self.model = model.EcoModel(
            width=self.width,
            height=self.height,
            n_agents=self.n_agents,
            regeneration_rate=self.regeneration_rate,
            metabolism_rate=self.metabolism_rate,
            initial_energy=self.initial_energy,
            harvest_amount=self.harvest_amount,
            equilibrium_fraction=self.equilibrium_fraction,
            coupling_rate=self.coupling_rate,
            initial_resource_fraction=self.initial_resource_fraction,
            capacity=self.capacity_map,
            seed=seed,
        )

        self.agents = self.possible_agents.copy()

        if seed is not None:
            for i, agent in enumerate(self.possible_agents):
                self.action_spaces[agent].seed(seed + i)

        observations = {
            agent: self.observe(agent)
            for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        assert self.model is not None

        current_agents = self.agents.copy()

        wealth_before = {
            name: self.model.by_name[name].wealth
            for name in current_agents
        }

        self.model.actions = actions
        self.model.step()

        rewards = {
            name: self.model.by_name[name].wealth - wealth_before[name]
            for name in current_agents
        }

        terminations = {
            name: False
            for name in current_agents
        }

        finished = self.model.steps >= self.max_steps

        truncations = {
            name: finished
            for name in current_agents
        }

        infos = {
            name: {}
            for name in current_agents
        }

        if finished:
            self.agents = []
            observations = {}
        else:
            observations = {
                name: self.observe(name)
                for name in current_agents
            }

        return observations, rewards, terminations, truncations, infos

    def observe(self, name: str):
        assert self.model is not None

        agent = self.model.by_name[name]
        x, y = agent.position

        resources = np.zeros((3, 3), dtype=np.float32)

        for row, dy in enumerate((-1, 0, 1)):
            for col, dx in enumerate((-1, 0, 1)):
                xx = int(x + dx)
                yy = int(y + dy)

                if 0 <= xx < self.width and 0 <= yy < self.height:
                    resources[row, col] = self.model.resource[yy, xx]

        own_state = np.array(
            [
                x / max(self.width - 1, 1),
                y / max(self.height - 1, 1),
                agent.wealth,
                agent.energy,
            ],
            dtype=np.float32,
        )

        return {
            "resources": resources,
            "self": own_state,
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        pass

    def close(self):
        pass
