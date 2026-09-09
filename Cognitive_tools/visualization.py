from __future__ import annotations

import time

import numpy as np
import solara
from matplotlib.figure import Figure

from .env import EcoEnv


# ---------------------------------------------------------------------
# Environment and UI state
# ---------------------------------------------------------------------

env = EcoEnv(
    width=10,
    height=10,
    n_agents=4,
    max_steps=10_000,
    regeneration_rate=0.05,
    metabolism_rate=0.05,
    initial_energy=1.0,
)

env.reset(seed=42)

selected_agent = solara.reactive("agent_0")
frame = solara.reactive(0)
playing = solara.reactive(False)

# Playback settings.
steps_per_refresh = solara.reactive(5)
refresh_seconds = solara.reactive(0.10)

# Environment settings.
# These are applied when Reset / Apply is pressed.
regeneration_setting = solara.reactive(0.05)
metabolism_setting = solara.reactive(0.05)
initial_energy_setting = solara.reactive(1.0)
max_steps_setting = solara.reactive(10_000)
seed_setting = solara.reactive(42)


# ---------------------------------------------------------------------
# Simulation controls
# ---------------------------------------------------------------------

def random_actions():
    return {
        agent: env.action_space(agent).sample()
        for agent in env.agents
    }


def advance_steps(n_steps: int):
    """
    Advance up to n_steps.

    The UI is refreshed only after the batch finishes.
    """

    for _ in range(max(0, int(n_steps))):

        if not env.agents:
            break

        env.step(
            random_actions()
        )

    if not env.agents:
        playing.value = False

    frame.value += 1


def step_once():
    advance_steps(1)


def reset():
    """
    Apply the current environment settings
    and start a new simulation.
    """

    playing.value = False

    env.regeneration_rate = max(
        0.0,
        float(regeneration_setting.value),
    )

    env.metabolism_rate = max(
        0.0,
        float(metabolism_setting.value),
    )

    env.initial_energy = max(
        0.0,
        float(initial_energy_setting.value),
    )

    env.max_steps = max(
        1,
        int(max_steps_setting.value),
    )

    env.reset(
        seed=int(seed_setting.value)
    )

    selected_agent.value = (
        env.possible_agents[0]
    )

    frame.value += 1


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def gini(values):
    """Gini coefficient for non-negative values."""

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return 0.0

    total = values.sum()

    if total == 0:
        return 0.0

    values = np.sort(values)

    n = len(values)

    index = np.arange(
        1,
        n + 1,
    )

    return (
        2
        * np.sum(index * values)
        / (n * total)
        - (n + 1) / n
    )


# ---------------------------------------------------------------------
# TRUE ECOLOGY
# ---------------------------------------------------------------------

def true_ecology_figure():
    """
    Render the true ecological state held
    by the Mesa model.
    """

    model = env.model

    fig = Figure(
        figsize=(6, 6)
    )

    ax = fig.subplots()

    image = ax.imshow(
        model.resource,
        origin="upper",
        vmin=0.0,
        vmax=1.0,
    )

    for name, agent in model.by_name.items():

        x, y = agent.position

        if name == selected_agent.value:
            marker = "s"
            size = 150
        else:
            marker = "o"
            size = 90

        ax.scatter(
            x,
            y,
            marker=marker,
            s=size,
            edgecolors="black",
        )

        ax.text(
            x,
            y - 0.3,
            name.replace(
                "agent_",
                "",
            ),
            ha="center",
            va="center",
            fontsize=8,
        )

    ax.set_title(
        "True ecology"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xticks(
        range(model.width)
    )

    ax.set_yticks(
        range(model.height)
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Resource stock",
        fraction=0.046,
    )

    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------
# AGENT VIEW
# ---------------------------------------------------------------------

def agent_view_figure(agent_name):
    """
    Render exactly the observation returned
    by EcoEnv.observe().
    """

    observation = env.observe(
        agent_name
    )

    resources = observation[
        "resources"
    ]

    fig = Figure(
        figsize=(5, 5)
    )

    ax = fig.subplots()

    image = ax.imshow(
        resources,
        origin="upper",
        vmin=0.0,
        vmax=1.0,
    )

    # Observing agent is at the centre.
    ax.scatter(
        1,
        1,
        marker="^",
        s=160,
        edgecolors="black",
    )

    for row in range(
        resources.shape[0]
    ):
        for col in range(
            resources.shape[1]
        ):

            ax.text(
                col,
                row,
                f"{resources[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    ax.set_title(
        f"{agent_name} observation"
    )

    ax.set_xticks(
        range(resources.shape[1])
    )

    ax.set_yticks(
        range(resources.shape[0])
    )

    ax.set_xticklabels(
        ["-1", "0", "+1"]
    )

    ax.set_yticklabels(
        ["-1", "0", "+1"]
    )

    ax.set_xlabel(
        "relative x"
    )

    ax.set_ylabel(
        "relative y"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Observed resource",
        fraction=0.046,
    )

    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

@solara.component
def Page():

    # Subscribe to simulation updates.
    _ = frame.value

    model = env.model

    agent_name = (
        selected_agent.value
    )

    if (
        agent_name
        not in env.possible_agents
    ):
        agent_name = (
            env.possible_agents[0]
        )

        selected_agent.value = (
            agent_name
        )

    observation = env.observe(
        agent_name
    )

    agent = model.by_name[
        agent_name
    ]

    wealth_values = [
        item.wealth
        for item
        in model.by_name.values()
    ]

    energy_values = [
        item.energy
        for item
        in model.by_name.values()
    ]

    total_resource = float(
        model.resource.sum()
    )

    total_wealth = float(
        sum(wealth_values)
    )

    wealth_gini = gini(
        wealth_values
    )

    mean_energy = float(
        np.mean(energy_values)
    )

    deprived_agents = sum(
        value <= 0.0
        for value
        in energy_values
    )

    # -------------------------------------------------------------
    # Autoplay worker
    # -------------------------------------------------------------

    def autoplay():

        while (
            playing.value
            and env.agents
        ):

            advance_steps(
                steps_per_refresh.value
            )

            time.sleep(
                refresh_seconds.value
            )

        if not env.agents:
            playing.value = False

    # The simulation loop runs outside
    # Solara's rendering thread.
    solara.use_thread(
        autoplay,
        dependencies=[
            playing.value,
            steps_per_refresh.value,
            refresh_seconds.value,
        ],
    )

    # -------------------------------------------------------------
    # Header / playback
    # -------------------------------------------------------------

    solara.Markdown(
        "# ECO-MARL"
    )

    with solara.Row():

        if playing.value:

            solara.Button(
                "Pause",
                on_click=lambda:
                    playing.set(False),
            )

        else:

            solara.Button(
                "Play",
                on_click=lambda:
                    playing.set(True),
                disabled=(
                    not bool(env.agents)
                ),
            )

        solara.Button(
            "Step",
            on_click=step_once,
            disabled=(
                playing.value
                or not bool(env.agents)
            ),
        )

        solara.Button(
            "Run 10",
            on_click=lambda:
                advance_steps(10),
            disabled=(
                playing.value
                or not bool(env.agents)
            ),
        )

        solara.Button(
            "Run 100",
            on_click=lambda:
                advance_steps(100),
            disabled=(
                playing.value
                or not bool(env.agents)
            ),
        )

        solara.Button(
            "Reset / Apply",
            on_click=reset,
            disabled=playing.value,
        )

    # -------------------------------------------------------------
    # Playback settings
    # -------------------------------------------------------------

    with solara.Row():

        solara.Select(
            label="Selected agent",
            values=env.possible_agents,
            value=selected_agent,
        )

        solara.Select(
            label="Steps per refresh",
            values=[
                1,
                2,
                5,
                10,
                25,
                50,
            ],
            value=steps_per_refresh,
            disabled=playing.value,
        )

        solara.Select(
            label="Refresh delay (seconds)",
            values=[
                0.05,
                0.10,
                0.25,
                0.50,
                1.00,
            ],
            value=refresh_seconds,
            disabled=playing.value,
        )

    status = (
        "Running"
        if playing.value
        else "Paused"
    )

    if not env.agents:
        status = "Finished"

    solara.Markdown(
        f"**Status:** {status}  |  "
        f"**Step:** {model.steps:,} "
        f"/ {env.max_steps:,}  |  "
        f"**Agents:** {len(env.agents)}"
    )

    # -------------------------------------------------------------
    # Environment settings
    # -------------------------------------------------------------

    with solara.Card(
        "Environment settings "
        "(apply on Reset)"
    ):

        with solara.Row():

            solara.InputFloat(
                "Regeneration rate",
                value=(
                    regeneration_setting
                ),
                disabled=playing.value,
            )

            solara.InputFloat(
                "Metabolism rate",
                value=(
                    metabolism_setting
                ),
                disabled=playing.value,
            )

            solara.InputFloat(
                "Initial energy",
                value=(
                    initial_energy_setting
                ),
                disabled=playing.value,
            )

        with solara.Row():

            solara.InputInt(
                "Maximum steps",
                value=(
                    max_steps_setting
                ),
                disabled=playing.value,
            )

            solara.InputInt(
                "Seed",
                value=seed_setting,
                disabled=playing.value,
            )

    # -------------------------------------------------------------
    # Main two-panel view
    # -------------------------------------------------------------

    with solara.Row():

        with solara.Card(
            "True ecology"
        ):

            solara.FigureMatplotlib(
                true_ecology_figure()
            )

        with solara.Card(
            "Agent view"
        ):

            solara.FigureMatplotlib(
                agent_view_figure(
                    agent_name
                )
            )

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------

    with solara.Row():

        with solara.Card(
            "System metrics"
        ):

            solara.Markdown(
                f"""
**Regeneration rate:** \
{model.regeneration_rate:.3f}

**Metabolism rate:** \
{model.metabolism_rate:.3f}

**Total resource:** \
{total_resource:.2f}

**Total wealth:** \
{total_wealth:.2f}

**Wealth Gini:** \
{wealth_gini:.3f}

**Mean energy:** \
{mean_energy:.3f}

**Agents at zero energy:** \
{deprived_agents}
"""
            )

        with solara.Card(
            "Agent state"
        ):

            solara.Markdown(
                f"""
**Agent:** `{agent_name}`

**Position:** \
`{tuple(agent.position)}`

**Wealth:** \
{agent.wealth:.3f}

**Energy:** \
{agent.energy:.3f}
"""
            )

            solara.Markdown(
                "**Exact PettingZoo "
                "`self` observation**  "
                "`[x_norm, y_norm, "
                "wealth, energy]`:"
            )

            solara.Markdown(
                f"`{observation['self'].tolist()}`"
            )