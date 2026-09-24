from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Cognitive_tools import EcoEnv
from Cognitive_tools.ecology import (
    ABUNDANCE_LEVELS,
    PATTERN_FAMILIES,
    make_capacity_map,
)

RESULTS_DIR = Path("results")
ANIMATIONS_DIR = RESULTS_DIR / "agent_animations"
FRAMES_DIR = RESULTS_DIR / "agent_frames"


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def gini(values) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    total = values.sum()

    if len(values) == 0 or total <= 0:
        return 0.0

    n = len(values)
    index = np.arange(1, n + 1)

    value = (
        2 * np.sum(index * values) / (n * total)
        - (n + 1) / n
    )

    if abs(value) < 1e-12:
        return 0.0

    return float(max(0.0, value))


def measure(env: EcoEnv, time: int) -> dict:
    agents = list(env.model.by_name.values())

    wealth = [
        agent.wealth
        for agent in agents
    ]

    energy = [
        agent.energy
        for agent in agents
    ]

    return {
        "time": time,
        "wealth_gini": gini(wealth),
        "mean_energy": float(np.mean(energy)),
        "mean_wealth": float(np.mean(wealth)),
        "total_resource": float(env.model.resource.sum()),
    }


# ---------------------------------------------------------------------
# Environment definitions
# ---------------------------------------------------------------------

def build_scenarios() -> list[dict]:
    scenarios = []

    for abundance, mean_capacity in ABUNDANCE_LEVELS.items():
        for family, family_spec in PATTERN_FAMILIES.items():
            scenarios.append(
                {
                    "scenario": f"{family}_{abundance}",
                    "family": family,
                    "family_label": family_spec["label"],
                    "abundance": abundance,
                    "mean_capacity": float(mean_capacity),
                }
            )

    return scenarios


def make_environment_capacity(
    spec: dict,
    width: int,
    height: int,
    seed: int,
    replicate: int,
) -> np.ndarray:
    """
    High- and low-abundance versions of a pattern family use
    the same underlying spatial geometry within a replicate.
    """

    family_index = list(PATTERN_FAMILIES).index(
        spec["family"]
    )

    map_seed = (
        seed
        + replicate
        + 1000 * family_index
    )

    return make_capacity_map(
        family=spec["family"],
        mean_capacity=spec["mean_capacity"],
        width=width,
        height=height,
        seed=map_seed,
    )


def make_env(
    spec: dict,
    population: int,
    steps: int,
    replicate: int,
    args,
) -> EcoEnv:
    capacity = make_environment_capacity(
        spec,
        width=args.width,
        height=args.height,
        seed=args.seed,
        replicate=replicate,
    )

    env = EcoEnv(
        width=args.width,
        height=args.height,
        n_agents=population,
        max_steps=steps,
        regeneration_rate=args.recovery_rate,
        metabolism_rate=args.metabolism,
        initial_energy=args.initial_energy,
        harvest_amount=args.harvest,
        equilibrium_fraction=args.equilibrium_fraction,
        coupling_rate=args.coupling,
        initial_resource_fraction=args.initial_resource_fraction,
        capacity_map=capacity,
    )

    # The same replicate seed is reused across environments.
    #
    # Therefore, for a fixed population size and replicate,
    # agents begin at the same randomly generated positions in
    # every environmental treatment.
    #
    # Different replicates use different starting positions.
    env.reset(
        seed=args.seed + 100_000 + replicate
    )

    return env


def random_actions(env: EcoEnv) -> dict[str, int]:
    return {
        agent: env.action_space(agent).sample()
        for agent in env.agents
    }


# ---------------------------------------------------------------------
# Initial-position screenshots
# ---------------------------------------------------------------------

def initial_position_records(
    env: EcoEnv,
    spec: dict,
    population: int,
    replicate: int,
) -> list[dict]:
    rows = []

    for name, agent in env.model.by_name.items():
        x, y = agent.position

        rows.append(
            {
                "scenario": spec["scenario"],
                "family": spec["family"],
                "abundance": spec["abundance"],
                "population": population,
                "replicate": replicate,
                "agent": name,
                "x": int(x),
                "y": int(y),
            }
        )

    return rows


def save_initialization_snapshot(
    env: EcoEnv,
    spec: dict,
    population: int,
    replicate: int,
) -> None:
    """
    Save the state before any movement, harvesting, or metabolism.

    One marker is drawn per occupied tile. The number written at
    that tile is the number of agents starting there. Exact agent
    IDs and coordinates are also written to the CSV file created
    by the experiment.
    """

    fig, ax = plt.subplots(
        figsize=(7.5, 6.5),
        layout="constrained",
    )

    image = ax.imshow(
        env.model.resource,
        origin="upper",
        vmin=0.0,
        vmax=1.0,
    )

    occupancy: dict[tuple[int, int], list[str]] = {}

    for name, agent in env.model.by_name.items():
        x, y = agent.position
        key = (int(x), int(y))
        occupancy.setdefault(key, []).append(name)

    for (x, y), names in occupancy.items():
        count = len(names)

        ax.scatter(
            x,
            y,
            s=75 + 35 * (count - 1),
            edgecolors="black",
            linewidths=0.8,
        )

        ax.text(
            x,
            y,
            str(count),
            ha="center",
            va="center",
            fontsize=8,
        )

    ax.set_title(
        f"Initial agent positions\n"
        f"{spec['scenario']} | "
        f"N={population} | replicate={replicate}"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xticks(
        range(env.width)
    )

    ax.set_yticks(
        range(env.height)
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Initial resource stock",
    )

    ax.text(
        0.5,
        -0.11,
        "Marker number = agents sharing that starting tile",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    path = (
        RESULTS_DIR
        / (
            f"initial_{spec['scenario']}"
            f"_N{population}"
            f"_rep{replicate}.png"
        )
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved {path}"
    )


def save_initializations_only(
    scenarios: list[dict],
    args,
) -> None:
    initialization_rows = []

    replicate = args.initialization_replicate

    print(
        "Saving initialization screenshots "
        f"for replicate {replicate}..."
    )

    for spec in scenarios:
        for population in args.populations:
            env = make_env(
                spec,
                population,
                steps=1,
                replicate=replicate,
                args=args,
            )

            save_initialization_snapshot(
                env,
                spec,
                population,
                replicate,
            )

            initialization_rows.extend(
                initial_position_records(
                    env,
                    spec,
                    population,
                    replicate,
                )
            )

    write_csv(
        RESULTS_DIR / "initial_agent_positions.csv",
        initialization_rows,
    )

    print(
        "\nInitialization export complete."
    )


# ---------------------------------------------------------------------
# Main experimental runs
# ---------------------------------------------------------------------

def run_condition(
    spec: dict,
    population: int,
    replicate: int,
    args,
    initialization_rows: list[dict],
) -> dict:
    env = make_env(
        spec,
        population,
        args.steps,
        replicate,
        args,
    )

    # Save initialization before the first model step.
    if (
        args.save_initializations
        and replicate == args.initialization_replicate
    ):
        save_initialization_snapshot(
            env,
            spec,
            population,
            replicate,
        )

        initialization_rows.extend(
            initial_position_records(
                env,
                spec,
                population,
                replicate,
            )
        )

    window = min(
        max(1, args.summary_window),
        args.steps,
    )

    recent_gini = []
    recent_welfare = []

    final = measure(
        env,
        0,
    )

    for time in range(
        1,
        args.steps + 1,
    ):
        env.step(
            random_actions(env)
        )

        final = measure(
            env,
            time,
        )

        if time > args.steps - window:
            recent_gini.append(
                final["wealth_gini"]
            )

            recent_welfare.append(
                final["mean_energy"]
            )

    return {
        "replicate": replicate,
        "scenario": spec["scenario"],
        "family": spec["family"],
        "family_label": spec["family_label"],
        "abundance": spec["abundance"],
        "mean_capacity": spec["mean_capacity"],
        "population": population,
        "window_mean_wealth_gini": float(
            np.mean(recent_gini)
        ),
        "window_mean_welfare": float(
            np.mean(recent_welfare)
        ),
        "final_wealth_gini": final[
            "wealth_gini"
        ],
        "final_mean_energy": final[
            "mean_energy"
        ],
    }


def aggregate(
    rows: list[dict],
) -> list[dict]:
    groups = {}

    for row in rows:
        key = (
            row["scenario"],
            row["family"],
            row["family_label"],
            row["abundance"],
            row["mean_capacity"],
            row["population"],
        )

        groups.setdefault(
            key,
            [],
        ).append(row)

    output = []

    for key, group in groups.items():
        (
            scenario,
            family,
            label,
            abundance,
            mean_capacity,
            population,
        ) = key

        gini_values = np.array(
            [
                row["window_mean_wealth_gini"]
                for row in group
            ],
            dtype=float,
        )

        welfare_values = np.array(
            [
                row["window_mean_welfare"]
                for row in group
            ],
            dtype=float,
        )

        output.append(
            {
                "scenario": scenario,
                "family": family,
                "family_label": label,
                "abundance": abundance,
                "mean_capacity": mean_capacity,
                "population": population,
                "replicates": len(group),
                "wealth_gini_mean": float(
                    gini_values.mean()
                ),
                "wealth_gini_sd": (
                    float(
                        gini_values.std(
                            ddof=1
                        )
                    )
                    if len(group) > 1
                    else 0.0
                ),
                "welfare_mean": float(
                    welfare_values.mean()
                ),
                "welfare_sd": (
                    float(
                        welfare_values.std(
                            ddof=1
                        )
                    )
                    if len(group) > 1
                    else 0.0
                ),
            }
        )

    return sorted(
        output,
        key=lambda row: (
            row["abundance"],
            row["family"],
            row["population"],
        ),
    )


# ---------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------

def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Saved {path}"
    )


# ---------------------------------------------------------------------
# Static figures
# ---------------------------------------------------------------------

def plot_vs_population(
    rows: list[dict],
    mean_key: str,
    sd_key: str,
    ylabel: str,
    filename: str,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
        sharey=True,
        layout="constrained",
    )

    for ax, abundance in zip(
        axes,
        ["high", "low"],
    ):
        for family, family_spec in (
            PATTERN_FAMILIES.items()
        ):
            subset = sorted(
                [
                    row
                    for row in rows
                    if (
                        row["abundance"] == abundance
                        and row["family"] == family
                    )
                ],
                key=lambda row: row["population"],
            )

            ax.errorbar(
                [
                    row["population"]
                    for row in subset
                ],
                [
                    row[mean_key]
                    for row in subset
                ],
                yerr=[
                    row[sd_key]
                    for row in subset
                ],
                marker="o",
                capsize=3,
                label=family_spec["label"],
            )

        ax.set_title(
            f"{abundance.capitalize()} abundance"
        )

        ax.set_xlabel(
            "Population size"
        )

        ax.grid(
            alpha=0.25
        )

    axes[0].set_ylabel(
        ylabel
    )

    axes[1].legend(
        title="Pattern family",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    path = (
        RESULTS_DIR
        / filename
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved {path}"
    )


def plot_heatmap(
    rows: list[dict],
    metric: str,
    title: str,
    label: str,
    filename: str,
) -> None:
    scenarios = [
        spec["scenario"]
        for spec in build_scenarios()
    ]

    populations = sorted(
        {
            row["population"]
            for row in rows
        }
    )

    matrix = np.full(
        (
            len(scenarios),
            len(populations),
        ),
        np.nan,
    )

    for row in rows:
        i = scenarios.index(
            row["scenario"]
        )

        j = populations.index(
            row["population"]
        )

        matrix[i, j] = row[
            metric
        ]

    fig, ax = plt.subplots(
        figsize=(8.5, 8),
        layout="constrained",
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
    )

    ax.set_xticks(
        range(
            len(populations)
        )
    )

    ax.set_xticklabels(
        populations
    )

    ax.set_yticks(
        range(
            len(scenarios)
        )
    )

    ax.set_yticklabels(
        scenarios
    )

    ax.set_xlabel(
        "Population size"
    )

    ax.set_ylabel(
        "Environment"
    )

    ax.set_title(
        title
    )

    for i in range(
        matrix.shape[0]
    ):
        for j in range(
            matrix.shape[1]
        ):
            if np.isfinite(
                matrix[i, j]
            ):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    fig.colorbar(
        image,
        ax=ax,
        label=label,
    )

    path = (
        RESULTS_DIR
        / filename
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved {path}"
    )


def plot_gini_welfare(
    rows: list[dict],
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
        layout="constrained",
    )

    for ax, abundance in zip(
        axes,
        ["high", "low"],
    ):
        for family, family_spec in (
            PATTERN_FAMILIES.items()
        ):
            subset = sorted(
                [
                    row
                    for row in rows
                    if (
                        row["abundance"] == abundance
                        and row["family"] == family
                    )
                ],
                key=lambda row: row["population"],
            )

            x = [
                row["wealth_gini_mean"]
                for row in subset
            ]

            y = [
                row["welfare_mean"]
                for row in subset
            ]

            ax.plot(
                x,
                y,
                marker="o",
                label=family_spec["label"],
            )

            for row in subset:
                ax.annotate(
                    f"N={row['population']}",
                    (
                        row["wealth_gini_mean"],
                        row["welfare_mean"],
                    ),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )

        ax.set_title(
            f"{abundance.capitalize()} abundance"
        )

        ax.set_xlabel(
            "Wealth Gini"
        )

        ax.grid(
            alpha=0.25
        )

    axes[0].set_ylabel(
        "Mean energy (welfare proxy)"
    )

    axes[1].legend(
        title="Pattern family",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    path = (
        RESULTS_DIR
        / "05_gini_welfare_tradeoff.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved {path}"
    )


# ---------------------------------------------------------------------
# GIF support
# ---------------------------------------------------------------------

def snapshot(
    env: EcoEnv,
) -> dict:
    return {
        "resource": env.model.resource.copy(),
        "positions": {
            name: agent.position.copy()
            for name, agent
            in env.model.by_name.items()
        },
    }


def collect_gif_run(
    spec: dict,
    population: int,
    args,
):
    env = make_env(
        spec,
        population,
        args.gif_steps,
        0,
        args,
    )

    metrics = [
        measure(
            env,
            0,
        )
    ]

    states = [
        snapshot(env)
    ]

    for time in range(
        1,
        args.gif_steps + 1,
    ):
        env.step(
            random_actions(env)
        )

        metrics.append(
            measure(
                env,
                time,
            )
        )

        states.append(
            snapshot(env)
        )

    return (
        env,
        metrics,
        states,
    )


def render_gif_frame(
    state: dict,
    metrics: list[dict],
    index: int,
    scenario: str,
    population: int,
    metric_key: str,
    metric_label: str,
) -> np.ndarray:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.5, 4.7),
        layout="constrained",
    )

    image = axes[0].imshow(
        state["resource"],
        origin="upper",
        vmin=0.0,
        vmax=1.0,
    )

    for position in (
        state["positions"].values()
    ):
        axes[0].scatter(
            position[0],
            position[1],
            s=28,
            edgecolors="black",
            linewidths=0.5,
        )

    axes[0].set_title(
        "Resource field + random agents"
    )

    axes[0].set_xticks([])
    axes[0].set_yticks([])

    fig.colorbar(
        image,
        ax=axes[0],
        label="Resource stock",
        fraction=0.046,
    )

    history = metrics[
        : index + 1
    ]

    x = [
        row["time"]
        for row in history
    ]

    y = [
        row[metric_key]
        for row in history
    ]

    axes[1].plot(
        x,
        y,
    )

    axes[1].scatter(
        [x[-1]],
        [y[-1]],
        s=45,
    )

    axes[1].set_xlabel(
        "Time step"
    )

    axes[1].set_ylabel(
        metric_label
    )

    axes[1].set_title(
        f"{metric_label} through time"
    )

    axes[1].set_xlim(
        0,
        metrics[-1]["time"],
    )

    axes[1].grid(
        alpha=0.25
    )

    if metric_key == "wealth_gini":
        axes[1].set_ylim(
            0,
            1,
        )
    else:
        maximum = max(
            [
                row[metric_key]
                for row in metrics
            ]
            + [1.0]
        )

        axes[1].set_ylim(
            0,
            maximum * 1.08,
        )

    fig.suptitle(
        f"{scenario} | "
        f"N={population} | "
        f"step={metrics[index]['time']}",
        fontsize=14,
    )

    fig.canvas.draw()

    width, height = (
        fig.canvas.get_width_height()
    )

    frame = np.frombuffer(
        fig.canvas.buffer_rgba(),
        dtype=np.uint8,
    )

    frame = frame.reshape(
        height,
        width,
        4,
    )[..., :3].copy()

    plt.close(fig)

    return frame


def save_gif(
    states: list[dict],
    metrics: list[dict],
    scenario: str,
    population: int,
    metric_key: str,
    metric_label: str,
    args,
) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "Install GIF dependencies with: "
            "pip install imageio pillow"
        ) from exc

    name = (
        f"{scenario}"
        f"_N{population}"
        f"_{metric_key}"
    )

    frame_dir = (
        FRAMES_DIR
        / name
    )

    if args.save_frames:
        frame_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    indices = list(
        range(
            0,
            len(metrics),
            max(
                1,
                args.frame_every,
            ),
        )
    )

    if (
        indices[-1]
        != len(metrics) - 1
    ):
        indices.append(
            len(metrics) - 1
        )

    frames = []

    for frame_number, index in enumerate(
        indices
    ):
        frame = render_gif_frame(
            states[index],
            metrics,
            index,
            scenario,
            population,
            metric_key,
            metric_label,
        )

        frames.append(
            frame
        )

        if args.save_frames:
            imageio.imwrite(
                frame_dir
                / f"frame_{frame_number:04d}.png",
                frame,
            )

    ANIMATIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        ANIMATIONS_DIR
        / f"{name}.gif"
    )

    imageio.mimsave(
        path,
        frames,
        fps=args.fps,
    )

    print(
        f"Saved {path}"
    )


def make_gifs(
    scenarios: list[dict],
    args,
) -> None:
    if args.make_gifs == "all":
        selected = scenarios
    else:
        selected = [
            spec
            for spec in scenarios
            if spec["scenario"]
            == args.make_gifs
        ]

    if not selected:
        valid = ", ".join(
            spec["scenario"]
            for spec in scenarios
        )

        raise ValueError(
            "Unknown scenario. "
            f"Valid values: {valid}"
        )

    for spec in selected:
        _, metrics, states = (
            collect_gif_run(
                spec,
                args.gif_population,
                args,
            )
        )

        save_gif(
            states,
            metrics,
            spec["scenario"],
            args.gif_population,
            "wealth_gini",
            "Wealth Gini",
            args,
        )

        save_gif(
            states,
            metrics,
            spec["scenario"],
            args.gif_population,
            "mean_energy",
            "Mean energy (welfare proxy)",
            args,
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Random-action agents across "
            "ecological environments."
        )
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--populations",
        type=int,
        nargs="+",
        default=[
            4,
            8,
            16,
            32,
        ],
    )

    parser.add_argument(
        "--replicates",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--summary-window",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--equilibrium-fraction",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--recovery-rate",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--coupling",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--initial-resource-fraction",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--metabolism",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--initial-energy",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--harvest",
        type=float,
        default=0.25,
    )

    # Initialization export.
    parser.add_argument(
        "--save-initializations",
        action="store_true",
        help=(
            "Save starting-position figures "
            "before agents move."
        ),
    )

    parser.add_argument(
        "--initialization-replicate",
        type=int,
        default=0,
        help=(
            "Which replicate's initial "
            "positions to save."
        ),
    )

    parser.add_argument(
        "--initialization-only",
        action="store_true",
        help=(
            "Create initialization figures "
            "and exit without running the "
            "full experiment."
        ),
    )

    # GIF export.
    parser.add_argument(
        "--make-gifs",
        default="",
        help=(
            "Scenario name such as "
            "patchy_low, centralized_high, "
            "or all."
        ),
    )

    parser.add_argument(
        "--gif-population",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--gif-steps",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--frame-every",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--save-frames",
        action="store_true",
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenarios = build_scenarios()

    # Fast mode: only save starting positions.
    if args.initialization_only:
        save_initializations_only(
            scenarios,
            args,
        )
        return

    rows = []
    initialization_rows = []

    total = (
        len(scenarios)
        * len(args.populations)
        * args.replicates
    )

    completed = 0

    print(
        f"Running {total} "
        "agent-environment runs..."
    )

    for replicate in range(
        args.replicates
    ):
        for spec in scenarios:
            for population in (
                args.populations
            ):
                rows.append(
                    run_condition(
                        spec,
                        population,
                        replicate,
                        args,
                        initialization_rows,
                    )
                )

                completed += 1

                print(
                    f"[{completed}/{total}] "
                    f"{spec['scenario']} "
                    f"N={population} "
                    f"replicate={replicate}"
                )

    summary = aggregate(
        rows
    )

    write_csv(
        RESULTS_DIR
        / "agent_environment_runs.csv",
        rows,
    )

    write_csv(
        RESULTS_DIR
        / "agent_environment_summary.csv",
        summary,
    )

    if initialization_rows:
        write_csv(
            RESULTS_DIR
            / "initial_agent_positions.csv",
            initialization_rows,
        )

    plot_vs_population(
        summary,
        "wealth_gini_mean",
        "wealth_gini_sd",
        "Wealth Gini",
        "01_wealth_gini_vs_population.png",
    )

    plot_vs_population(
        summary,
        "welfare_mean",
        "welfare_sd",
        "Mean energy (welfare proxy)",
        "02_welfare_vs_population.png",
    )

    plot_heatmap(
        summary,
        "wealth_gini_mean",
        (
            "Wealth inequality by "
            "environment and population"
        ),
        "Mean wealth Gini",
        "03_wealth_gini_heatmap.png",
    )

    plot_heatmap(
        summary,
        "welfare_mean",
        (
            "Agent welfare by "
            "environment and population"
        ),
        "Mean energy (welfare proxy)",
        "04_welfare_heatmap.png",
    )

    plot_gini_welfare(
        summary
    )

    if args.make_gifs:
        make_gifs(
            scenarios,
            args,
        )

    print(
        "\nExperiment complete."
    )

    print(
        f"Results: "
        f"{RESULTS_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
