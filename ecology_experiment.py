from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Cognitive_tools.ecology import EcologyModel


RESULTS_DIR = Path("results")

# ---------------------------------------------------------------------
# Archetype environments
# ---------------------------------------------------------------------
#
# Mean carrying capacity is held fixed here so that the A-H comparison
# isolates three dimensions only:
#     abundance          -> equilibrium_fraction
#     spatial inequality -> heterogeneity
#     recovery speed     -> recovery_rate
#
# Mean capacity is varied separately in the response-surface experiment
# later in this script.

ARCHETYPES = {
    "A": {
        "abundance": "High",
        "inequality": "Low",
        "recovery": "Fast",
        "mean_capacity": 0.60,
        "equilibrium_fraction": 0.85,
        "recovery_rate": 0.10,
        "heterogeneity": 0.00,
    },
    "B": {
        "abundance": "High",
        "inequality": "High",
        "recovery": "Fast",
        "mean_capacity": 0.60,
        "equilibrium_fraction": 0.85,
        "recovery_rate": 0.10,
        "heterogeneity": 0.80,
    },
    "C": {
        "abundance": "Low",
        "inequality": "Low",
        "recovery": "Fast",
        "mean_capacity": 0.60,
        "equilibrium_fraction": 0.35,
        "recovery_rate": 0.10,
        "heterogeneity": 0.00,
    },
    "D": {
        "abundance": "Low",
        "inequality": "High",
        "recovery": "Fast",
        "mean_capacity": 0.60,
        "equilibrium_fraction": 0.35,
        "recovery_rate": 0.10,
        "heterogeneity": 0.80,
    },
    "E": {
        "abundance": "High",
        "inequality": "Low",
        "recovery": "Slow",
        "mean_capacity": 0.60,
        "equilibrium_fraction": 0.85,
        "recovery_rate": 0.02,
        "heterogeneity": 0.00,
    },
    "F": {
        "abundance": "Low",
        "inequality": "High",
        "recovery": "Slow",
        "mean_capacity": 0.60,
        "equilibrium_fraction": 0.35,
        "recovery_rate": 0.02,
        "heterogeneity": 0.80,
    },
    # The original A-F set is not a complete 2 x 2 x 2 factorial.
    # G and H add the two missing combinations.
    "G": {
        "abundance": "High",
        "inequality": "High",
        "recovery": "Slow",
        "mean_capacity": 0.60,
        "equilibrium_fraction": 0.85,
        "recovery_rate": 0.02,
        "heterogeneity": 0.80,
    },
    "H": {
        "abundance": "Low",
        "inequality": "Low",
        "recovery": "Slow",
        "mean_capacity": 0.60,
        "equilibrium_fraction": 0.35,
        "recovery_rate": 0.02,
        "heterogeneity": 0.00,
    },
}


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def gini(field: np.ndarray) -> float:
    values = np.sort(
        np.asarray(
            field,
            dtype=float,
        ).ravel()
    )

    total = values.sum()

    if total <= 0:
        return 0.0

    n = len(values)
    index = np.arange(1, n + 1)

    value = float(
        2 * np.sum(index * values) / (n * total)
        - (n + 1) / n
    )

    # Remove tiny negative values caused only by floating-point rounding.
    return max(0.0, value)


def resource_cv(field: np.ndarray) -> float:
    mean = float(np.mean(field))

    if mean <= 0:
        return 0.0

    return float(np.std(field) / mean)


def scarcity_fraction(
    model: EcologyModel,
    threshold: float = 0.20,
) -> float:
    scarce = model.resource < threshold * model.capacity
    return float(np.mean(scarce))


def depletion_from_equilibrium(
    recovery_rate: float,
    equilibrium_fraction: float,
) -> float:
    """Choose depletion so an uncoupled tile has equilibrium R*/K = q."""

    return recovery_rate * (1.0 - equilibrium_fraction)


def build_model(
    *,
    mean_capacity: float,
    equilibrium_fraction: float,
    recovery_rate: float,
    heterogeneity: float,
    coupling_rate: float,
    initial_fraction: float,
    seed: int,
    distribution: str = "patchy",
) -> EcologyModel:
    depletion_rate = depletion_from_equilibrium(
        recovery_rate,
        equilibrium_fraction,
    )

    return EcologyModel(
        width=10,
        height=10,
        distribution=distribution,
        mean_capacity=mean_capacity,
        heterogeneity=heterogeneity,
        initial_fraction=initial_fraction,
        regeneration_rate=recovery_rate,
        depletion_rate=depletion_rate,
        coupling_rate=coupling_rate,
        seed=seed,
    )


def measure(
    model: EcologyModel,
    *,
    time: int,
) -> dict[str, float | int]:
    return {
        "time": time,
        "mean_resource": float(np.mean(model.resource)),
        "resource_fraction": float(
            np.mean(model.resource / model.capacity)
        ),
        "resource_gini": gini(model.resource),
        "capacity_gini": gini(model.capacity),
        "resource_cv": resource_cv(model.resource),
        "scarcity_fraction": scarcity_fraction(model),
    }


def run_trajectory(
    spec: dict,
    *,
    steps: int,
    coupling_rate: float,
    initial_fraction: float,
    seed: int,
) -> tuple[EcologyModel, list[dict]]:
    model = build_model(
        mean_capacity=spec["mean_capacity"],
        equilibrium_fraction=spec["equilibrium_fraction"],
        recovery_rate=spec["recovery_rate"],
        heterogeneity=spec["heterogeneity"],
        coupling_rate=coupling_rate,
        initial_fraction=initial_fraction,
        seed=seed,
    )

    records = [measure(model, time=0)]

    for time in range(1, steps + 1):
        model.step()
        records.append(measure(model, time=time))

    return model, records


def add_environment_metadata(
    environment_id: str,
    spec: dict,
    row: dict,
) -> dict:
    return {
        "environment": environment_id,
        "abundance": spec["abundance"],
        "inequality": spec["inequality"],
        "recovery": spec["recovery"],
        "mean_capacity": spec["mean_capacity"],
        "equilibrium_fraction": spec["equilibrium_fraction"],
        "recovery_rate": spec["recovery_rate"],
        "heterogeneity": spec["heterogeneity"],
        **row,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {path}")


# ---------------------------------------------------------------------
# Archetype experiment
# ---------------------------------------------------------------------


def run_archetypes(
    *,
    steps: int,
    coupling_rate: float,
    initial_fraction: float,
    seed: int,
) -> tuple[dict[str, EcologyModel], list[dict], list[dict]]:
    models = {}
    timeseries_rows = []
    summary_rows = []

    for environment_id, spec in ARCHETYPES.items():
        model, records = run_trajectory(
            spec,
            steps=steps,
            coupling_rate=coupling_rate,
            initial_fraction=initial_fraction,
            seed=seed,
        )

        models[environment_id] = model

        for row in records:
            timeseries_rows.append(
                add_environment_metadata(environment_id, spec, row)
            )

        final = records[-1]
        summary_rows.append(
            add_environment_metadata(environment_id, spec, final)
        )

    return models, timeseries_rows, summary_rows


def plot_archetype_maps(
    models: dict[str, EcologyModel],
    *,
    field_name: str,
    filename: str,
    title: str,
) -> None:
    """
    Plot the eight A-H archetypes.

    A dedicated fifth GridSpec column is reserved for the colorbar so it
    cannot overlap the H panel or any subplot title.
    """

    fig = plt.figure(
        figsize=(16, 8),
        layout="constrained",
    )

    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=[1, 1, 1, 1, 0.055],
    )

    axes = np.empty((2, 4), dtype=object)
    image = None

    for index, environment_id in enumerate(ARCHETYPES):
        row = index // 4
        col = index % 4

        ax = fig.add_subplot(grid[row, col])
        axes[row, col] = ax

        model = models[environment_id]
        spec = ARCHETYPES[environment_id]
        field = getattr(model, field_name)

        image = ax.imshow(
            field,
            origin="upper",
            vmin=0.0,
            vmax=1.0,
        )

        ax.set_title(
            f"{environment_id}: {spec['abundance']} abundance\n"
            f"{spec['inequality']} inequality, "
            f"{spec['recovery']} recovery",
            fontsize=10,
            pad=8,
        )

        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        title,
        fontsize=16,
    )

    if image is not None:
        colorbar_ax = fig.add_subplot(grid[:, 4])

        fig.colorbar(
            image,
            cax=colorbar_ax,
            label=(
                "Carrying capacity"
                if field_name == "capacity"
                else "Resource stock"
            ),
        )

    path = RESULTS_DIR / filename

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)
    print(f"Saved {path}")


def plot_timeseries(
    rows: list[dict],
    *,
    metric: str,
    ylabel: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(
        figsize=(10.5, 6),
        layout="constrained",
    )

    for environment_id in ARCHETYPES:
        env_rows = [
            row
            for row in rows
            if row["environment"] == environment_id
        ]

        ax.plot(
            [row["time"] for row in env_rows],
            [row[metric] for row in env_rows],
            label=environment_id,
        )

    ax.set_xlabel("Time step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} through time")
    ax.grid(alpha=0.25)
    ax.margins(x=0)

    values = [float(row[metric]) for row in rows]

    if metric in {"resource_fraction", "scarcity_fraction"}:
        ax.set_ylim(0.0, 1.0)

    elif metric == "resource_gini":
        upper = max(values) * 1.10 if max(values) > 0 else 1.0
        ax.set_ylim(0.0, upper)

    if metric == "scarcity_fraction" and max(values) == 0.0:
        ax.text(
            0.5,
            0.52,
            "No tiles fell below 20% of local carrying capacity",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    ax.legend(
        title="Environment",
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    path = RESULTS_DIR / filename

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------
# Parameter-response experiment 1: mean capacity x equilibrium fraction
# ---------------------------------------------------------------------


def capacity_equilibrium_response(
    *,
    steps: int,
    coupling_rate: float,
    initial_fraction: float,
    seed: int,
) -> list[dict]:
    capacities = [0.30, 0.50, 0.70, 0.90]
    equilibrium_fractions = [0.30, 0.50, 0.70, 0.90]
    rows = []

    for mean_capacity in capacities:
        for equilibrium_fraction in equilibrium_fractions:
            model = build_model(
                mean_capacity=mean_capacity,
                equilibrium_fraction=equilibrium_fraction,
                recovery_rate=0.05,
                heterogeneity=0.50,
                coupling_rate=coupling_rate,
                initial_fraction=initial_fraction,
                seed=seed,
            )

            for _ in range(steps):
                model.step()

            rows.append(
                {
                    "mean_capacity": mean_capacity,
                    "equilibrium_fraction": equilibrium_fraction,
                    "mean_resource": float(np.mean(model.resource)),
                    "resource_fraction": float(
                        np.mean(model.resource / model.capacity)
                    ),
                }
            )

    return rows


def plot_capacity_equilibrium_response(rows: list[dict]) -> None:
    capacities = sorted(
        {row["mean_capacity"] for row in rows}
    )

    equilibrium_fractions = sorted(
        {row["equilibrium_fraction"] for row in rows}
    )

    matrix = np.zeros(
        (
            len(capacities),
            len(equilibrium_fractions),
        )
    )

    for row in rows:
        i = capacities.index(row["mean_capacity"])
        j = equilibrium_fractions.index(
            row["equilibrium_fraction"]
        )
        matrix[i, j] = row["mean_resource"]

    fig, ax = plt.subplots(
        figsize=(8, 6.5),
        layout="constrained",
    )

    image = ax.imshow(
        matrix,
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xticks(range(len(equilibrium_fractions)))
    ax.set_xticklabels(
        [f"{value:.2f}" for value in equilibrium_fractions]
    )
    ax.set_yticks(range(len(capacities)))
    ax.set_yticklabels(
        [f"{value:.2f}" for value in capacities]
    )

    ax.set_xlabel("Equilibrium fraction q")
    ax.set_ylabel("Mean carrying capacity")
    ax.set_title(
        "Effect of mean capacity and equilibrium fraction\n"
        "on final mean resource"
    )

    for i in range(len(capacities)):
        for j in range(len(equilibrium_fractions)):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Final mean resource",
        pad=0.02,
    )

    path = RESULTS_DIR / "06_capacity_equilibrium_response.png"

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------
# Parameter-response experiment 2: heterogeneity x recovery rate
# ---------------------------------------------------------------------


def heterogeneity_gini_response(
    *,
    steps: int,
    coupling_rate: float,
    initial_fraction: float,
    seed: int,
) -> list[dict]:
    heterogeneity_levels = [0.00, 0.25, 0.50, 0.75, 1.00]
    recovery_rates = [0.02, 0.05, 0.10]
    rows = []

    for recovery_rate in recovery_rates:
        for heterogeneity in heterogeneity_levels:
            model = build_model(
                mean_capacity=0.60,
                equilibrium_fraction=0.70,
                recovery_rate=recovery_rate,
                heterogeneity=heterogeneity,
                coupling_rate=coupling_rate,
                initial_fraction=initial_fraction,
                seed=seed,
            )

            for _ in range(steps):
                model.step()

            rows.append(
                {
                    "heterogeneity": heterogeneity,
                    "recovery_rate": recovery_rate,
                    "resource_gini": gini(model.resource),
                    "capacity_gini": gini(model.capacity),
                    "resource_fraction": float(
                        np.mean(model.resource / model.capacity)
                    ),
                }
            )

    return rows


def plot_heterogeneity_gini_response(rows: list[dict]) -> None:
    fig, ax = plt.subplots(
        figsize=(9.5, 6),
        layout="constrained",
    )

    recovery_rates = sorted(
        {row["recovery_rate"] for row in rows}
    )

    for recovery_rate in recovery_rates:
        subset = sorted(
            [
                row
                for row in rows
                if row["recovery_rate"] == recovery_rate
            ],
            key=lambda row: row["heterogeneity"],
        )

        ax.plot(
            [row["heterogeneity"] for row in subset],
            [
                max(0.0, row["resource_gini"])
                for row in subset
            ],
            marker="o",
            label=f"Recovery rate = {recovery_rate:.2f}",
        )

    maximum = max(
        max(0.0, row["resource_gini"])
        for row in rows
    )

    ax.set_xlabel("Structural heterogeneity")
    ax.set_ylabel("Final resource Gini")
    ax.set_title(
        "Effect of structural heterogeneity on\n"
        "realized resource inequality"
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(
        0.0,
        maximum * 1.10 if maximum > 0 else 1.0,
    )
    ax.grid(alpha=0.25)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    path = RESULTS_DIR / "07_heterogeneity_gini_response.png"

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------
# Recovery experiment: standardized local shock
# ---------------------------------------------------------------------


def shock_mask(model: EcologyModel, radius: float = 2.0) -> np.ndarray:
    y, x = np.indices(model.resource.shape)
    centre_x = (model.width - 1) / 2.0
    centre_y = (model.height - 1) / 2.0

    return (
        (x - centre_x) ** 2
        + (y - centre_y) ** 2
        <= radius**2
    )


def measure_recovery_time(
    *,
    recovery_rate: float,
    steps_before_shock: int,
    max_recovery_steps: int,
    coupling_rate: float,
    seed: int,
    shock_loss: float = 0.80,
    recovery_threshold: float = 0.90,
) -> int | None:
    model = build_model(
        mean_capacity=0.60,
        equilibrium_fraction=0.70,
        recovery_rate=recovery_rate,
        heterogeneity=0.80,
        coupling_rate=coupling_rate,
        initial_fraction=0.50,
        seed=seed,
    )

    for _ in range(steps_before_shock):
        model.step()

    mask = shock_mask(model)
    pre_shock_mean = float(np.mean(model.resource[mask]))
    target = recovery_threshold * pre_shock_mean

    model.resource[mask] *= (1.0 - shock_loss)

    for time in range(1, max_recovery_steps + 1):
        model.step()

        current_mean = float(np.mean(model.resource[mask]))

        if current_mean >= target:
            return time

    return None


def recovery_response(
    *,
    coupling_rate: float,
    seed: int,
) -> list[dict]:
    recovery_rates = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10]
    rows = []

    for recovery_rate in recovery_rates:
        recovery_time = measure_recovery_time(
            recovery_rate=recovery_rate,
            steps_before_shock=400,
            max_recovery_steps=2000,
            coupling_rate=coupling_rate,
            seed=seed,
        )

        rows.append(
            {
                "recovery_rate": recovery_rate,
                "recovery_time": (
                    recovery_time if recovery_time is not None else ""
                ),
            }
        )

    return rows


def plot_recovery_response(rows: list[dict]) -> None:
    usable = [
        row
        for row in rows
        if row["recovery_time"] != ""
    ]

    fig, ax = plt.subplots(
        figsize=(8.5, 6),
        layout="constrained",
    )

    x_values = [
        row["recovery_rate"]
        for row in usable
    ]

    y_values = [
        row["recovery_time"]
        for row in usable
    ]

    ax.plot(
        x_values,
        y_values,
        marker="o",
    )

    for x, y in zip(x_values, y_values):
        ax.annotate(
            str(y),
            (x, y),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
        )

    ax.set_xlabel("Specified recovery rate")
    ax.set_ylabel(
        "Steps to recover 90% of pre-shock resource"
    )
    ax.set_title(
        "Effect of recovery rate on observed recovery time"
    )
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=0)

    path = RESULTS_DIR / "08_recovery_rate_vs_recovery_time.png"

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scenario 1 ecology-only experiment and figures"
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coupling", type=float, default=0.10)
    parser.add_argument("--initial-fraction", type=float, default=0.50)

    args = parser.parse_args()

    ensure_results_dir()

    models, timeseries_rows, summary_rows = run_archetypes(
        steps=args.steps,
        coupling_rate=args.coupling,
        initial_fraction=args.initial_fraction,
        seed=args.seed,
    )

    write_csv(
        RESULTS_DIR / "archetype_timeseries.csv",
        timeseries_rows,
    )
    write_csv(
        RESULTS_DIR / "archetype_summary.csv",
        summary_rows,
    )

    plot_archetype_maps(
        models,
        field_name="capacity",
        filename="01_archetype_capacity_maps.png",
        title="Scenario 1 archetypes: carrying capacity",
    )
    plot_archetype_maps(
        models,
        field_name="resource",
        filename="02_archetype_final_resource_maps.png",
        title="Scenario 1 archetypes: final resource stock",
    )

    plot_timeseries(
        timeseries_rows,
        metric="resource_fraction",
        ylabel="Mean resource fraction",
        filename="03_resource_fraction_over_time.png",
    )
    plot_timeseries(
        timeseries_rows,
        metric="resource_gini",
        ylabel="Resource Gini",
        filename="04_resource_gini_over_time.png",
    )
    plot_timeseries(
        timeseries_rows,
        metric="scarcity_fraction",
        ylabel="Severe scarcity fraction (<20% of local capacity)",
        filename="05_scarcity_over_time.png",
    )

    capacity_rows = capacity_equilibrium_response(
        steps=args.steps,
        coupling_rate=args.coupling,
        initial_fraction=args.initial_fraction,
        seed=args.seed,
    )
    write_csv(
        RESULTS_DIR / "capacity_equilibrium_response.csv",
        capacity_rows,
    )
    plot_capacity_equilibrium_response(capacity_rows)

    heterogeneity_rows = heterogeneity_gini_response(
        steps=args.steps,
        coupling_rate=args.coupling,
        initial_fraction=args.initial_fraction,
        seed=args.seed,
    )
    write_csv(
        RESULTS_DIR / "heterogeneity_gini_response.csv",
        heterogeneity_rows,
    )
    plot_heterogeneity_gini_response(heterogeneity_rows)

    recovery_rows = recovery_response(
        coupling_rate=args.coupling,
        seed=args.seed,
    )
    write_csv(
        RESULTS_DIR / "recovery_response.csv",
        recovery_rows,
    )
    plot_recovery_response(recovery_rows)

    print("\nExperiment complete.")
    print(f"All CSV files and figures are in: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
