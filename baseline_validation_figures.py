from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qlearning_experiment import SCENARIOS, build_environment_maps


RESULTS_ROOT = Path("results") / "q_learning_baseline" / "experiments"

SHORT_LABELS = {
    "uniform_high": "Uniform high",
    "patchy_high": "Patchy high",
    "centralized_low": "Central low",
    "decentralized_high": "Decentralized high",
    "patchy_high_central_low": "Patchy high\n+ central low",
    "central_high_patchy_low": "Central high\n+ patchy low",
    "split_high_low": "High/low\nsplit",
    "decentralized_high_in_low": "High islands\n+ low background",
}

STRATEGY_LABELS = {
    "q_learning": "Q-learning",
    "always_low": "Always low",
    "always_high": "Always high",
    "random_50": "Random 50/50",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def as_float(row: dict, key: str) -> float:
    value = row.get(key, "")

    if value in ("", None):
        return float("nan")

    return float(value)


def ci95(values) -> tuple[float, float]:
    values = np.asarray(
        [value for value in values if np.isfinite(value)],
        dtype=float,
    )

    if len(values) == 0:
        return float("nan"), float("nan")

    mean = float(values.mean())

    if len(values) == 1:
        return mean, 0.0

    sem = float(values.std(ddof=1) / math.sqrt(len(values)))
    return mean, 1.96 * sem


def aggregate(
    rows: list[dict],
    *,
    group_keys: list[str],
    metric: str,
) -> list[dict]:
    groups = {}

    for row in rows:
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(as_float(row, metric))

    output = []

    for key, values in groups.items():
        mean, ci = ci95(values)

        record = {
            name: value
            for name, value in zip(group_keys, key)
        }

        record["mean"] = mean
        record["ci95"] = ci
        output.append(record)

    return output


def scenario_grid(n: int):
    columns = 4
    rows = int(math.ceil(n / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 3.5 * rows),
        squeeze=False,
        layout="constrained",
    )

    return fig, axes.ravel()


def hide_unused(axes, used: int) -> None:
    for ax in axes[used:]:
        ax.axis("off")


def qlearning_fresh_rows(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if (
            row["strategy"] == "q_learning"
            and row["evaluation_mode"] == "fresh_reset"
        )
    ]


def environment_catalog(config: dict, figures_dir: Path) -> None:
    scenarios = list(config["scenarios"])
    fig, axes = scenario_grid(len(scenarios))
    image = None

    for index, scenario in enumerate(scenarios):
        capacity, _, _, _ = build_environment_maps(
            scenario,
            width=int(config["width"]),
            height=int(config["height"]),
            seed=int(config["seed"]),
        )

        image = axes[index].imshow(
            capacity,
            vmin=0.0,
            vmax=1.0,
        )

        axes[index].set_title(
            SHORT_LABELS.get(scenario, scenario),
            fontsize=10,
        )
        axes[index].set_xticks([])
        axes[index].set_yticks([])

    hide_unused(axes, len(scenarios))

    fig.suptitle("Ecological environments", fontsize=14)

    if image is not None:
        fig.colorbar(
            image,
            ax=axes[:len(scenarios)].tolist(),
            label="Carrying capacity K",
            shrink=0.80,
        )

    path = figures_dir / "00_environment_catalog.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_q_metric(
    evaluation_rows: list[dict],
    config: dict,
    figures_dir: Path,
    *,
    metric: str,
    ylabel: str,
    filename: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    rows = qlearning_fresh_rows(evaluation_rows)

    agg = aggregate(
        rows,
        group_keys=["scenario", "population"],
        metric=metric,
    )

    scenarios = list(config["scenarios"])
    fig, axes = scenario_grid(len(scenarios))

    for index, scenario in enumerate(scenarios):
        subset = sorted(
            [row for row in agg if row["scenario"] == scenario],
            key=lambda row: int(row["population"]),
        )

        x = [int(row["population"]) for row in subset]
        y = [row["mean"] for row in subset]
        ci = [row["ci95"] for row in subset]

        axes[index].errorbar(
            x,
            y,
            yerr=ci,
            marker="o",
            capsize=3,
        )

        axes[index].set_title(
            SHORT_LABELS.get(scenario, scenario),
            fontsize=10,
        )
        axes[index].set_xlabel("Population")
        axes[index].set_xticks(x)
        axes[index].grid(alpha=0.25)

        if ylim is not None:
            axes[index].set_ylim(*ylim)

        if index % 4 == 0:
            axes[index].set_ylabel(ylabel)

    hide_unused(axes, len(scenarios))

    path = figures_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def welfare_inequality_state_space(
    evaluation_rows: list[dict],
    config: dict,
    figures_dir: Path,
) -> None:
    rows = qlearning_fresh_rows(evaluation_rows)
    scenarios = list(config["scenarios"])
    fig, axes = scenario_grid(len(scenarios))

    for index, scenario in enumerate(scenarios):
        subset = [row for row in rows if row["scenario"] == scenario]
        populations = sorted({int(row["population"]) for row in subset})

        for population in populations:
            group = [
                row
                for row in subset
                if int(row["population"]) == population
            ]

            welfare, welfare_ci = ci95(
                [
                    as_float(row, "eval_mean_mean_reserve_welfare")
                    for row in group
                ]
            )

            inequality, inequality_ci = ci95(
                [
                    as_float(row, "final_wealth_gini")
                    for row in group
                ]
            )

            axes[index].errorbar(
                inequality,
                welfare,
                xerr=inequality_ci,
                yerr=welfare_ci,
                marker="o",
                capsize=2,
            )

            axes[index].annotate(
                f"N={population}",
                (inequality, welfare),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        axes[index].set_title(
            SHORT_LABELS.get(scenario, scenario),
            fontsize=10,
        )
        axes[index].set_xlabel("Wealth Gini")
        axes[index].set_xlim(0.0, 1.0)
        axes[index].set_ylim(0.0, 1.0)
        axes[index].grid(alpha=0.25)

        if index % 4 == 0:
            axes[index].set_ylabel("Reserve welfare")

    hide_unused(axes, len(scenarios))

    path = figures_dir / "03_welfare_inequality_state_space.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_strategy_metric(
    evaluation_rows: list[dict],
    config: dict,
    figures_dir: Path,
    *,
    metric: str,
    ylabel: str,
    filename: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    rows = [
        row
        for row in evaluation_rows
        if row["evaluation_mode"] == "fresh_reset"
    ]

    agg = aggregate(
        rows,
        group_keys=["scenario", "population", "strategy"],
        metric=metric,
    )

    strategies = [
        "q_learning",
        "always_low",
        "always_high",
        "random_50",
    ]

    scenarios = list(config["scenarios"])
    fig, axes = scenario_grid(len(scenarios))
    legend_handles = []
    legend_labels = []

    for index, scenario in enumerate(scenarios):
        for strategy in strategies:
            subset = sorted(
                [
                    row
                    for row in agg
                    if (
                        row["scenario"] == scenario
                        and row["strategy"] == strategy
                    )
                ],
                key=lambda row: int(row["population"]),
            )

            if not subset:
                continue

            x = [int(row["population"]) for row in subset]
            y = [row["mean"] for row in subset]
            ci = [row["ci95"] for row in subset]

            line = axes[index].errorbar(
                x,
                y,
                yerr=ci,
                marker="o",
                capsize=2,
            )

            if index == 0:
                legend_handles.append(line)
                legend_labels.append(STRATEGY_LABELS[strategy])

        axes[index].set_title(
            SHORT_LABELS.get(scenario, scenario),
            fontsize=10,
        )
        axes[index].set_xlabel("Population")
        axes[index].grid(alpha=0.25)

        if ylim is not None:
            axes[index].set_ylim(*ylim)

        if index % 4 == 0:
            axes[index].set_ylabel(ylabel)

    hide_unused(axes, len(scenarios))

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
    )

    path = figures_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def state_occupancy_and_behavior(
    evaluation_rows: list[dict],
    config: dict,
    figures_dir: Path,
) -> None:
    rows = qlearning_fresh_rows(evaluation_rows)
    scenarios = list(config["scenarios"])
    populations = sorted({int(row["population"]) for row in rows})

    labels = []
    occupancy = []
    behavior = []

    for scenario in scenarios:
        for population in populations:
            subset = [
                row
                for row in rows
                if (
                    row["scenario"] == scenario
                    and int(row["population"]) == population
                )
            ]

            if not subset:
                continue

            labels.append(
                f"{SHORT_LABELS.get(scenario, scenario).replace(chr(10), ' ')} | N={population}"
            )

            occupancy.append(
                [
                    ci95(
                        [
                            as_float(row, f"state_occupancy_{state}")
                            for row in subset
                        ]
                    )[0]
                    for state in ("scarce", "moderate", "abundant")
                ]
            )

            behavior.append(
                [
                    ci95(
                        [
                            as_float(row, f"low_given_{state}")
                            for row in subset
                        ]
                    )[0]
                    for state in ("scarce", "moderate", "abundant")
                ]
            )

    occupancy = np.asarray(occupancy, dtype=float)
    behavior = np.asarray(behavior, dtype=float)

    height = max(7.0, 0.34 * len(labels))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, height),
        layout="constrained",
    )

    for ax, matrix, title, cbar_label in (
        (
            axes[0],
            occupancy,
            "Experienced ecological states",
            "Fraction of agent-time",
        ),
        (
            axes[1],
            behavior,
            "Behavior conditional on state",
            "P(low extraction | state)",
        ),
    ):
        image = ax.imshow(
            matrix,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )

        ax.set_xticks(range(3))
        ax.set_xticklabels(["Scarce", "Moderate", "Abundant"])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title)

        fig.colorbar(
            image,
            ax=ax,
            label=cbar_label,
            shrink=0.85,
        )

    path = figures_dir / "09_state_occupancy_and_behavior.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def reset_comparison(
    evaluation_rows: list[dict],
    config: dict,
    figures_dir: Path,
) -> None:
    rows = [
        row
        for row in evaluation_rows
        if row["strategy"] == "q_learning"
    ]

    metrics = [
        (
            "eval_mean_mean_reserve_welfare",
            "Reserve welfare",
        ),
        (
            "final_wealth_gini",
            "Wealth Gini",
        ),
    ]

    scenarios = list(config["scenarios"])

    fig, axes = plt.subplots(
        2,
        len(scenarios),
        figsize=(3.0 * len(scenarios), 7.0),
        squeeze=False,
        layout="constrained",
    )

    modes = [
        ("continuation", "Continue trained ecology"),
        ("fresh_reset", "Fresh reset"),
    ]

    handles = []
    labels = []

    for row_index, (metric, ylabel) in enumerate(metrics):
        agg = aggregate(
            rows,
            group_keys=["scenario", "population", "evaluation_mode"],
            metric=metric,
        )

        for col_index, scenario in enumerate(scenarios):
            ax = axes[row_index, col_index]

            for mode, label in modes:
                subset = sorted(
                    [
                        row
                        for row in agg
                        if (
                            row["scenario"] == scenario
                            and row["evaluation_mode"] == mode
                        )
                    ],
                    key=lambda row: int(row["population"]),
                )

                if not subset:
                    continue

                x = [int(row["population"]) for row in subset]
                y = [row["mean"] for row in subset]
                ci = [row["ci95"] for row in subset]

                line = ax.errorbar(
                    x,
                    y,
                    yerr=ci,
                    marker="o",
                    capsize=2,
                )

                if row_index == 0 and col_index == 0:
                    handles.append(line)
                    labels.append(label)

            if row_index == 0:
                ax.set_title(
                    SHORT_LABELS.get(scenario, scenario),
                    fontsize=9,
                )

            ax.set_xlabel("Population")
            ax.grid(alpha=0.25)

            if col_index == 0:
                ax.set_ylabel(ylabel)

            if metric == "eval_mean_mean_reserve_welfare":
                ax.set_ylim(0.0, 1.0)
            elif metric == "final_wealth_gini":
                ax.set_ylim(0.0, 1.0)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
    )

    path = figures_dir / "10_continuation_vs_fresh_reset.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def policy_heterogeneity(
    policy_rows: list[dict],
    config: dict,
    figures_dir: Path,
) -> None:
    agg = aggregate(
        policy_rows,
        group_keys=["scenario", "population"],
        metric="policy_hamming_mean",
    )

    scenarios = list(config["scenarios"])
    fig, axes = scenario_grid(len(scenarios))

    for index, scenario in enumerate(scenarios):
        subset = sorted(
            [row for row in agg if row["scenario"] == scenario],
            key=lambda row: int(row["population"]),
        )

        x = [int(row["population"]) for row in subset]
        y = [row["mean"] for row in subset]
        ci = [row["ci95"] for row in subset]

        axes[index].errorbar(
            x,
            y,
            yerr=ci,
            marker="o",
            capsize=3,
        )

        axes[index].set_title(
            SHORT_LABELS.get(scenario, scenario),
            fontsize=10,
        )
        axes[index].set_xlabel("Population")
        axes[index].set_ylim(0.0, 1.0)
        axes[index].grid(alpha=0.25)

        if index % 4 == 0:
            axes[index].set_ylabel("Pairwise policy Hamming distance")

    hide_unused(axes, len(scenarios))

    path = figures_dir / "11_policy_heterogeneity.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def training_trajectories(
    training_rows: list[dict],
    config: dict,
    figures_dir: Path,
) -> None:
    scenarios = list(config["scenarios"])
    max_population = max(int(value) for value in config["populations"])

    metrics = [
        (
            "mean_reserve_welfare",
            "Reserve welfare",
            (0.0, 1.0),
        ),
        (
            "wealth_gini",
            "Wealth Gini",
            (0.0, 1.0),
        ),
    ]

    rows = [
        row
        for row in training_rows
        if int(row["population"]) == max_population
    ]

    fig, axes = plt.subplots(
        2,
        len(scenarios),
        figsize=(3.0 * len(scenarios), 7.0),
        squeeze=False,
        layout="constrained",
    )

    for row_index, (metric, ylabel, ylim) in enumerate(metrics):
        agg = aggregate(
            rows,
            group_keys=["scenario", "time"],
            metric=metric,
        )

        for col_index, scenario in enumerate(scenarios):
            ax = axes[row_index, col_index]

            subset = sorted(
                [row for row in agg if row["scenario"] == scenario],
                key=lambda row: int(row["time"]),
            )

            x = np.asarray(
                [int(row["time"]) for row in subset],
                dtype=float,
            )
            y = np.asarray(
                [row["mean"] for row in subset],
                dtype=float,
            )
            ci = np.asarray(
                [row["ci95"] for row in subset],
                dtype=float,
            )

            ax.plot(x, y)
            ax.fill_between(
                x,
                y - ci,
                y + ci,
                alpha=0.2,
            )

            if row_index == 0:
                ax.set_title(
                    SHORT_LABELS.get(scenario, scenario),
                    fontsize=9,
                )

            ax.set_xlabel("Training step")
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.25)

            if col_index == 0:
                ax.set_ylabel(ylabel)

    fig.suptitle(
        f"Training trajectories at N={max_population}",
        fontsize=14,
    )

    path = figures_dir / "12_training_welfare_inequality.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def write_aggregate_summary(
    evaluation_rows: list[dict],
    run_dir: Path,
) -> None:
    metrics = [
        "eval_mean_mean_reserve_welfare",
        "eval_mean_mean_need_satisfaction",
        "eval_mean_deprivation_rate",
        "eval_mean_mean_metabolic_shortfall",
        "final_wealth_gini",
        "eval_mean_mean_wealth",
        "eval_mean_low_extraction_rate",
        "eval_mean_collective_order",
        "eval_mean_mean_resource_fraction",
        "state_occupancy_scarce",
        "state_occupancy_moderate",
        "state_occupancy_abundant",
        "low_given_scarce",
        "low_given_moderate",
        "low_given_abundant",
    ]

    group_keys = [
        "scenario",
        "strategy",
        "evaluation_mode",
        "population",
    ]

    combined = {}

    for metric in metrics:
        rows = aggregate(
            evaluation_rows,
            group_keys=group_keys,
            metric=metric,
        )

        for row in rows:
            key = tuple(row[name] for name in group_keys)

            if key not in combined:
                combined[key] = {
                    name: value
                    for name, value in zip(group_keys, key)
                }

            combined[key][f"{metric}_mean"] = row["mean"]
            combined[key][f"{metric}_ci95"] = row["ci95"]

    output = list(combined.values())
    path = run_dir / "data" / "aggregate_summary.csv"

    if output:
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=output[0].keys())
            writer.writeheader()
            writer.writerows(output)

    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate clean baseline-validation figures."
    )
    parser.add_argument("--run-name", default="baseline_validation_v1")
    args = parser.parse_args()

    run_dir = RESULTS_ROOT / args.run_name
    data_dir = run_dir / "data"
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open() as file:
        config = json.load(file)

    evaluation_rows = read_csv(data_dir / "evaluation_summary.csv")
    training_rows = read_csv(data_dir / "training_timeseries.csv")
    policy_rows = read_csv(data_dir / "policy_summary.csv")

    write_aggregate_summary(evaluation_rows, run_dir)
    environment_catalog(config, figures_dir)

    # Primary outcomes: welfare and inequality.
    plot_q_metric(
        evaluation_rows,
        config,
        figures_dir,
        metric="eval_mean_mean_reserve_welfare",
        ylabel="Reserve welfare",
        filename="01_primary_welfare.png",
        ylim=(0.0, 1.0),
    )

    plot_q_metric(
        evaluation_rows,
        config,
        figures_dir,
        metric="final_wealth_gini",
        ylabel="Final wealth Gini",
        filename="02_primary_inequality.png",
        ylim=(0.0, 1.0),
    )

    welfare_inequality_state_space(
        evaluation_rows,
        config,
        figures_dir,
    )

    # Explanatory social-ecological variables.
    plot_q_metric(
        evaluation_rows,
        config,
        figures_dir,
        metric="eval_mean_low_extraction_rate",
        ylabel="Low-extraction rate",
        filename="04_low_extraction_cooperation_proxy.png",
        ylim=(0.0, 1.0),
    )

    plot_q_metric(
        evaluation_rows,
        config,
        figures_dir,
        metric="eval_mean_mean_resource_fraction",
        ylabel="Mean resource fraction R/K",
        filename="05_resource_condition.png",
        ylim=(0.0, 1.0),
    )

    # Fixed-policy controls show how policy type changes the main outcomes.
    plot_strategy_metric(
        evaluation_rows,
        config,
        figures_dir,
        metric="eval_mean_mean_reserve_welfare",
        ylabel="Reserve welfare",
        filename="06_fixed_policy_welfare.png",
        ylim=(0.0, 1.0),
    )

    plot_strategy_metric(
        evaluation_rows,
        config,
        figures_dir,
        metric="final_wealth_gini",
        ylabel="Final wealth Gini",
        filename="07_fixed_policy_inequality.png",
        ylim=(0.0, 1.0),
    )

    plot_strategy_metric(
        evaluation_rows,
        config,
        figures_dir,
        metric="eval_mean_mean_resource_fraction",
        ylabel="Mean resource fraction R/K",
        filename="08_fixed_policy_resource.png",
        ylim=(0.0, 1.0),
    )

    state_occupancy_and_behavior(
        evaluation_rows,
        config,
        figures_dir,
    )

    reset_comparison(
        evaluation_rows,
        config,
        figures_dir,
    )

    policy_heterogeneity(
        policy_rows,
        config,
        figures_dir,
    )

    training_trajectories(
        training_rows,
        config,
        figures_dir,
    )

    print("\nFigure generation complete.")
    print(f"Figures: {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
