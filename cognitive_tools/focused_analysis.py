"""The frozen ecology/perception study; legacy diagnostics remain opt-in."""

from collections import defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

from .analysis import (
    as_float,
    as_int,
    bootstrap_mean_ci,
    primary_evaluation_rows,
    run_label_order,
    stable_seed,
    write_csv_rows,
)

WINDOW = 1000
METRICS = {
    "social_perception_error": "Population-perception error",
    "majority_mismatch_rate": "Majority mismatch (non-ties)",
    "majority_tie_rate": "Majority tie rate",
    "visibility_gini": "Visibility Gini",
    "low_extraction_rate": "Low-extraction share",
}
OUTCOMES = {
    "eval_mean_mean_resource_fraction": "Resource / capacity",
    "eval_mean_mean_reserve_welfare": "Reserve welfare",
    "final_wealth_gini": "Final wealth Gini",
}


def build_window_rows(runs):
    """Equal-weight recorded checkpoints in (max(0, T-1000), T]."""
    output = []
    for run in runs:
        end = int(run.config["training_steps"])
        start = max(0, end - WINDOW)
        groups = defaultdict(list)
        seen = set()
        for row in run.tables["training_timeseries"]:
            key = (row["scenario"], as_int(row, "population"), as_int(row, "replicate"))
            time = as_int(row, "time")
            if start < time <= end:
                if (*key, time) in seen:
                    raise ValueError(f"Duplicate training checkpoint: {run.run_id}, {key}, {time}")
                seen.add((*key, time))
                groups[key].append(row)
        if not groups:
            raise ValueError(f"No training checkpoints in primary window: {run.run_id}")
        for (scenario, population, replicate), rows in groups.items():
            for metric in METRICS:
                if run.treatment == "B0" and metric != "low_extraction_rate":
                    continue
                values = [as_float(row, metric) for row in rows]
                valid = [value for value in values if np.isfinite(value)]
                output.append(
                    dict(
                        run_id=run.run_id,
                        analysis_label=run.label,
                        scenario=scenario,
                        population=population,
                        replicate=replicate,
                        metric=metric,
                        value=float(np.mean(valid)) if valid else float("nan"),
                        window_start_exclusive=start,
                        window_end_inclusive=end,
                        n_checkpoints=len(rows),
                        n_valid=len(valid),
                    )
                )
    return output


def summarize(rows, keys, *, bootstrap_reps, bootstrap_seed):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row["value"])
    output = []
    for key, values in groups.items():
        mean, low, high = bootstrap_mean_ci(
            values, bootstrap_reps=bootstrap_reps, seed=stable_seed(bootstrap_seed, "focused", *key)
        )
        output.append(
            dict(zip(keys, key))
            | dict(
                n=int(np.isfinite(values).sum()),
                n_total=len(values),
                mean=mean,
                bootstrap_ci_low=low,
                bootstrap_ci_high=high,
            )
        )
    return output


def build_contrasts(rows, runs, **bootstrap):
    """Three planned families, with exact within-replicate pairing."""
    cells = defaultdict(dict)
    for row in rows:
        key = (row["run_id"], row["scenario"], row["population"], row["metric"])
        rep = row["replicate"]
        if rep in cells[key]:
            raise ValueError(f"Duplicate replicate: {key}, {rep}")
        cells[key][rep] = row["value"]
    lookup = {run.run_id: run for run in runs}
    output = []
    for left, right in combinations(cells, 2):
        a, b = lookup[left[0]], lookup[right[0]]
        if left[2:] != right[2:]:
            continue
        family = None
        if a.run_id == b.run_id and left[1] != right[1] and a.treatment != "B0":
            family = "ecology"
        elif left[1] == right[1] and a.run_id != b.run_id:
            if b.paired_adaptive_run_id == a.run_id:
                family = "adaptive_minus_matched"
            elif a.paired_adaptive_run_id == b.run_id:
                left, right, a, b = right, left, b, a
                family = "adaptive_minus_matched"
            elif {a.treatment, b.treatment} == {"S1", "S2"}:
                family = "fixed_structure"
                if a.treatment == "S1":
                    left, right = right, left
            elif (
                a.treatment in {"R1", "R2", "R3"}
                and b.treatment in {"R1", "R2", "R3"}
                and a.mu == b.mu
            ):
                family = "adaptive_scope"
            elif a.treatment == b.treatment == "R0" and a.mu == b.mu and a.theta != b.theta:
                family = "random_scope"
        if family is None:
            continue
        if cells[left].keys() != cells[right].keys():
            raise ValueError(f"Unmatched replicate sets: {left}, {right}")
        for rep, value in cells[left].items():
            output.append(
                dict(
                    family=family,
                    left_run=left[0],
                    right_run=right[0],
                    left_scenario=left[1],
                    right_scenario=right[1],
                    population=left[2],
                    metric=left[3],
                    replicate=rep,
                    value=value - cells[right][rep],
                )
            )
    keys = [
        "family",
        "left_run",
        "right_run",
        "left_scenario",
        "right_scenario",
        "population",
        "metric",
    ]
    return output, summarize(output, keys, **bootstrap)


def save_figure(fig, directory, name):
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_design(directory):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    boxes = [
        (0.13, "Ecological conditions\nResource renewal and space"),
        (0.50, "Individual learning + observation\nExtraction and local perception"),
        (0.87, "Population patterns\nVisibility and behavior"),
    ]
    for x, label in boxes:
        ax.text(
            x,
            0.68,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=.7", fc="#eef4f5", ec="#31688e"),
        )
    for left, right in ((0.27, 0.35), (0.67, 0.73)):
        ax.annotate(
            "",
            xy=(right, 0.68),
            xytext=(left, 0.68),
            arrowprops=dict(arrowstyle="<->", color="#31688e"),
        )
    ax.text(
        0.5,
        0.32,
        "Consequences: shared resources • individual welfare • wealth inequality",
        ha="center",
        fontsize=12,
    )
    ax.text(
        0.5,
        0.10,
        "B0: ecology only | S1 / S2: fixed observation\n"
        "Local / mixed / global replacement: adaptive versus event-count-matched random",
        ha="center",
        fontsize=10,
    )
    ax.set_title("Ecology, social perception, and collective organization", fontsize=16)
    save_figure(fig, directory, "01_mechanism_and_design")


def save_summary_figure(summary, runs, directory, metrics, name, title, *, differences=False):
    scenarios = list(dict.fromkeys(row["scenario"] for row in summary))
    populations = sorted({row["population"] for row in summary})
    present = {row["analysis_label"] for row in summary if row["metric"] in metrics}
    labels = [label for label in run_label_order(runs) if label in present]
    display = {
        run.label: (
            f"R0 / { {0: 'local', 0.25: 'mixed', 1: 'global'}.get(run.theta, run.theta) }"
            if run.treatment == "R0"
            else run.label
        )
        for run in runs
    }
    fig, axes = plt.subplots(
        len(metrics),
        len(scenarios) * len(populations),
        figsize=(5 * len(scenarios) * len(populations), 3.8 * len(metrics)),
        squeeze=False,
        layout="constrained",
    )
    for col, (scenario, pop) in enumerate((s, p) for s in scenarios for p in populations):
        for index, (metric, label) in enumerate(metrics.items()):
            ax = axes[index, col]
            subset = [
                r
                for r in summary
                if r["scenario"] == scenario and r["population"] == pop and r["metric"] == metric
            ]
            for row in subset:
                y = labels.index(row["analysis_label"])
                ax.plot(
                    [row["bootstrap_ci_low"], row["bootstrap_ci_high"]], [y, y], color="#31688e"
                )
                ax.plot(row["mean"], y, "o", color="#31688e", markersize=4)
            upper = [
                r["bootstrap_ci_high"]
                for r in summary
                if r["metric"] == metric and np.isfinite(r["bootstrap_ci_high"])
            ]
            limits = (0, min(1, max(0.1, np.ceil(max(upper, default=1) * 11) / 10)))
            if differences:
                bounds = [
                    abs(r[field])
                    for r in summary
                    if r["metric"] == metric
                    for field in ("bootstrap_ci_low", "bootstrap_ci_high")
                    if np.isfinite(r[field])
                ]
                extent = max([0.01, *bounds]) * 1.1
                limits = (-extent, extent)
                ax.axvline(0, color="0.5", linewidth=0.8)
            ax.set(
                yticks=range(len(labels)),
                yticklabels=[display[item] for item in labels],
                xlim=limits,
                xlabel=("Difference in " + label if differences else label),
            )
            ax.invert_yaxis()
            if index == 0:
                ax.set_title(f"{scenario.replace('_', ' ')} · N={pop}")
            ax.grid(axis="x", alpha=0.15)
    fig.suptitle(
        title
        + "\n"
        + ("Paired differences" if differences else "Replicate means")
        + " and pointwise 95% bootstrap intervals",
        fontsize=14,
    )
    save_figure(fig, directory, name)


def save_organization(runs, directory):
    groups = defaultdict(list)
    for run in runs:
        for row in run.tables["training_timeseries"]:
            for metric in ("visibility_gini", "low_extraction_rate"):
                if run.treatment == "B0" and metric == "visibility_gini":
                    continue
                key = (
                    row["scenario"],
                    as_int(row, "population"),
                    run.label,
                    as_int(row, "time"),
                    metric,
                )
                groups[key].append(as_float(row, metric))
    panels = sorted({key[:2] for key in groups})
    labels = run_label_order(runs)
    colors = dict(zip(labels, plt.get_cmap("tab10").colors))
    fig, axes = plt.subplots(2, len(panels), figsize=(5 * len(panels), 7), squeeze=False)
    for col, (scenario, pop) in enumerate(panels):
        for i, metric in enumerate(("visibility_gini", "low_extraction_rate")):
            ax = axes[i, col]
            for label in labels:
                series = sorted(
                    (key[3], np.mean(values))
                    for key, values in groups.items()
                    if key[:3] == (scenario, pop, label) and key[4] == metric
                )
                if series:
                    x, y = zip(*series)
                    ax.plot(
                        x,
                        y,
                        label=label,
                        color=colors[label],
                        linestyle="--" if label.startswith("R0") else "-",
                        linewidth=1,
                    )
            ax.set(ylim=(0, 1), xlabel="Training step", ylabel=METRICS[metric])
            ax.grid(alpha=0.15)
        axes[0, col].set_title(f"{scenario.replace('_', ' ')} · N={pop}")
    handles, legend = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, legend, loc="lower center", ncol=5, fontsize=9)
    fig.suptitle("Collective organization during training — replicate means, unsmoothed")
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    save_figure(fig, directory, "03_collective_organization")


def run_focused_analysis(runs, tables_dir, figures_dir, *, bootstrap_reps, bootstrap_seed):
    bootstrap = dict(bootstrap_reps=bootstrap_reps, bootstrap_seed=bootstrap_seed)
    window = build_window_rows(runs)
    outcomes = []
    for run in runs:
        for row in primary_evaluation_rows(run):
            for metric in OUTCOMES:
                outcomes.append(
                    dict(
                        run_id=run.run_id,
                        analysis_label=run.label,
                        scenario=row["scenario"],
                        population=as_int(row, "population"),
                        replicate=as_int(row, "replicate"),
                        metric=metric,
                        value=as_float(row, metric),
                    )
                )
    keys = ["run_id", "analysis_label", "scenario", "population", "metric"]
    summary = summarize(window, keys, **bootstrap)
    outcome_summary = summarize(outcomes, keys, **bootstrap)
    contrasts, contrast_summary = build_contrasts(window + outcomes, runs, **bootstrap)
    for name, rows in (
        ("primary_window_replicates", window),
        ("primary_window_summary", summary),
        ("outcome_replicates", outcomes),
        ("outcome_summary", outcome_summary),
        ("planned_contrasts", contrasts),
        ("planned_contrast_summary", contrast_summary),
    ):
        write_csv_rows(tables_dir / f"{name}.csv", rows)
    save_design(figures_dir)
    save_summary_figure(
        summary,
        runs,
        figures_dir,
        dict(list(METRICS.items())[:3]),
        "02_population_perception",
        "Population perception — final training window",
    )
    save_organization(runs, figures_dir)
    labels = {run.run_id: run.label for run in runs}
    matched = [
        dict(row, analysis_label=labels[row["left_run"]], scenario=row["left_scenario"])
        for row in contrast_summary
        if row["family"] == "adaptive_minus_matched" and row["metric"] in OUTCOMES
    ]
    save_summary_figure(
        matched or outcome_summary,
        runs,
        figures_dir,
        OUTCOMES,
        "04_resource_and_welfare",
        "Adaptive minus matched random — frozen evaluation"
        if matched
        else "Resource and welfare — frozen evaluation",
        differences=bool(matched),
    )
