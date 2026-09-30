"""Five Stage-5 figures; statistical aggregation stays in visibility_analysis."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .visibility import PROFILES

LABELS = {
    "equal": "Equal",
    "random": "Random",
    "normal_centered": "Normal centered",
    "low_propensity_majority": "Low majority",
    "high_propensity_majority": "High majority",
    "uniform_high": "Uniform high",
    "patchy_high": "Patchy high",
    "split_high_low": "Split high/low",
    "fixed": "Fixed",
    "adaptive_bounded": "Adaptive",
}
COLORS = {"fixed": "#237b66", "adaptive_bounded": "#c15c36"}


def _save(fig, directory, name, caption=""):
    if caption:
        fig.text(0.5, -0.025, caption, ha="center", va="top", fontsize=9)
    fig.savefig(directory / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _n(rows):
    return max((int(r.get("n_replicates", 0)) for r in rows), default=0)


def _point(ax, row, x, *, color):
    mean = float(row["mean"])
    lower, upper = float(row["low"]), float(row["high"])
    ax.errorbar(
        x,
        mean,
        yerr=[[max(0, mean - lower)], [max(0, upper - mean)]],
        fmt="o",
        color=color,
        ms=4,
        lw=1.1,
        capsize=2,
    )


def _design(directory):
    from .scenarios import build_environment_maps

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.3), constrained_layout=True)
    for ax, ecology in zip(axes[0], ("uniform_high", "patchy_high", "split_high_low")):
        capacity, *_ = build_environment_maps(ecology, width=10, height=10, seed=20261002)
        ax.imshow(capacity, cmap="YlGn", vmin=0, vmax=1)
        ax.set_title(LABELS[ecology])
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1, 0].text(
        0.02,
        0.9,
        "Initial propensity → graph → realized visibility\n\n"
        "Equal · Random · Normal centered\nLow majority · High majority\n"
        "All observers attend k=4 sources",
        va="top",
        fontsize=11,
    )
    axes[1, 1].text(
        0.02,
        0.9,
        "Fixed: graph held constant\n\nAdaptive bounded:\n"
        "prediction error > .25\nrewire probability .10 every 50 steps\n"
        "75% local / 25% global search",
        va="top",
        fontsize=11,
    )
    axes[1, 2].text(
        0.02,
        0.9,
        "RQ1  Local perception vs population\n\n"
        "RQ2  Visibility concentration\n        and extraction behavior\n\n"
        "Secondary: resources, reserves, inequality",
        va="top",
        fontsize=11,
    )
    for ax in axes[1]:
        ax.axis("off")
    fig.suptitle("Ecology, social perception, and collective organization", fontsize=15)
    _save(
        fig,
        directory,
        "01_mechanism_and_design",
        "Configured ecology maps and study mechanism; no statistical estimates",
    )


def _manipulation(directory, tables):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    agents = tables["initial_visibility_agents"]
    # Reused ecologies/dynamics share graphs; one row per profile, replicate and agent.
    unique = {
        (r["visibility_profile"], r["population"], r["replicate"], r["agent"]): r for r in agents
    }
    for index, profile in enumerate(PROFILES):
        counts = [
            int(r["initial_visibility_count"]) for key, r in unique.items() if key[0] == profile
        ]
        if counts:
            axes[0].scatter([index] * len(counts), counts, alpha=0.08, s=5, color="#276f67")
    for ax, metric in ((axes[1], "visibility_gini"), (axes[2], "zero_visibility_fraction")):
        for index, profile in enumerate(PROFILES):
            row = next(
                (
                    r
                    for r in tables["initial_visibility_summary"]
                    if r["profile"] == profile and r["metric"] == metric
                ),
                None,
            )
            if row:
                _point(ax, row, index, color="#276f67")
        ax.set_xticks(range(5), [LABELS[p] for p in PROFILES], rotation=30, ha="right")
    axes[0].set_xticks(range(5), [LABELS[p] for p in PROFILES], rotation=30, ha="right")
    axes[0].set_ylabel("Observer count per source")
    axes[1].set_ylabel("Initial visibility Gini")
    axes[2].set_ylabel("Initially invisible fraction")
    fig.suptitle("Initial visibility: agent distributions and replicate intervals")
    _save(
        fig,
        directory,
        "02_initial_visibility_manipulation",
        f"Agent counts descriptive; Gini and invisible fraction are replicate means with 95% intervals (n≤{_n(tables['initial_visibility_summary'])})",
    )


def _factor_grid(directory, table, metrics, filename, title, caption=""):

    ecologies = ("uniform_high", "patchy_high", "split_high_low")
    fig, axes = plt.subplots(
        len(metrics),
        3,
        figsize=(16, 3.3 * len(metrics)),
        constrained_layout=True,
        squeeze=False,
        sharey="row",
    )
    for col, ecology in enumerate(ecologies):
        for row_index, (metric, label) in enumerate(metrics):
            ax = axes[row_index, col]
            for dynamic, offset in (("fixed", -0.13), ("adaptive_bounded", 0.13)):
                matches = [
                    r
                    for r in table
                    if r["scenario"] == ecology
                    and r["metric"] == metric
                    and r["dynamics"] == dynamic
                ]
                for index, profile in enumerate(PROFILES):
                    point = next((r for r in matches if r["profile"] == profile), None)
                    if point:
                        _point(ax, point, index + offset, color=COLORS[dynamic])
            ax.set_xticks(range(5), [LABELS[p] for p in PROFILES], rotation=30, ha="right")
            ax.set_xlim(-0.5, 4.5)
            if col == 0:
                ax.set_ylabel(label)
            if row_index == 0:
                ax.set_title(LABELS[ecology])
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=LABELS[name])
        for name, color in COLORS.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.085))
    fig.suptitle(title, y=1.15)
    _save(fig, directory, filename, caption)


def _paired_consequences(directory, table):
    metrics = (
        ("resource_fraction", "Resource / capacity"),
        ("reserve_welfare", "Reserve welfare"),
        ("final_wealth_gini", "Final wealth Gini"),
    )
    ecologies = ("uniform_high", "patchy_high", "split_high_low")
    fig, axes = plt.subplots(
        3, 3, figsize=(15, 10), constrained_layout=True, squeeze=False, sharey="row"
    )
    for row_index, (metric, label) in enumerate(metrics):
        for col, ecology in enumerate(ecologies):
            ax = axes[row_index, col]
            ax.axhline(0, color="#626262", lw=0.8)
            for index, profile in enumerate(PROFILES):
                point = next(
                    (
                        r
                        for r in table
                        if r["scenario"] == ecology
                        and r["profile"] == profile
                        and r["metric"] == metric
                    ),
                    None,
                )
                if point:
                    _point(ax, point, index, color="#594981")
            ax.set_xticks(range(5), [LABELS[p] for p in PROFILES], rotation=30, ha="right")
            ax.set_xlim(-0.5, 4.5)
            if col == 0:
                ax.set_ylabel(f"Adaptive − fixed\n{label}")
            if row_index == 0:
                ax.set_title(LABELS[ecology])
    fig.suptitle("Secondary consequences of adaptation: paired replicate differences")
    _save(
        fig,
        directory,
        "05_secondary_consequences",
        f"Fresh evaluation; paired Adaptive−Fixed replicate differences with 95% intervals (n≤{_n(table)})",
    )


def _trajectory(directory, table, metric, filename, ylabel):
    if not table:
        return
    ecologies = ("uniform_high", "patchy_high", "split_high_low")
    fig, axes = plt.subplots(
        5, 3, figsize=(15, 12), constrained_layout=True, squeeze=False, sharey=True
    )
    for row_index, profile in enumerate(PROFILES):
        for col, ecology in enumerate(ecologies):
            ax = axes[row_index, col]
            for dynamic in COLORS:
                rows = sorted(
                    (
                        r
                        for r in table
                        if r["scenario"] == ecology
                        and r["profile"] == profile
                        and r["dynamics"] == dynamic
                        and r["metric"] == metric
                    ),
                    key=lambda r: int(r["time"]),
                )
                if rows:
                    ax.plot(
                        [int(r["time"]) for r in rows],
                        [float(r["mean"]) for r in rows],
                        color=COLORS[dynamic],
                        label=LABELS[dynamic],
                    )
            if col == 0:
                ax.set_ylabel(f"{LABELS[profile]}\n{ylabel}")
            if row_index == 0:
                ax.set_title(LABELS[ecology])
            if row_index == 4:
                ax.set_xlabel("Training step")
    handles = [
        plt.Line2D([0], [0], color=color, label=LABELS[name]) for name, color in COLORS.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"Training trajectories · {ylabel}", y=1.14)
    _save(
        fig,
        directory,
        filename,
        f"Training checkpoints; descriptive means across independent replicates (n≤{_n(table)})",
    )


def _gini_decomposition(directory, table):
    _factor_grid(
        directory,
        table,
        (
            ("initial_visibility_gini", "Initial Gini G0"),
            ("terminal_visibility_gini", "Terminal Gini GT"),
            ("visibility_gini_change", "GT − G0"),
        ),
        "supplement_gini_decomposition",
        "Initial and learned visibility concentration",
        caption=f"Training step 0 and terminal step; replicate means with 95% intervals (n≤{_n(table)})",
    )


def save_visibility_figures(directory, tables):
    directory.mkdir(parents=True, exist_ok=True)
    _design(directory)
    _manipulation(directory, tables)
    _factor_grid(
        directory,
        tables["primary_window_summary"],
        (
            ("social_perception_error", "Perception error"),
            ("majority_mismatch_rate", "Majority mismatch (non-ties)"),
            ("majority_tie_rate", "Majority tie rate"),
        ),
        "03_population_perception",
        "RQ1 · Local views and population behavior",
        caption=f"Final min(1000,T) training steps; replicate means with 95% intervals (n≤{_n(tables['primary_window_summary'])})",
    )
    _factor_grid(
        directory,
        tables["primary_window_summary"],
        (
            ("visibility_gini", "Final-window visibility Gini"),
            ("low_extraction_rate", "Final-window low-extraction share"),
        ),
        "04_collective_organization",
        "RQ2 · Visibility and extraction",
        caption=f"Final min(1000,T) training steps; replicate means with 95% intervals (n≤{_n(tables['primary_window_summary'])})",
    )
    _paired_consequences(directory, tables["adaptive_fixed_summary"])
    _gini_decomposition(directory, tables["outcome_summary"])
    for metric, filename, label in (
        ("visibility_gini", "S2_visibility_gini_trajectories", "Visibility Gini"),
        ("low_extraction_rate", "S2_low_extraction_trajectories", "Low-extraction share"),
    ):
        _trajectory(directory, tables["training_trajectory_summary"], metric, filename, label)
