"""Five Stage-5 figures; statistical aggregation stays in visibility_analysis."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from .visibility import PROFILES
from .visibility_palette import DYNAMICS_COLORS, DYNAMICS_MARKERS, GRID, INK, MUTED

LABELS = {
    "equal": "Equal",
    "random": "Random",
    "normal_centered": "Normal centered",
    "low_propensity_majority": "Low majority",
    "high_propensity_majority": "High majority",
    "uniform_high": "Uniform high",
    "patchy_high": "Patchy high",
    "split_high_low": "Split high/low",
    "balanced_uniform": "Uniform · K=75",
    "balanced_dispersed": "Dispersed · K=75",
    "balanced_segregated": "Segregated · K=75",
    "fixed": "Fixed",
    "adaptive_bounded": "Adaptive",
}
COLORS = DYNAMICS_COLORS


def _save(fig, directory, name, caption=""):
    if caption:
        fig.text(0.5, -0.025, caption, ha="center", va="top", fontsize=9, color=MUTED)
        (directory / f"{name}.caption.txt").write_text(caption + "\n")
    fig.savefig(directory / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _n(rows):
    return max((int(r.get("n_replicates", 0)) for r in rows), default=0)


def _point(ax, row, x, *, color, marker="o"):
    mean = float(row["mean"])
    lower, upper = float(row["low"]), float(row["high"])
    ax.errorbar(
        x,
        mean,
        yerr=[[max(0, mean - lower)], [max(0, upper - mean)]],
        fmt=marker,
        color=color,
        ms=5,
        lw=1.35,
        capsize=2,
    )


def _design(directory, ecologies, *, width=10, height=10, seed=20261002):
    from .scenarios import build_environment_maps

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.3), constrained_layout=True)
    for ax, ecology in zip(axes[0], ecologies):
        capacity, *_ = build_environment_maps(ecology, width=width, height=height, seed=seed)
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
        "75% two-hop / 25% global candidate search",
        va="top",
        fontsize=11,
    )
    axes[1, 2].text(
        0.02,
        0.9,
        "RQ1  Four peers vs population extraction\n\n"
        "RQ2  Extraction and observer inequality\n"
        "        Resource stock, reserve welfare, wealth Gini",
        va="top",
        fontsize=11,
    )
    # Small schematics show that equal and random share propensities but not
    # necessarily realized observer counts; they are not experimental estimates.
    profiles = (
        (0.5, 0.5, 0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5, 0.5, 0.5),
        (0.2, 0.4, 0.5, 0.6, 0.8),
        (0.1, 0.15, 0.2, 0.25, 0.75),
        (0.25, 0.7, 0.8, 0.85, 0.9),
    )
    axes[1, 0].text(0.02, 0.40, "Schematic propensity patterns", fontsize=9)
    for index, values in enumerate(profiles):
        left = 0.02 + 0.19 * index
        for offset, value in enumerate(values):
            axes[1, 0].add_patch(
                Rectangle(
                    (left + 0.025 * offset, 0.12),
                    0.018,
                    0.20 * value,
                    transform=axes[1, 0].transAxes,
                    color="#297d63",
                )
            )
        axes[1, 0].text(
            left + 0.05,
            0.065,
            "ERNLH"[index],
            ha="center",
            fontsize=8,
            transform=axes[1, 0].transAxes,
        )
    axes[1, 1].text(0.02, 0.39, "Example directed attention links", fontsize=9)
    positions = ((0.16, 0.18), (0.35, 0.29), (0.52, 0.13), (0.72, 0.22))
    for x, y in positions:
        axes[1, 1].add_patch(
            Circle(
                (x, y),
                0.033,
                transform=axes[1, 1].transAxes,
                facecolor="#2e8069",
                edgecolor="white",
                zorder=3,
            )
        )
    for start, end in ((0, 3), (1, 3), (2, 3)):
        axes[1, 1].add_patch(
            FancyArrowPatch(
                positions[start],
                positions[end],
                transform=axes[1, 1].transAxes,
                arrowstyle="->",
                mutation_scale=12,
                color="#5d6770",
                linewidth=1.3,
                zorder=2,
            )
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
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True, sharey=True)
    agents = tables["initial_visibility_agents"]
    # Reused ecologies/dynamics share graphs; one row per profile, replicate and agent.
    unique = {
        (r["visibility_profile"], r["population"], r["replicate"], r["agent"]): r for r in agents
    }
    profiles = [profile for profile in PROFILES if any(key[0] == profile for key in unique)]
    distributions = [
        [int(r["initial_visibility_count"]) for key, r in unique.items() if key[0] == profile]
        for profile in profiles
    ]
    if not distributions or any(not values for values in distributions):
        raise ValueError("Missing initial observer counts for a visibility profile")
    axes[0].boxplot(
        distributions,
        positions=range(len(profiles)),
        orientation="horizontal",
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#DCE8EB", "edgecolor": DYNAMICS_COLORS["fixed"]},
        medianprops={"color": INK, "linewidth": 1.6},
        whiskerprops={"color": DYNAMICS_COLORS["fixed"]},
        capprops={"color": DYNAMICS_COLORS["fixed"]},
    )
    summaries = tables["initial_visibility_summary"]
    for ax, metric in ((axes[1], "visibility_gini"), (axes[2], "zero_visibility_fraction")):
        for index, profile in enumerate(profiles):
            row = next(
                (r for r in summaries if r["profile"] == profile and r["metric"] == metric), None
            )
            if row is None:
                raise ValueError(f"Missing initial visibility summary: {profile}, {metric}")
            estimate, low, high = (float(row[key]) for key in ("mean", "low", "high"))
            ax.plot([low, high], [index, index], color=DYNAMICS_COLORS["fixed"], lw=1.6)
            ax.scatter([estimate], [index], color=DYNAMICS_COLORS["fixed"], s=34, zorder=3)
    for ax in axes:
        ax.set_yticks(range(len(profiles)), [LABELS[p] for p in profiles])
        ax.set_ylim(len(profiles) - 0.5, -0.5)
        ax.grid(axis="x", color=GRID, lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_xlabel("Actual observers per agent")
    axes[1].set_xlabel("Observer-count Gini")
    axes[2].set_xlabel("Fraction with no observers")
    axes[0].set_title("Agent distributions", loc="left", fontweight="bold")
    axes[1].set_title("Inequality", loc="left", fontweight="bold")
    axes[2].set_title("Unseen agents", loc="left", fontweight="bold")
    fig.suptitle("Initial visibility: who can be seen?", fontsize=15, color=INK)
    _save(
        fig,
        directory,
        "02_initial_visibility_manipulation",
        f"Boxplots summarize agent observer counts; Gini and unseen fraction show replicate means with 95% intervals (n≤{_n(summaries)})",
    )


def _factor_grid(directory, table, metrics, filename, title, caption="", *, ecologies):

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
                        _point(
                            ax,
                            point,
                            index + offset,
                            color=COLORS[dynamic],
                            marker=DYNAMICS_MARKERS[dynamic],
                        )
            if row_index == len(metrics) - 1:
                ax.set_xticks(range(5), [LABELS[p] for p in PROFILES], rotation=25, ha="right")
            else:
                ax.set_xticks(range(5), [])
            ax.set_xlim(-0.5, 4.5)
            if col == 0:
                ax.set_ylabel(label)
            if row_index == 0:
                ax.set_title(LABELS[ecology])
    handles = [
        plt.Line2D(
            [0], [0], marker=DYNAMICS_MARKERS[name], linestyle="", color=color, label=LABELS[name]
        )
        for name, color in COLORS.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.085))
    fig.suptitle(title, y=1.15)
    _save(fig, directory, filename, caption)


def _paired_consequences(directory, table, ecologies):
    metrics = (
        ("resource_fraction", "Resource / capacity"),
        ("reserve_welfare", "Reserve welfare"),
        ("final_wealth_gini", "Final wealth Gini"),
    )
    fig, axes = plt.subplots(
        3, 3, figsize=(15, 10), constrained_layout=True, squeeze=False, sharey="row"
    )
    for row_index, (metric, label) in enumerate(metrics):
        for col, ecology in enumerate(ecologies):
            ax = axes[row_index, col]
            ax.axhline(0, color=INK, lw=0.85)
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
                    _point(ax, point, index, color="#6654A4")
            if row_index == len(metrics) - 1:
                ax.set_xticks(range(5), [LABELS[p] for p in PROFILES], rotation=25, ha="right")
            else:
                ax.set_xticks(range(5), [])
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


def _trajectory(directory, table, metric, filename, ylabel, ecologies):
    if not table:
        return
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
                        linestyle="-" if dynamic == "fixed" else "--",
                        label=LABELS[dynamic],
                    )
            if col == 0:
                ax.set_ylabel(f"{LABELS[profile]}\n{ylabel}")
            if row_index == 0:
                ax.set_title(LABELS[ecology])
            if row_index == 4:
                ax.set_xlabel("Training step")
    handles = [
        plt.Line2D(
            [0], [0], color=color, linestyle="-" if name == "fixed" else "--", label=LABELS[name]
        )
        for name, color in COLORS.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"Training trajectories · {ylabel}", y=1.14)
    _save(
        fig,
        directory,
        filename,
        f"Training checkpoints; descriptive means across independent replicates (n≤{_n(table)})",
    )


def _gini_decomposition(directory, table, ecologies):
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
        ecologies=ecologies,
    )


def save_visibility_figures(
    directory,
    tables,
    *,
    ecologies=("uniform_high", "patchy_high", "split_high_low"),
    width=10,
    height=10,
    seed=20261002,
):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
        }
    )
    _design(directory, ecologies, width=width, height=height, seed=seed)
    _manipulation(directory, tables)
    _factor_grid(
        directory,
        tables["primary_window_summary"],
        (
            ("social_perception_error", "Local-view error"),
            ("majority_mismatch_rate", "Majority mismatch (non-ties)"),
            ("majority_tie_rate", "Majority tie rate"),
        ),
        "03_population_perception",
        "RQ1 · Local views and population behavior",
        caption=f"Final min(1000,T) training steps; replicate means with 95% intervals (n≤{_n(tables['primary_window_summary'])})",
        ecologies=ecologies,
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
        ecologies=ecologies,
    )
    _paired_consequences(directory, tables["adaptive_fixed_summary"], ecologies)
    _gini_decomposition(directory, tables["outcome_summary"], ecologies)
    for metric, filename, label in (
        ("visibility_gini", "S2_visibility_gini_trajectories", "Visibility Gini"),
        ("low_extraction_rate", "S2_low_extraction_trajectories", "Low-extraction share"),
    ):
        _trajectory(
            directory, tables["training_trajectory_summary"], metric, filename, label, ecologies
        )
