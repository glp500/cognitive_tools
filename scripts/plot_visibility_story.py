"""Render five narrative Stage-5 figures from a completed visibility analysis.

These exploratory publication candidates do not replace the frozen study figures
or recompute the planned inferential contrasts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from cognitive_tools.scenarios import build_environment_maps
from cognitive_tools.visibility import PROFILES
from cognitive_tools.visibility_figures import LABELS

ECOLOGIES = ("uniform_high", "patchy_high", "split_high_low")
ECOLOGY_COLORS = {
    "uniform_high": "#287d78",
    "patchy_high": "#ad6534",
    "split_high_low": "#69599a",
    "balanced_uniform": "#287d78",
    "balanced_dispersed": "#ad6534",
    "balanced_segregated": "#69599a",
}
ECOLOGY_MARKERS = {
    "uniform_high": "o",
    "patchy_high": "s",
    "split_high_low": "^",
    "balanced_uniform": "o",
    "balanced_dispersed": "s",
    "balanced_segregated": "^",
}
INK = "#24323b"
MUTED = "#5b6972"
INITIAL = "#317b89"
FINAL = "#c6683d"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty figure input: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(fig, output: Path, stem: str, note: str) -> None:
    for extension in ("pdf", "svg", "png"):
        options = {"dpi": 300} if extension == "png" else {}
        fig.savefig(output / f"{stem}.{extension}", bbox_inches="tight", pad_inches=0.12, **options)
    (output / f"{stem}.caption.txt").write_text(note + "\n")
    plt.close(fig)


def _point(ax, row: dict[str, str], y: float, *, color: str, marker: str = "o") -> None:
    estimate, low, high = (float(row[key]) for key in ("mean", "low", "high"))
    if not low <= estimate <= high:
        raise ValueError(f"Invalid confidence interval: {row}")
    ax.plot([low, high], [y, y], color=color, lw=1.5, solid_capstyle="round")
    ax.scatter([estimate], [y], color=color, marker=marker, s=32, zorder=3)


def _clean_axis(ax, *, zero: bool = False) -> None:
    if zero:
        ax.axvline(0, color=INK, lw=0.9, zorder=0)
    ax.grid(axis="x", color="#e4e9ea", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def _inputs(root: Path) -> tuple[list[dict[str, str]], dict, dict]:
    manifest = json.loads((root / "analysis_manifest.json").read_text())
    if manifest.get("profile") != "visibility":
        raise ValueError("Expected a Stage-5 visibility analysis")
    runs = manifest["inputs"]
    social = next((run for run in runs if run["treatment"] != "B0"), None)
    if social is None:
        raise ValueError("No social runs in analysis manifest")
    config = json.loads((Path(social["run_path"]) / "config.json").read_text())
    if len(runs) != 11:
        raise ValueError(f"Expected 11 Stage-5 runs, found {len(runs)}")
    for run in runs:
        if _sha256(Path(run["run_path"]) / "config.json") != run["config_sha256"]:
            raise ValueError(f"Source configuration changed after analysis: {run['run_id']}")
    if config["populations"] != [64] or int(config["attention_k"]) != 4:
        raise ValueError("Story figures require the N=64, k=4 scientific pilot or full study")
    for run in runs:
        other = json.loads((Path(run["run_path"]) / "config.json").read_text())
        if other["scenarios"] != config["scenarios"] or other.get(
            "environment_design"
        ) != config.get("environment_design"):
            raise ValueError("Story inputs mix environment designs or scenario sets")
    return runs, manifest, config


def _ground(output: Path, config: dict, note: str) -> None:
    fig = plt.figure(figsize=(11.5, 5.6), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[3.5, 1.3])
    maps = []
    capacities = []
    for index, ecology in enumerate(ECOLOGIES):
        ax = fig.add_subplot(grid[0, index])
        capacity, _, _, regions = build_environment_maps(
            ecology,
            width=int(config["width"]),
            height=int(config["height"]),
            seed=int(config["seed"]),
        )
        image = ax.imshow(capacity, cmap="YlGn", vmin=0, vmax=1, interpolation="nearest")
        maps.append(image)
        capacities.append(capacity)
        ax.set_title(LABELS[ecology], loc="left", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"Replicate 0 · total capacity {capacity.sum():.1f}", color=MUTED)
        if ecology in ("split_high_low", "balanced_segregated"):
            labels = (
                (("left_high_patchy", "High region"), ("right_low_fragmented", "Low region"))
                if ecology == "split_high_low"
                else (("high_capacity", "High capacity"), ("low_capacity", "Low capacity"))
            )
            for region, label in labels:
                positions = [
                    (y, x)
                    for y, row in enumerate(regions)
                    for x, value in enumerate(row)
                    if value == region
                ]
                if positions:
                    y = mean(position[0] for position in positions)
                    x = mean(position[1] for position in positions)
                    ax.text(
                        x,
                        y,
                        label,
                        ha="center",
                        va="center",
                        fontsize=10,
                        color=INK,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
                    )
    if ECOLOGIES[0] == "balanced_uniform":
        if not np.allclose([capacity.sum() for capacity in capacities], 75.0) or not np.array_equal(
            np.sort(capacities[1].ravel()), np.sort(capacities[2].ravel())
        ):
            raise ValueError("Balanced story maps fail matched-capacity validation")
    fig.colorbar(
        maps[0], ax=fig.axes[:3], location="right", shrink=0.8, label="Local resource capacity"
    )
    process = fig.add_subplot(grid[1, :])
    process.axis("off")
    stages = (
        "Observe four sources",
        "Choose extraction",
        "Resources renew",
        "Update view (adaptive only)",
    )
    for index, stage in enumerate(stages):
        x = 0.05 + index * 0.245
        process.text(
            x,
            0.55,
            stage,
            transform=process.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="normal",
            color=INK,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#edf2ef", "edgecolor": "#c9d4d0"},
        )
        if index < 3:
            process.annotate(
                "",
                (x + 0.19, 0.55),
                (x + 0.105, 0.55),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": MUTED},
            )
    fig.suptitle(
        "1  ·  Same total capacity, different resource geography"
        if ECOLOGIES[0] == "balanced_uniform"
        else "1  ·  Agents begin on unequal resource landscapes",
        x=0.02,
        ha="left",
        fontsize=15,
        color=INK,
    )
    _save(
        fig,
        output,
        "01_uneven_ground",
        f"{note}; capacity maps show configured replicate 0, not an across-replicate mean.",
    )


def _attention(output: Path, runs: list[dict[str, str]], root: Path, note: str) -> None:
    initial_rows = _read(root / "data" / "initial_visibility_agents.csv")
    initial = defaultdict(list)
    for row in initial_rows:
        initial[(row["visibility_profile"], int(row["replicate"]))].append(
            int(row["initial_visibility_count"])
        )
    final = defaultdict(list)
    for run in runs:
        if run["treatment"] == "B0" or not run["treatment"].endswith("adaptive_bounded"):
            continue
        profile = run["treatment"].removesuffix("_adaptive_bounded")
        rows = _read(Path(run["run_path"]) / "data" / "agent_social_summary.csv")
        for row in rows:
            final[(profile, row["scenario"], int(row["replicate"]))].append(
                int(row["final_visibility_degree"])
            )
    expected = len(PROFILES) * int(runs[0]["replicates"])
    if len(initial) != expected or len(final) != expected * len(ECOLOGIES):
        raise ValueError("Incomplete initial or terminal attention distributions")
    for counts in list(initial.values()) + list(final.values()):
        if len(counts) != 64 or sum(counts) != 256:
            raise ValueError("Observer counts do not cover 64 agents and 256 attention links")
    fig, axes = plt.subplots(
        5,
        2,
        figsize=(11.5, 8.3),
        sharex="col",
        layout="constrained",
        gridspec_kw={"width_ratios": [2.3, 1]},
    )
    for index, profile in enumerate(PROFILES):
        ax, share_ax = axes[index]
        starts = [count for key, values in initial.items() if key[0] == profile for count in values]
        ends = [count for key, values in final.items() if key[0] == profile for count in values]
        bins = range(0, max(max(starts), max(ends)) + 1)
        start_fraction = [starts.count(value) / len(starts) for value in bins]
        end_fraction = [ends.count(value) / len(ends) for value in bins]
        ax.bar(
            [value - 0.19 for value in bins],
            start_fraction,
            width=0.38,
            color=INITIAL,
            label="Initial",
        )
        ax.bar(
            [value + 0.19 for value in bins],
            end_fraction,
            width=0.38,
            color=FINAL,
            label="Adaptive terminal",
        )
        ax.set_ylabel(LABELS[profile], rotation=0, ha="right", va="center", labelpad=12)
        ax.set_ylim(0, 1.05 if profile == "equal" else 0.32)
        ax.spines[["top", "right"]].set_visible(False)
        first_shares = [
            sum(sorted(values, reverse=True)[:6]) / 256
            for key, values in initial.items()
            if key[0] == profile
        ]
        initial_mean = mean(first_shares)
        share_ax.scatter([initial_mean], [0], color=INITIAL, s=42, marker="s", zorder=3)
        for ecology_index, ecology in enumerate(ECOLOGIES):
            shares = [
                sum(sorted(values, reverse=True)[:6]) / 256
                for key, values in final.items()
                if key[0] == profile and key[1] == ecology
            ]
            share_ax.plot(
                [initial_mean, mean(shares)], [0, ecology_index + 1], color="#c9cdd0", lw=1
            )
            share_ax.plot(
                [min(shares), max(shares)],
                [ecology_index + 1] * 2,
                color=ECOLOGY_COLORS[ecology],
                lw=2,
            )
            share_ax.scatter(
                [mean(shares)],
                [ecology_index + 1],
                color=ECOLOGY_COLORS[ecology],
                marker=ECOLOGY_MARKERS[ecology],
                s=34,
                zorder=3,
            )
        share_ax.set_yticks([])
        share_ax.set_ylim(3.5, -0.5)
        share_ax.set_xlim(0.07, 0.35)
        _clean_axis(share_ax)
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper right")
    axes[0, 0].set_title("Fraction of sources at each observer count", loc="left")
    axes[0, 1].set_title("Top-six attention share", loc="left")
    axes[-1, 0].set_xticks((0, 4, 8, 12, 16))
    axes[-1, 0].set_xlabel("Number of observers per source")
    axes[-1, 1].set_xlabel("Share of 256 links held by top six sources")
    legend = [
        Line2D(
            [0],
            [0],
            color=ECOLOGY_COLORS[e],
            marker=ECOLOGY_MARKERS[e],
            linestyle="",
            label=LABELS[e],
        )
        for e in ECOLOGIES
    ]
    fig.legend(
        handles=legend, loc="upper right", frameon=False, ncol=3, bbox_to_anchor=(0.96, 1.005)
    )
    fig.suptitle("2  ·  Who gets seen?", x=0.02, ha="left", fontsize=15, color=INK)
    _save(
        fig,
        output,
        "02_who_gets_seen",
        f"{note}; initial graphs deduplicated; terminal adaptive runs pool ecologies. Top six are re-ranked at each snapshot; horizontal lines show replicate ranges.",
    )


def _perception(output: Path, root: Path, note: str) -> None:
    h1 = [
        row
        for row in _read(root / "data" / "ecology_contrast_summary.csv")
        if row["metric"] == "social_perception_error"
    ]
    h2 = [
        row
        for row in _read(root / "data" / "visibility_profile_contrast_summary.csv")
        if row["metric"] == "social_perception_error"
    ]
    balanced = ECOLOGIES[0] == "balanced_uniform"
    h1_expected = (3 if balanced else 2) * len(PROFILES) * 2
    h2_expected = 4 * len(ECOLOGIES)
    if len(h1) != h1_expected or len(h2) != h2_expected:
        raise ValueError(f"Incomplete H1/H2 contrasts: H1={len(h1)}, H2={len(h2)}")
    h1_by_key = {(row["comparison"], row["profile"], row["dynamics"]): row for row in h1}
    h2_by_key = {(row["comparison"], row["scenario"]): row for row in h2}
    comparisons = (
        ("random-equal", "Random − equal"),
        ("normal_centered-random", "Normal − random"),
        ("low_propensity_majority-random", "Low majority − random"),
        ("high_propensity_majority-random", "High majority − random"),
    )
    extent = 1.12 * max(abs(float(row[key])) for row in h1 + h2 for key in ("low", "high"))
    fig = plt.figure(figsize=(14.2, 6.6), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[7, 1], width_ratios=[1.2, 1])
    axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]))
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.axis("off")
    ax = axes[0]
    labels = [
        f"{LABELS[profile]} · {LABELS[dynamics]}"
        for profile in PROFILES
        for dynamics in ("fixed", "adaptive_bounded")
    ]
    h1_styles = (
        (
            (
                "balanced_dispersed-balanced_uniform",
                ECOLOGY_COLORS["balanced_dispersed"],
                "o",
                "Dispersed − uniform",
            ),
            (
                "balanced_segregated-balanced_dispersed",
                ECOLOGY_COLORS["balanced_segregated"],
                "s",
                "Segregated − dispersed (primary)",
            ),
            ("balanced_segregated-balanced_uniform", INK, "^", "Segregated − uniform"),
        )
        if balanced
        else (
            ("patchy_high-uniform_high", ECOLOGY_COLORS["patchy_high"], "o", "Patchy − uniform"),
            (
                "split_high_low-uniform_high",
                ECOLOGY_COLORS["split_high_low"],
                "s",
                "Split − uniform",
            ),
        )
    )
    for y, (profile, dynamics) in enumerate(
        (p, d) for p in PROFILES for d in ("fixed", "adaptive_bounded")
    ):
        offsets = (-0.20, 0.0, 0.20) if balanced else (-0.15, 0.15)
        for offset, (comparison, color, marker, _) in zip(offsets, h1_styles):
            _point(
                ax,
                h1_by_key[(comparison, profile, dynamics)],
                y + offset,
                color=color,
                marker=marker,
            )
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlim(-extent, extent)
    ax.set_xlabel("Difference in mean absolute perception error")
    ax.set_title(
        "H1 · Resource geography\nat matched capacity"
        if balanced
        else "H1 · Ecological setting\nversus uniform high",
        loc="left",
        fontweight="bold",
    )
    _clean_axis(ax, zero=True)
    ax = axes[1]
    offsets = (-0.18, 0.0, 0.18)
    for y, (comparison, _) in enumerate(comparisons):
        for offset, ecology in zip(offsets, ECOLOGIES):
            _point(
                ax,
                h2_by_key[(comparison, ecology)],
                y + offset,
                color=ECOLOGY_COLORS[ecology],
                marker=ECOLOGY_MARKERS[ecology],
            )
    ax.set_yticks(range(4), [label for _, label in comparisons])
    ax.set_ylim(3.5, -0.5)
    ax.set_xlim(-extent, extent)
    ax.set_xlabel("Difference in mean absolute perception error")
    ax.set_title("H2 · Initial attention profile\non fixed networks", loc="left", fontweight="bold")
    _clean_axis(ax, zero=True)
    contrast_handles = [
        Line2D([0], [0], color=color, marker=marker, linestyle="", label=label)
        for _, color, marker, label in h1_styles
    ]
    ecology_handles = [
        Line2D(
            [0],
            [0],
            marker=ECOLOGY_MARKERS[e],
            color=ECOLOGY_COLORS[e],
            linestyle="",
            label=LABELS[e],
        )
        for e in ECOLOGIES
    ]
    handles = (
        [item for pair in zip(contrast_handles, ecology_handles) for item in pair]
        if balanced
        else contrast_handles + ecology_handles
    )
    legend_ax.legend(
        handles=handles, frameon=False, loc="center", ncol=3 if balanced else 2, fontsize=9.5
    )
    fig.suptitle(
        "3  ·  Different local views of the same population",
        x=0.02,
        ha="left",
        fontsize=15,
        color=INK,
    )
    _save(
        fig,
        output,
        "03_different_windows",
        f"{note}; planned paired H1/H2 contrasts, pointwise replicate-bootstrap 95% intervals. Zero means no difference.",
    )


def _evaluation_points(runs: list[dict[str, str]], root: Path) -> list[dict]:
    balanced = ECOLOGIES[0] == "balanced_uniform"
    grouped = defaultdict(list)
    for run in runs:
        for row in _read(Path(run["run_path"]) / "data" / "evaluation_summary.csv"):
            if row["strategy"] == "q_learning" and row["evaluation_mode"] == "fresh_reset":
                grouped[(run["run_id"], row["scenario"])].append(row)
    summary = _read(root / "data" / "outcome_summary.csv")
    expected = {
        (row["scenario"], row["profile"], row["dynamics"], row["metric"]): float(row["mean"])
        for row in summary
        if row["metric"] in ("resource_fraction", "reserve_welfare")
    }
    points = []
    for run in runs:
        if run["treatment"] == "B0":
            profile, dynamics = "none", "none"
        elif run["treatment"].endswith("_adaptive_bounded"):
            profile = run["treatment"].removesuffix("_adaptive_bounded")
            dynamics = "adaptive_bounded"
        elif run["treatment"].endswith("_fixed"):
            profile = run["treatment"].removesuffix("_fixed")
            dynamics = "fixed"
        else:
            raise ValueError(f"Unknown Stage-5 treatment: {run['treatment']}")
        for ecology in ECOLOGIES:
            rows = grouped[(run["run_id"], ecology)]
            if len(rows) != int(run["replicates"]):
                raise ValueError(f"Incomplete fresh-evaluation rows: {run['run_id']} {ecology}")
            point = {
                "ecology": ecology,
                "profile": profile,
                "dynamics": dynamics,
                "stock": mean(float(row["eval_mean_total_resource"]) for row in rows),
                "fraction": mean(
                    float(row["eval_mean_capacity_weighted_resource_fraction"])
                    if balanced
                    else float(row["eval_mean_mean_resource_fraction"])
                    for row in rows
                ),
                "local_fraction": mean(
                    float(row["eval_mean_mean_resource_fraction"]) for row in rows
                ),
                "welfare": mean(float(row["eval_mean_mean_reserve_welfare"]) for row in rows),
            }
            for field, metric in (
                ("fraction", "resource_fraction"),
                ("welfare", "reserve_welfare"),
            ):
                if abs(point[field] - expected[(ecology, profile, dynamics, metric)]) > 1e-9:
                    raise ValueError(
                        f"Fresh-evaluation mean disagrees with analysis: {run['run_id']} {ecology} {metric}"
                    )
            points.append(point)
    if len(points) != 33:
        raise ValueError(f"Expected 33 ecology/treatment means, found {len(points)}")
    return points


def _regional_welfare(runs: list[dict[str, str]]) -> list[dict]:
    result = []
    balanced = ECOLOGIES[0] == "balanced_uniform"
    scenario = "balanced_segregated" if balanced else "split_high_low"
    high_region = "high_capacity" if balanced else "left_high_patchy"
    low_region = "low_capacity" if balanced else "right_low_fragmented"
    for run in runs:
        regions = defaultdict(list)
        for row in _read(Path(run["run_path"]) / "data" / "agent_social_summary.csv"):
            if row["scenario"] == scenario:
                regions[(int(row["replicate"]), row["region"])].append(
                    float(row["final_reserve_welfare"])
                )
        expected = {
            (rep, region)
            for rep in range(int(run["replicates"]))
            for region in (high_region, low_region)
        }
        if set(regions) != expected:
            raise ValueError(f"Incomplete regional training welfare: {run['run_id']}")
        result.append(
            {
                "run_id": run["run_id"],
                "baseline": run["treatment"] == "B0",
                "high": mean(
                    mean(regions[(rep, high_region)]) for rep in range(int(run["replicates"]))
                ),
                "low": mean(
                    mean(regions[(rep, low_region)]) for rep in range(int(run["replicates"]))
                ),
            }
        )
    return result


def _stock_welfare(output: Path, runs: list[dict[str, str]], root: Path, note: str) -> None:
    points = _evaluation_points(runs, root)
    regions = _regional_welfare(runs)
    balanced = ECOLOGIES[0] == "balanced_uniform"
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.4, 4.7),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]},
    )
    markers = {"none": "D", "fixed": "o", "adaptive_bounded": "^"}
    for point in points:
        color = ECOLOGY_COLORS[point["ecology"]]
        for ax, x_key in zip(
            axes[:2], ("fraction", "local_fraction") if balanced else ("stock", "fraction")
        ):
            ax.scatter(
                point[x_key],
                point["welfare"],
                marker=markers[point["dynamics"]],
                s=95 if point["dynamics"] == "none" else 42,
                facecolors="white" if point["dynamics"] == "none" else color,
                edgecolors=color if point["dynamics"] == "none" else "white",
                linewidths=1.5 if point["dynamics"] == "none" else 0.5,
                alpha=0.95 if point["dynamics"] == "none" else 0.73,
            )
    axes[0].set_xlabel(
        "Total resource / total capacity · fresh evaluation"
        if balanced
        else "Mean total resource · fresh evaluation"
    )
    axes[1].set_xlabel(
        "Mean local fill fraction · fresh evaluation"
        if balanced
        else "Mean resource / capacity · fresh evaluation"
    )
    axes[0].set_ylabel("Mean reserve welfare · fresh evaluation")
    axes[0].set_title(
        "Collective stock" if balanced else "Absolute stock", loc="left", fontweight="bold"
    )
    axes[1].set_title(
        "Local fill" if balanced else "Remaining fraction", loc="left", fontweight="bold"
    )
    welfare_values = [point["welfare"] for point in points]
    welfare_pad = max(max(welfare_values) - min(welfare_values), 0.10) * 0.2
    welfare_limits = (
        max(0, min(welfare_values) - welfare_pad),
        min(1.02, max(welfare_values) + welfare_pad),
    )
    for ax in axes[:2]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e4e9ea", lw=0.6, zorder=0)
        ax.set_ylim(*welfare_limits)
    axes[1].set_yticklabels([])
    ax = axes[2]
    for item in regions:
        color = INK if item["baseline"] else "#b9c1c5"
        ax.plot(
            [0, 1],
            [item["high"], item["low"]],
            color=color,
            lw=2.4 if item["baseline"] else 1.2,
            alpha=1 if item["baseline"] else 0.8,
            zorder=3 if item["baseline"] else 1,
        )
        ax.scatter(
            [0, 1],
            [item["high"], item["low"]],
            color=color,
            s=32 if item["baseline"] else 10,
            zorder=3 if item["baseline"] else 1,
        )
    ax.set_xticks((0, 1), ("High region", "Low region"))
    ax.set_xlim(-0.22, 1.22)
    regional_values = [value for item in regions for value in (item["high"], item["low"])]
    regional_pad = max(max(regional_values) - min(regional_values), 0.10) * 0.2
    ax.set_ylim(
        max(0, min(regional_values) - regional_pad), min(1.02, max(regional_values) + regional_pad)
    )
    ax.set_title(
        "Segregated-world regional gap\ntraining end"
        if balanced
        else "Split-world regional gap\ntraining end",
        loc="left",
        fontweight="bold",
    )
    ax.set_ylabel("Mean reserve welfare · training end")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e9ea", lw=0.6)
    ecology_handles = [
        Line2D([0], [0], color=ECOLOGY_COLORS[e], marker="o", linestyle="", label=LABELS[e])
        for e in ECOLOGIES
    ]
    treatment_handles = [
        Line2D(
            [0],
            [0],
            color=INK,
            marker=markers[d],
            linestyle="",
            markerfacecolor="white" if d == "none" else INK,
            label=label,
        )
        for d, label in (("none", "B0"), ("fixed", "Fixed"), ("adaptive_bounded", "Adaptive"))
    ]
    fig.legend(
        handles=ecology_handles + treatment_handles,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 1.045),
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "4  ·  Resource stock and welfare tell different parts of the story",
        x=0.02,
        y=1.15,
        ha="left",
        fontsize=15,
        color=INK,
    )
    _save(
        fig,
        output,
        "04_stock_and_wellbeing",
        f"{note}; left: 33 fresh-evaluation condition means. Right: 11 treatment means; B0 dark, social treatments grey. Regional welfare is from training end. No pooled regression.",
    )


def _adaptation(output: Path, root: Path, note: str) -> None:
    rows = _read(root / "data" / "adaptive_fixed_summary.csv")
    metrics = (
        ("visibility_gini", "Who is seen?", "Visibility Gini"),
        ("social_perception_error", "What is perceived?", "Mean absolute error"),
        ("low_extraction_rate", "What does the group do?", "Low-extraction share"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 5), layout="constrained")
    for ax, (metric, question, label) in zip(axes, metrics):
        relevant = [row for row in rows if row["metric"] == metric]
        if len(relevant) != len(PROFILES) * len(ECOLOGIES):
            raise ValueError(f"Incomplete adaptive-fixed contrast set: {metric}")
        by_key = {(row["scenario"], row["profile"]): row for row in relevant}
        extent = 1.12 * max(abs(float(row[key])) for row in relevant for key in ("low", "high"))
        for index, profile in enumerate(PROFILES):
            for offset, ecology in zip((-0.18, 0, 0.18), ECOLOGIES):
                _point(
                    ax,
                    by_key[(ecology, profile)],
                    index + offset,
                    color=ECOLOGY_COLORS[ecology],
                    marker=ECOLOGY_MARKERS[ecology],
                )
        ax.set_xlim(
            min(-0.02, min(float(row["low"]) for row in relevant))
            if metric == "visibility_gini"
            else -extent,
            max(float(row["high"]) for row in relevant) * 1.08
            if metric == "visibility_gini"
            else extent,
        )
        ax.set_ylim(len(PROFILES) - 0.5, -0.5)
        ax.set_yticks(range(len(PROFILES)), [LABELS[p] for p in PROFILES])
        ax.set_xlabel(f"Adaptive − fixed · {label}")
        ax.set_title(question, loc="left", fontweight="bold")
        _clean_axis(ax, zero=True)
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=ECOLOGY_COLORS[e],
                marker=ECOLOGY_MARKERS[e],
                linestyle="",
                label=LABELS[e],
            )
            for e in ECOLOGIES
        ],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
    )
    fig.suptitle(
        "5  ·  What changed when agents adapted their attention?",
        x=0.02,
        y=1.22,
        ha="left",
        fontsize=15,
        color=INK,
    )
    _save(
        fig,
        output,
        "05_adaptation_pathway",
        f"{note}; all 15 paired ecology/profile cells per outcome; pointwise replicate-bootstrap 95% intervals. Zero means no difference.",
    )


def main() -> None:
    global ECOLOGIES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.analysis_dir.resolve()
    output = (args.output or root / "story_candidates").resolve()
    try:
        runs, analysis, config = _inputs(root)
        ecologies = tuple(config["scenarios"])
        if ecologies not in (
            ("uniform_high", "patchy_high", "split_high_low"),
            ("balanced_uniform", "balanced_dispersed", "balanced_segregated"),
        ):
            raise ValueError(f"Unsupported story ecology set: {ecologies}")
        ECOLOGIES = ecologies
    except Exception as error:
        print(
            json.dumps(
                {
                    "event": "story_input_failed",
                    "analysis": root.name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            ),
            flush=True,
        )
        raise
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "story_manifest.json"
    report_path.unlink(missing_ok=True)
    identity = analysis.get("analysis_name", root.name)
    scope = (
        "Exploratory pilot"
        if "pilot" in identity
        else ("Mechanical smoke test" if "smoke" in identity else "Stage-5 study")
    )
    note = (
        f"{scope} · N={config['populations'][0]}, k={config['attention_k']}, "
        f"T={config['training_steps']:,}; n={config['replicates']} independent replicates"
    )
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 10.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
        }
    )
    figures = (
        ("01_uneven_ground", lambda: _ground(output, config, note)),
        ("02_who_gets_seen", lambda: _attention(output, runs, root, note)),
        ("03_different_windows", lambda: _perception(output, root, note)),
        ("04_stock_and_wellbeing", lambda: _stock_welfare(output, runs, root, note)),
        ("05_adaptation_pathway", lambda: _adaptation(output, root, note)),
    )
    print(
        json.dumps(
            {
                "event": "story_start",
                "analysis": identity,
                "figures": len(figures),
                "runs": len(runs),
                "replicates_per_run": config["replicates"],
            }
        ),
        flush=True,
    )
    completed = []
    for stem, render in figures:
        started = time.perf_counter()
        try:
            render()
        except Exception as error:
            print(
                json.dumps(
                    {
                        "event": "story_figure_failed",
                        "analysis": identity,
                        "figure": stem,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                ),
                flush=True,
            )
            raise
        completed.append(stem)
        print(
            json.dumps(
                {
                    "event": "story_figure_saved",
                    "analysis": identity,
                    "figure": stem,
                    "duration_seconds": round(time.perf_counter() - started, 3),
                }
            ),
            flush=True,
        )
    analysis_files = (
        "initial_visibility_agents.csv",
        "ecology_contrast_summary.csv",
        "visibility_profile_contrast_summary.csv",
        "outcome_summary.csv",
        "adaptive_fixed_summary.csv",
    )
    try:
        source_hashes = {
            "analysis/analysis_manifest.json": _sha256(root / "analysis_manifest.json")
        }
        source_hashes.update(
            {f"analysis/data/{name}": _sha256(root / "data" / name) for name in analysis_files}
        )
        for run in runs:
            for name in (
                "config.json",
                "data/evaluation_summary.csv",
                "data/agent_social_summary.csv",
            ):
                digest = _sha256(Path(run["run_path"]) / name)
                if name == "config.json" and digest != run["config_sha256"]:
                    raise ValueError(
                        f"Source configuration changed after analysis: {run['run_id']}"
                    )
                source_hashes[f"{run['run_id']}/{name}"] = digest
    except Exception as error:
        print(
            json.dumps(
                {
                    "event": "story_provenance_failed",
                    "analysis": identity,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            ),
            flush=True,
        )
        raise
    report = {
        "schema_version": 1,
        "analysis": identity,
        "story_source_sha256": _sha256(Path(__file__)),
        "environment_design": config.get("environment_design", "legacy"),
        "ecologies": ECOLOGIES,
        "analysis_source_sha256": analysis.get("analysis_source_sha256"),
        "study_runs": [
            {"run_id": run["run_id"], "config_sha256": run["config_sha256"]} for run in runs
        ],
        "figures": completed,
        "formats": ["pdf", "svg", "png", "caption.txt"],
        "replicates_per_run": config["replicates"],
        "population": config["populations"][0],
        "attention_k": config["attention_k"],
        "training_steps": config["training_steps"],
        "evaluation_mode": "fresh_reset",
        "strategy": "q_learning",
        "source_tables": {
            "01_uneven_ground": ["source run config.json", "cognitive_tools.scenarios"],
            "02_who_gets_seen": [
                "data/initial_visibility_agents.csv",
                "adaptive runs/data/agent_social_summary.csv",
            ],
            "03_different_windows": [
                "data/ecology_contrast_summary.csv",
                "data/visibility_profile_contrast_summary.csv",
            ],
            "04_stock_and_wellbeing": [
                "all runs/data/evaluation_summary.csv",
                "all runs/data/agent_social_summary.csv",
                "data/outcome_summary.csv",
            ],
            "05_adaptation_pathway": ["data/adaptive_fixed_summary.csv"],
        },
        "input_sha256": source_hashes,
        "checks_passed": {
            "run_count": 11,
            "ecology_treatment_means": 33,
            "h1_contrast_cells": 30 if ECOLOGIES[0] == "balanced_uniform" else 20,
            "h2_contrast_cells": 12,
            "h3_cells_per_outcome": 15,
            "observers_per_snapshot": 64,
            "attention_links_per_snapshot": 256,
            "raw_fresh_means_match_analysis": True,
            "regional_replicate_coverage": True,
            "source_config_hashes_match_analysis": True,
        },
        "limitations": ["pointwise intervals", "training-end regional welfare shown separately"],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "story_complete",
                "analysis": identity,
                "figure_count": len(completed),
                "input_files_hashed": len(source_hashes),
                "manifest": str(report_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
