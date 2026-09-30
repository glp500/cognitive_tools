"""Render compact hypothesis-first figures from a completed Stage-5 analysis.

These are publication candidates, separate from the frozen five-figure study
contract. No simulations or statistical estimates are recomputed here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from cognitive_tools.visibility import PROFILES
from cognitive_tools.visibility_figures import LABELS

ECOLOGIES = ("uniform_high", "patchy_high", "split_high_low")
COLORS = {"fixed": "#216b72", "adaptive_bounded": "#b45335"}


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty summary table: {path}")
    return rows


def _point(ax, row: dict, y: float, *, color: str) -> None:
    estimate, low, high = (float(row[key]) for key in ("mean", "low", "high"))
    if not low <= estimate <= high:
        raise ValueError(f"Invalid interval for {row}")
    ax.plot([low, high], [y, y], color=color, lw=1.5, solid_capstyle="round")
    ax.plot([low, low], [y - 0.09, y + 0.09], color=color, lw=1)
    ax.plot([high, high], [y - 0.09, y + 0.09], color=color, lw=1)
    ax.scatter([estimate], [y], s=22, color=color, zorder=3)


def _axis(ax, labels: list[str], limits: tuple[float, float], *, show_labels: bool) -> None:
    ax.axvline(0, color="#555555", lw=0.9, zorder=0)
    ax.set_xlim(*limits)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_yticks(range(len(labels)), labels)
    if not show_labels:
        ax.tick_params(axis="y", labelleft=False)
    ax.grid(axis="x", color="#dedede", lw=0.55, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def _limits(rows: list[dict], *, symmetric: bool = True) -> tuple[float, float]:
    minimum = min(float(row["low"]) for row in rows)
    maximum = max(float(row["high"]) for row in rows)
    extent = max(abs(minimum), abs(maximum), 0.001)
    if symmetric:
        return -1.12 * extent, 1.12 * extent
    return min(-0.05 * extent, minimum - 0.08 * extent), maximum + 0.08 * extent


def _finish(fig, output: Path, stem: str, note: str) -> None:
    fig.text(0.01, 0.012, note, fontsize=9, color="#444444")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _h1(rows: list[dict], output: Path, n: int, note: str) -> None:
    rows = [row for row in rows if row["metric"] == "social_perception_error"]
    dynamics = ("fixed", "adaptive_bounded")
    labels = [
        f"{LABELS[profile]} · {LABELS[dynamic]}" for profile in PROFILES for dynamic in dynamics
    ]
    comparisons = (
        ("patchy_high-uniform_high", "Patchy high − uniform high"),
        ("split_high_low-uniform_high", "Split high/low − uniform high"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.3), sharey=True)
    limits = _limits(rows)
    for ax, (comparison, title) in zip(axes, comparisons):
        subset = {(r["profile"], r["dynamics"]): r for r in rows if r["comparison"] == comparison}
        if len(subset) != len(labels):
            raise ValueError(f"Incomplete H1 panel: {comparison}")
        _axis(ax, labels, limits, show_labels=ax is axes[0])
        for profile_index, profile in enumerate(PROFILES):
            for dynamic_index, dynamic in enumerate(dynamics):
                _point(
                    ax,
                    subset[(profile, dynamic)],
                    2 * profile_index + dynamic_index,
                    color=COLORS[dynamic],
                )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Difference in mean absolute perception error")
    fig.suptitle("H1  ·  Ecological contrasts within each social condition", fontsize=12)
    fig.tight_layout(rect=(0, 0.065, 1, 0.93))
    _finish(
        fig,
        output,
        "H1_ecology_perception",
        f"{note}; n={n} matched replicates per point. Orange: adaptive; teal: fixed.",
    )


def _h2(rows: list[dict], output: Path, n: int, note: str) -> None:
    rows = [row for row in rows if row["metric"] == "social_perception_error"]
    comparisons = (
        ("random-equal", "Random − equal"),
        ("normal_centered-random", "Normal − random"),
        ("low_propensity_majority-random", "Low majority − random"),
        ("high_propensity_majority-random", "High majority − random"),
    )
    observed = {row["comparison"] for row in rows}
    if observed != {name for name, _ in comparisons}:
        raise ValueError("H2 contrast directions differ from the frozen study; regenerate analysis")
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.5), sharey=True)
    limits = _limits(rows)
    labels = [label for _, label in comparisons]
    for ax, ecology in zip(axes, ECOLOGIES):
        subset = {r["comparison"]: r for r in rows if r["scenario"] == ecology}
        if len(subset) != len(comparisons):
            raise ValueError(f"Incomplete H2 panel: {ecology}")
        _axis(ax, labels, limits, show_labels=ax is axes[0])
        for index, (comparison, _) in enumerate(comparisons):
            _point(ax, subset[comparison], index, color="#434a7c")
        ax.set_title(LABELS[ecology], fontsize=10)
        ax.set_xlabel("Difference in perception error")
    fig.suptitle("H2  ·  Initial visibility contrasts on fixed networks", fontsize=12)
    fig.tight_layout(rect=(0, 0.095, 1, 0.88))
    _finish(fig, output, "H2_profile_perception", f"{note}; n={n} matched replicates per point.")


def _h3(rows: list[dict], output: Path, n: int, note: str, *, companion: bool) -> None:
    metrics = (
        (("low_extraction_rate", "Low-extraction share"),)
        if companion
        else (
            ("social_perception_error", "Perception error"),
            ("visibility_gini", "Visibility Gini"),
        )
    )
    selected = [row for row in rows if row["metric"] in dict(metrics)]
    fig, axes = plt.subplots(
        len(metrics), 3, figsize=(9.2, 3.6 if companion else 6.2), squeeze=False
    )
    labels = [LABELS[profile] for profile in PROFILES]
    for row_index, (metric, label) in enumerate(metrics):
        metric_rows = [row for row in selected if row["metric"] == metric]
        limits = _limits(metric_rows, symmetric=metric != "visibility_gini")
        for col, ecology in enumerate(ECOLOGIES):
            ax = axes[row_index, col]
            subset = {r["profile"]: r for r in metric_rows if r["scenario"] == ecology}
            if len(subset) != len(PROFILES):
                raise ValueError(f"Incomplete H3 panel: {metric}, {ecology}")
            _axis(ax, labels, limits, show_labels=col == 0)
            for index, profile in enumerate(PROFILES):
                _point(ax, subset[profile], index, color="#b45335")
            if row_index == 0:
                ax.set_title(LABELS[ecology], fontsize=10)
            ax.set_xlabel(f"Adaptive − fixed · {label}")
    title = (
        "Companion outcome  ·  Adaptive effects on extraction"
        if companion
        else "H3  ·  Adaptive effects on perception and visibility"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.095 if companion else 0.065, 1, 0.92))
    stem = "S3_adaptation_extraction" if companion else "H3_adaptation_primary"
    _finish(
        fig,
        output,
        stem,
        f"{note}; n={n} paired replicates per point. Zero indicates no difference.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.analysis_dir
    manifest = json.loads((root / "analysis_manifest.json").read_text())
    if manifest.get("profile") != "visibility":
        raise ValueError("Expected a Stage-5 visibility analysis")
    output = args.output or root / "publication_candidates"
    output.mkdir(parents=True, exist_ok=True)
    tables = root / "data"
    ecology = _read(tables / "ecology_contrast_summary.csv")
    profiles = _read(tables / "visibility_profile_contrast_summary.csv")
    adaptation = _read(tables / "adaptive_fixed_summary.csv")
    all_rows = ecology + profiles + adaptation
    counts = {int(row["n_replicates"]) for row in all_rows}
    if len(counts) != 1:
        raise ValueError(f"Unequal replicate counts across plotted estimates: {counts}")
    n = counts.pop()
    populations = {int(row["population"]) for row in all_rows}
    if len(populations) != 1:
        raise ValueError(f"Unequal populations across plotted estimates: {populations}")
    social_input = next((item for item in manifest["inputs"] if item["treatment"] != "B0"), None)
    if social_input is None:
        raise ValueError("No social treatment found in the analysis manifest")
    config = json.loads((Path(social_input["run_path"]) / "config.json").read_text())
    steps = int(config["training_steps"])
    note = (
        f"Simulation; N={populations.pop()}, k={int(config['attention_k'])}, "
        f"T={steps:,} (final {min(1000, steps):,}); 95% pointwise bootstrap intervals"
    )
    plt.rcParams.update(
        {"font.size": 10, "axes.labelsize": 10, "pdf.fonttype": 42, "svg.fonttype": "none"}
    )
    _h1(ecology, output, n, note)
    _h2(profiles, output, n, note)
    _h3(adaptation, output, n, note, companion=False)
    _h3(adaptation, output, n, note, companion=True)
    print(f"Saved four hypothesis-first figure sets to {output}")


if __name__ == "__main__":
    main()
