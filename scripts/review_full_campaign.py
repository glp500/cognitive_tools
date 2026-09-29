"""Reproduce the full-run review figures from saved CSVs; never run simulations.

Run from any directory with Python, numpy, pandas, and matplotlib installed.
Original campaign files and the frozen analysis are read-only inputs.
"""

import hashlib
import json
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "results/q_learning_baseline/social_analysis/ecology_perception_parallel_v1_analysis"
)
OUT = ROOT / "results/review/full_run_2026-09-29"
OUT.mkdir(parents=True, exist_ok=True)
DATA = SOURCE / "data"
CAT = pd.read_csv(DATA / "run_catalog.csv")
PRIMARY = pd.read_csv(DATA / "primary_window_summary.csv")
OUTCOME = pd.read_csv(DATA / "outcome_summary.csv")
CONTRAST = pd.read_csv(DATA / "planned_contrast_summary.csv")
REPS = pd.read_csv(DATA / "primary_window_replicates.csv")
OREPS = pd.read_csv(DATA / "outcome_replicates.csv")
SCENARIOS = ["uniform_high", "patchy_high", "split_high_low"]
COLORS = ["#247BA0", "#BD6C18", "#8B5094"]
NAMES = ["Uniform high", "Patchy high", "Split high/low"]
LABELS = {
    "b0": "B0 · no social input",
    "s1": "S1 · random, fixed k=4",
    "s2": "S2 · scale-free, variable k",
    "r1_mu010": "R1 · adaptive local",
    "r0_r1_mu010": "R0 · matched local",
    "r2_mu010": "R2 · adaptive mixed",
    "r0_r2_mu010": "R0 · matched mixed",
    "r3_mu010": "R3 · adaptive global",
    "r0_r3_mu010": "R0 · matched global",
}
PREFIX = "ecology_perception_parallel_v1_"
RUNS = [PREFIX + key for key in LABELS]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "savefig.dpi": 180,
        "pdf.fonttype": 42,
    }
)
INPUT_HASHES = {}


def read_raw(run, name):
    path = ROOT / "results/q_learning_baseline/experiments" / run / "data" / f"{name}.csv"
    INPUT_HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def interval(values):
    values = np.asarray(values, dtype=float)
    assert len(values) == 100 and np.isfinite(values).all()
    boot = np.random.default_rng(1729).choice(values, (5000, len(values))).mean(axis=1)
    return values.mean(), *np.quantile(boot, [0.025, 0.975])


def random_four_error(n_low, n=64, k=4):
    """Exact expected MAE against the other N-1 agents, without replacement."""
    total = 0.0
    for focal_low, weight in [(1, n_low / n), (0, 1 - n_low / n)]:
        if not weight:
            continue
        successes = n_low - focal_low
        for observed in range(k + 1):
            failures = k - observed
            if observed <= successes and failures <= n - 1 - successes:
                probability = (
                    comb(successes, observed) * comb(n - 1 - successes, failures) / comb(n - 1, k)
                )
                total += weight * probability * abs(observed / k - successes / (n - 1))
    return total


def audit_and_supplement():
    assert not REPS.duplicated(["run_id", "scenario", "replicate", "metric"]).any()
    assert (REPS.n_checkpoints == 20).all() and (REPS.n_valid == 20).all()
    assert (PRIMARY.n == 100).all() and (OUTCOME.n == 100).all()
    null_lookup = np.array([random_four_error(x) for x in range(65)])
    # Independent identity check for the hypergeometric formula at every prevalence.
    assert np.isclose(null_lookup[0], 0) and np.isclose(null_lookup[-1], 0)
    assert np.allclose(null_lookup, null_lookup[::-1])
    supplements, evaluations, all_stats = [], [], []
    for run in RUNS:
        raw = read_raw(run, "training_timeseries")
        assert not raw.duplicated(["scenario", "population", "replicate", "time"]).any()
        window = raw[(raw.time > 4000) & (raw.time <= 5000)].copy()
        sizes = window.groupby(["scenario", "population", "replicate"]).size()
        assert len(sizes) == 300 and (sizes == 20).all()
        reference = REPS[REPS.run_id == run]
        for metric in reference.metric.unique():
            rebuilt = window.groupby(["scenario", "replicate"])[metric].mean()
            stored = (
                reference[reference.metric == metric].set_index(["scenario", "replicate"]).value
            )
            assert np.allclose(rebuilt.sort_index(), stored.sort_index(), atol=1e-12)
        ev = read_raw(run, "evaluation_summary")
        ev["run_id"] = run
        evaluations.append(ev)
        primary_eval = ev[(ev.strategy == "q_learning") & (ev.evaluation_mode == "fresh_reset")]
        assert len(primary_eval) == 300
        assert not primary_eval.duplicated(["scenario", "replicate"]).any()
        for metric in OREPS.metric.unique():
            stored = OREPS[(OREPS.run_id == run) & (OREPS.metric == metric)]
            assert np.allclose(
                primary_eval.set_index(["scenario", "replicate"])[metric].sort_index(),
                stored.set_index(["scenario", "replicate"]).value.sort_index(),
                atol=1e-12,
            )
        if run.endswith(("_b0", "_s2")):
            continue
        counts = np.rint(window.low_extraction_rate.to_numpy() * 64).astype(int)
        assert np.allclose(counts / 64, window.low_extraction_rate)
        window["random_four_mae"] = null_lookup[counts]
        window["excess_error"] = window.social_perception_error - window.random_four_mae
        grouped = window.groupby(["scenario", "replicate"])[
            ["random_four_mae", "excess_error"]
        ].mean()
        for scenario in SCENARIOS:
            for metric in grouped.columns:
                values = grouped.loc[scenario, metric]
                mean, lo, hi = interval(values)
                all_stats.append(
                    dict(
                        run_id=run,
                        scenario=scenario,
                        metric=metric,
                        n=100,
                        mean=mean,
                        bootstrap_ci_low=lo,
                        bootstrap_ci_high=hi,
                    )
                )
        supplements.append(grouped.reset_index().assign(run_id=run))
    pd.concat(supplements).to_csv(OUT / "sampling_baseline_replicates.csv", index=False)
    sampling = pd.DataFrame(all_stats)
    sampling.to_csv(OUT / "sampling_baseline_summary.csv", index=False)
    evaluation = pd.concat(evaluations, ignore_index=True)
    evaluation.groupby(["run_id", "scenario", "strategy", "evaluation_mode"])[
        [
            "eval_mean_mean_reserve_welfare",
            "eval_mean_mean_resource_fraction",
            "final_mean_wealth",
            "state_occupancy_scarce",
            "state_occupancy_abundant",
        ]
    ].mean().to_csv(OUT / "evaluation_diagnostics.csv")
    for table, reps in [(PRIMARY, REPS), (OUTCOME, OREPS)]:
        for row in table.itertuples():
            vals = reps[
                (reps.run_id == row.run_id)
                & (reps.scenario == row.scenario)
                & (reps.metric == row.metric)
            ].value
            assert len(vals) == 100 and np.isclose(vals.mean(), row.mean, atol=1e-12)
    # Rebuild every reported paired contrast from replicate-level observations.
    combined = (
        pd.concat([REPS, OREPS])
        .set_index(["run_id", "scenario", "metric", "replicate"])
        .value.sort_index()
    )
    for row in CONTRAST.itertuples():
        left = combined.loc[(row.left_run, row.left_scenario, row.metric)].sort_index()
        right = combined.loc[(row.right_run, row.right_scenario, row.metric)].sort_index()
        assert left.index.equals(right.index) and len(left) == 100
        assert np.isclose((left - right).mean(), row.mean, atol=1e-12)
    return sampling, evaluation


def chart_frame(title, subtitle, ncols=3, height=7.3):
    fig, axes = plt.subplots(1, ncols, figsize=(15, height), squeeze=False)
    fig.suptitle(title, x=0.03, y=0.98, ha="left", fontsize=20, weight="bold", color="#192D43")
    fig.text(0.03, 0.915, subtitle, ha="left", fontsize=11, color="#45576A")
    fig.subplots_adjust(left=0.20, right=0.975, top=0.79, bottom=0.16, wspace=0.28)
    for ax in axes[0]:
        ax.grid(axis="x", color="#E5E9ED", zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    return fig, axes[0]


def ecology_legend(fig):
    handles = [
        Line2D([0], [0], marker=marker, color=color, label=name, lw=1.3)
        for marker, color, name in zip(["o", "s", "^"], COLORS, NAMES)
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.61, 0.885), ncol=3, frameon=False
    )


def dotplot(ax, table, metric, runs, scale, xlabel, limits, show_labels=False):
    for j, (scenario, color, marker) in enumerate(zip(SCENARIOS, COLORS, ["o", "s", "^"])):
        subset = table[(table.scenario == scenario) & (table.metric == metric)].set_index("run_id")
        for i, run in enumerate(runs):
            if run not in subset.index:
                continue
            row = subset.loc[run]
            mean, low, high = (
                np.array([row["mean"], row.bootstrap_ci_low, row.bootstrap_ci_high]) * scale
            )
            assert limits[0] <= low <= mean <= high <= limits[1], (metric, run, scenario, low, high)
            ax.errorbar(
                mean,
                i + (j - 1) * 0.19,
                xerr=[[mean - low], [high - mean]],
                fmt=marker,
                color=color,
                ms=5,
                capsize=2,
                lw=1.25,
            )
    ax.set(yticks=range(len(runs)), ylim=(len(runs) - 0.5, -0.6), xlim=limits, xlabel=xlabel)
    ax.set_yticklabels([LABELS[r.removeprefix(PREFIX)] for r in runs] if show_labels else [])
    if limits[0] < 0:
        ax.axvline(0, color="#788695", lw=1, ls="--")


def save(fig, name, pdf, caption):
    fig.text(0.03, 0.035, caption, fontsize=9, color="#45576A", va="bottom")
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{name}.{suffix}", facecolor="white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def main():
    sampling, evaluation = audit_and_supplement()
    primary_note = "100 independent replicates per condition; means and pointwise 95% bootstrap CIs. Training checkpoints: 4000 < t ≤ 5000.\nMismatch excludes tied comparisons; tie rate is shown separately. Axes are deliberately zoomed. Source: ecology_perception_parallel_v1."
    with PdfPages(OUT / "revised_figures.pdf") as pdf:
        fig, axes = chart_frame(
            "Population perception changes more with observation rules than ecology",
            "Three complementary measures; S2 also changes how many peers each agent observes.",
        )
        ecology_legend(fig)
        for i, (metric, label, limits) in enumerate(
            [
                ("social_perception_error", "Mean absolute error (percentage points)", (15, 24)),
                ("majority_mismatch_rate", "Majority mismatch among non-ties (%)", (7, 20)),
                ("majority_tie_rate", "Tied comparisons (%)", (19, 30)),
            ]
        ):
            dotplot(axes[i], PRIMARY, metric, RUNS[1:], 100, label, limits, i == 0)
        save(fig, "01_population_perception", pdf, primary_note)

        fig, axes = chart_frame(
            "Network search scope changes concentration, with little behavior change",
            "Local means sources-of-sources in the social graph. It does not mean geographic proximity.",
            2,
        )
        ecology_legend(fig)
        dotplot(
            axes[0],
            PRIMARY,
            "visibility_gini",
            RUNS,
            1,
            "Visibility Gini (0–1)",
            (0.20, 0.45),
            True,
        )
        dotplot(
            axes[1], PRIMARY, "low_extraction_rate", RUNS, 100, "Low-extraction share (%)", (23, 35)
        )
        save(
            fig,
            "02_collective_organization",
            pdf,
            "100 replicates per condition; pointwise 95% bootstrap CIs; final 1,000 training steps. B0 has no observation network.\nCompare adjacent adaptive and matched rows: concentration mainly follows search scope. Axes are zoomed; no equivalence test was specified.",
        )

        fig, axes = chart_frame(
            "Ecological scenarios differ substantially in welfare and inequality",
            "Absolute outcomes reveal differences hidden by a plot of adaptive-minus-control effects alone.",
        )
        ecology_legend(fig)
        for i, (metric, scale, label, limits) in enumerate(
            [
                (
                    "eval_mean_mean_resource_fraction",
                    100,
                    "Mean local resource / capacity (%)",
                    (9, 15),
                ),
                ("eval_mean_mean_reserve_welfare", 100, "Mean reserve welfare (%)", (60, 82)),
                ("final_wealth_gini", 1, "Final accumulated-harvest Gini (0–1)", (0.40, 0.65)),
            ]
        ):
            dotplot(axes[i], OUTCOME, metric, RUNS, scale, label, limits, i == 0)
        save(
            fig,
            "03_ecology_and_outcomes",
            pdf,
            "100 replicates; pointwise 95% bootstrap CIs. Frozen policies/networks, 1,000-step fresh-reset evaluation (full initial reserves).\nSplit changes capacity, regeneration, equilibrium and layout together. Resource fraction is not total resource. These are finite-horizon outcomes.",
        )

        adaptive = [PREFIX + f"r{i}_mu010" for i in [1, 2, 3]]
        diff = CONTRAST[CONTRAST.family == "adaptive_minus_matched"].rename(
            columns={"left_run": "run_id", "left_scenario": "scenario"}
        )
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle(
            "Adaptive selection improves perception; downstream benefits are inconsistent",
            x=0.03,
            y=0.98,
            ha="left",
            fontsize=19,
            weight="bold",
            color="#192D43",
        )
        fig.text(
            0.03,
            0.932,
            "Paired differences: adaptive minus its event-count-matched random control. Each point uses 100 paired replicates.",
            fontsize=11,
        )
        ecology_legend(fig)
        fig.subplots_adjust(left=0.19, right=0.97, top=0.80, bottom=0.17, hspace=0.58, wspace=0.30)
        for i, (metric, scale, label, limits) in enumerate(
            [
                ("social_perception_error", 100, "Perception error difference (pp)", (-0.8, 0.4)),
                ("majority_mismatch_rate", 100, "Non-tie mismatch difference (pp)", (-4, 1)),
                ("low_extraction_rate", 100, "Low-extraction difference (pp)", (-3, 3)),
                (
                    "eval_mean_mean_resource_fraction",
                    100,
                    "Resource fraction difference (pp)",
                    (-1.5, 1.5),
                ),
                ("eval_mean_mean_reserve_welfare", 100, "Reserve welfare difference (pp)", (-4, 4)),
                ("final_wealth_gini", 1, "Wealth Gini difference", (-0.025, 0.025)),
            ]
        ):
            dotplot(axes.flat[i], diff, metric, adaptive, scale, label, limits, i % 3 == 0)
            axes.flat[i].grid(axis="x", color="#E5E9ED")
        save(
            fig,
            "04_adaptive_minus_matched",
            pdf,
            "Pointwise 95% paired bootstrap CIs; no multiplicity adjustment or equivalence margin. pp = percentage points.\nTop row: final training window. Bottom row: frozen, fresh-reset evaluation. Isolated intervals excluding zero do not establish a broad welfare benefit.",
        )

        fig, axes = chart_frame(
            "Diagnostic: most raw perception error is expected from four-peer sampling",
            "Post-hoc finite-population benchmark, conditioned on each checkpoint's actual population actions.",
            2,
        )
        ecology_legend(fig)
        fixed_k = [r for r in RUNS[1:] if r != PREFIX + "s2"]
        dotplot(
            axes[0],
            sampling,
            "random_four_mae",
            fixed_k,
            100,
            "Expected random-four MAE (pp)",
            (16, 20),
            True,
        )
        dotplot(
            axes[1],
            sampling,
            "excess_error",
            fixed_k,
            100,
            "Observed minus random-four MAE (pp)",
            (-1, 1),
        )
        save(
            fig,
            "05_sampling_diagnostic",
            pdf,
            "Exact hypergeometric expectation: four distinct peers from the other 63 agents; excludes the focal agent in the target population.\nReplicate means and pointwise 95% bootstrap CIs. Not an endogenous network counterfactual. S2 excluded: attention capacity varies across agents.",
        )

        # Existing evaluation modes are a diagnostic, not a replacement primary endpoint.
        mode_rows = []
        for run in RUNS:
            for scenario in SCENARIOS:
                sub = evaluation[
                    (evaluation.run_id == run)
                    & (evaluation.scenario == scenario)
                    & (evaluation.strategy == "q_learning")
                ]
                for mode in ["fresh_reset", "continuation"]:
                    mean, lo, hi = interval(
                        sub[sub.evaluation_mode == mode].eval_mean_mean_reserve_welfare
                    )
                    mode_rows.append(
                        dict(
                            run_id=run,
                            scenario=scenario,
                            metric=mode,
                            mean=mean,
                            bootstrap_ci_low=lo,
                            bootstrap_ci_high=hi,
                        )
                    )
        modes = pd.DataFrame(mode_rows)
        modes.to_csv(OUT / "evaluation_mode_summary.csv", index=False)
        fig, axes = chart_frame(
            "Diagnostic: welfare depends strongly on the evaluation starting state",
            "Both evaluations freeze learning and network adaptation; only one begins from the trained ecological and agent state.",
            2,
        )
        ecology_legend(fig)
        for i, (mode, label) in enumerate(
            [
                ("fresh_reset", "Fresh reset: mean reserve welfare (%)"),
                ("continuation", "Continuation: mean reserve welfare (%)"),
            ]
        ):
            dotplot(axes[i], modes, mode, RUNS, 100, label, (0, 100), i == 0)
        save(
            fig,
            "06_evaluation_diagnostic",
            pdf,
            "Post-hoc display of already-recorded outcomes; 100 replicates, pointwise 95% bootstrap CIs, identical 0–100% scales.\nFresh-reset welfare includes initial full reserves and a reset ecology. Neither 1,000-step evaluation establishes long-run sustainability.",
        )
    for path in DATA.glob("*.csv"):
        INPUT_HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(
        campaign=PREFIX.rstrip("_"),
        review_date="2026-09-29",
        primary_analysis_changed=False,
        simulation_rerun=False,
        n_conditions=27,
        n_replicates_per_condition=100,
        checked="All primary replicate values, outcome values, summary means and paired contrast means rebuilt from source tables",
        original_intervals="Frozen analysis bootstrap intervals retained for figures 1–4",
        supplemental_intervals="5000 percentile-bootstrap resamples, seed 1729, across independent replicates",
        input_sha256=INPUT_HASHES,
    )
    (OUT / "review_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified 2,700 condition-replicates; wrote six figures and PDF to {OUT}")
    print(sampling.groupby("metric")["mean"].agg(["min", "max"]).to_string())


if __name__ == "__main__":
    main()
