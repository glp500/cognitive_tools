"""Audit and analyze a completed population-payoff experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .env import reward_definition
from .payoff import PayoffConfig, assignment, sha256, write_csv

CONTRASTS = ("collective", "exploitation", "greed", "fear")


def simultaneous_intervals(values, *, resamples=5000, seed=1729):
    """Centered max-standardized bootstrap, resampling independent replicate rows."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite independent replicate rows")
    if resamples < 100:
        raise ValueError("At least 100 bootstrap resamples required")
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / np.sqrt(len(values))
    active = se > np.finfo(float).eps
    rng = np.random.default_rng(seed)
    maxima = []
    for start in range(0, resamples, 250):
        indices = rng.integers(len(values), size=(min(250, resamples - start), len(values)))
        deviations = np.abs(values[indices].mean(axis=1) - mean)
        maxima.extend(
            np.max(deviations[:, active] / se[active], axis=1)
            if active.any()
            else np.zeros(len(indices))
        )
    critical = float(np.quantile(maxima, 0.95))
    return mean, mean - critical * se, mean + critical * se


def condition_status(low, high):
    if low > 1e-12:
        return "supported"
    if high <= 1e-12:
        return "contradicted"
    return "inconclusive"


def classify(bounds):
    """G AND E AND (greed OR fear), for strict positive contrasts."""
    statuses = [condition_status(*pair) for pair in bounds]
    g, e, greed, fear = statuses
    if g == "contradicted" or e == "contradicted" or greed == fear == "contradicted":
        return "contradicted"
    if g == e == "supported" and "supported" in (greed, fear):
        return "supported"
    return "inconclusive"


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_run(directory):
    """Verify hashes, exact coverage, assignment semantics and payoff accounting."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("schema") not in ("population_payoff_v1", "population_payoff_v2")
        or manifest.get("status") != "complete"
    ):
        raise ValueError("Require a complete population_payoff_v1/v2 experiment")
    for name in (
        "paired_returns.csv",
        "episode_summary.csv",
        "agent_returns.csv",
        "assignments.csv",
    ):
        if sha256(directory / name) != manifest["output_sha256"].get(name):
            raise ValueError(f"Output hash mismatch: {name}")
    protocol_hash = hashlib.sha256(
        json.dumps(manifest["config"], sort_keys=True).encode()
    ).hexdigest()
    if protocol_hash != manifest["protocol_sha256"]:
        raise ValueError("Protocol hash mismatch")
    config = PayoffConfig(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in manifest["config"].items()}
    )
    config.validate()
    version2 = manifest["schema"] == "population_payoff_v2"
    if version2:
        if manifest.get("reward_definition") != reward_definition(
            config.reward_mode, config.metabolism
        ):
            raise ValueError("Invalid reward definition")
    elif config.reward_mode != "harvest":
        raise ValueError("Legacy schema only supports harvest reward")
    if config.replicates < 2:
        raise ValueError("At least two independent replicates required for inference")
    pairs = read_csv(directory / "paired_returns.csv")
    episodes = read_csv(directory / "episode_summary.csv")
    assignments = read_csv(directory / "assignments.csv")
    reps = range(config.replicate_start, config.replicate_start + config.replicates)
    blocks = {(s, r) for s in config.scenarios for r in reps}
    assigned = {}
    for row in assignments:
        key = (row["scenario"], int(row["replicate"]), int(row["focal"]), int(row["assignment"]))
        if key in assigned or key[:2] not in blocks or not 0 <= key[3] < config.assignments:
            raise ValueError("Duplicate or unexpected assignment")
        order = json.loads(row["coplayer_order"])
        assignment(config.population, key[2], order, 0, "C")
        assigned[key] = order
    for block in blocks:
        focals = {key[2] for key in assigned if key[:2] == block}
        if len(focals) != config.focal_count:
            raise ValueError("Incomplete focal coverage")
        if any((*block, f, a) not in assigned for f in focals for a in range(config.assignments)):
            raise ValueError("Incomplete assignment coverage")
    expected_pairs = {
        (*key, k, h) for key in assigned for k in config.compositions for h in config.horizons
    }
    pair_map = {}
    for row in pairs:
        key = (
            row["scenario"],
            int(row["replicate"]),
            int(row["focal"]),
            int(row["assignment"]),
            int(row["k"]),
            int(row["horizon"]),
        )
        if key in pair_map or key not in expected_pairs:
            raise ValueError("Duplicate or unexpected pair")
        for metric in ("sum", "discounted"):
            c, d, delta = [float(row[f"{prefix}_{metric}"]) for prefix in ("c", "d", "delta")]
            if not np.isfinite([c, d, delta]).all() or not np.isclose(d - c, delta, atol=1e-12):
                raise ValueError("Invalid paired return accounting")
        for branch in ("C", "D"):
            homogeneous = (branch == "C" and key[4] == config.population - 1) or (
                branch == "D" and key[4] == 0
            )
            expected = f"all_{branch}" if homogeneous else f"f{key[2]}_a{key[3]}_k{key[4]}_{branch}"
            if row[f"{branch.lower()}_episode"] != expected:
                raise ValueError("Pair refers to incorrect episode")
        pair_map[key] = row
    if set(pair_map) != expected_pairs:
        raise ValueError("Incomplete pair coverage")
    expected_episodes = {
        (s, r, f"all_{b}", h) for s, r in blocks for b in ("C", "D") for h in config.horizons
    }
    for key, row in pair_map.items():
        expected_episodes.update((key[0], key[1], row[f"{b}_episode"], key[-1]) for b in ("c", "d"))
    episode_map = {}
    for row in episodes:
        key = (row["scenario"], int(row["replicate"]), row["episode"], int(row["horizon"]))
        if key in episode_map or key not in expected_episodes:
            raise ValueError("Duplicate or unexpected episode")
        episode_map[key] = row
    if set(episode_map) != expected_episodes:
        raise ValueError("Incomplete episode coverage")
    # Stream large per-agent tables; keep only aggregate accounting and focal lookups.
    seen = defaultdict(set)
    sums = defaultdict(lambda: np.zeros(2))
    gross_sums = defaultdict(lambda: np.zeros(2))
    focal_returns = {}
    positions = {}
    with (directory / "agent_returns.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario"], int(row["replicate"]), row["episode"], int(row["horizon"]))
            agent = int(row["agent"])
            if key not in episode_map or agent in seen[key] or not 0 <= agent < config.population:
                raise ValueError("Duplicate or unexpected agent row")
            seen[key].add(agent)
            position_key = (*key[:2], agent)
            location = (row["x"], row["y"], row["region"])
            if position_key in positions and positions[position_key] != location:
                raise ValueError("Agent position/region differs between paired resets")
            positions[position_key] = location
            ep = episode_map[key]
            if ep["kind"] == "endpoint":
                label = ep["branch"]
            else:
                akey = (*key[:2], int(ep["focal"]), int(ep["assignment"]))
                label = assignment(
                    config.population, akey[2], assigned[akey], int(ep["k"]), ep["branch"]
                )[agent]
            if row["policy"] != label:
                raise ValueError("Incorrect policy assignment")
            values = np.array([float(row["return_sum"]), float(row["return_discounted"])])
            if version2:
                gross = np.array([float(row[f"harvest_{m}"]) for m in ("sum", "discounted")])
                surplus = np.array(
                    [float(row[f"uncredited_harvest_{m}"]) for m in ("sum", "discounted")]
                )
                if (
                    not np.isfinite([*values, *gross, *surplus]).all()
                    or min(*values, *gross, *surplus) < -1e-12
                    or not np.allclose(values + surplus, gross, rtol=1e-10, atol=1e-12)
                    or not np.isclose(gross[0], float(row["wealth_delta"]), rtol=1e-10, atol=1e-12)
                ):
                    raise ValueError("Invalid utility/harvest accounting")
                if config.reward_mode == "harvest":
                    if not np.allclose(values, gross, rtol=1e-10, atol=1e-12):
                        raise ValueError("Invalid harvest-mode accounting")
                else:
                    h = key[-1]
                    weight = h if config.gamma == 1 else (1 - config.gamma**h) / (1 - config.gamma)
                    if np.any(values > config.metabolism * np.array([h, weight]) + 1e-12):
                        raise ValueError("Invalid capped reward accounting")
                gross_sums[key] += gross
            elif not np.isfinite(values).all() or not np.isclose(
                values[0], float(row["wealth_delta"]), atol=1e-12
            ):
                raise ValueError("Invalid agent reward accounting")
            sums[key] += values
            if ep["kind"] == "endpoint" or agent == int(ep["focal"]):
                focal_returns[(*key, agent)] = values
    for key, row in episode_map.items():
        if len(seen[key]) != config.population:
            raise ValueError("Incomplete per-agent coverage")
        reported = [float(row["mean_return_sum"]), float(row["mean_return_discounted"])]
        if not np.allclose(sums[key] / config.population, reported, atol=1e-12):
            raise ValueError("Episode means do not match agent returns")
        if version2 and not np.allclose(
            gross_sums[key] / config.population,
            [float(row["mean_harvest_sum"]), float(row["mean_harvest_discounted"])],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("Episode harvest means do not match agent accounting")
    for key, row in pair_map.items():
        for b in ("c", "d"):
            values = focal_returns[(*key[:2], row[f"{b}_episode"], key[-1], key[2])]
            if not np.allclose(
                values, [float(row[f"{b}_sum"]), float(row[f"{b}_discounted"])], atol=1e-12
            ):
                raise ValueError("Paired returns differ from focal agent returns")
    return manifest, config, pairs, episodes


def summarize(config, pairs, episodes, *, resamples, seed):
    """Equal-weight independent replicate means, never pooled agent bootstrap."""
    grouped = defaultdict(list)
    for row in pairs:
        key = (row["scenario"], int(row["replicate"]), int(row["horizon"]), int(row["k"]))
        grouped[key].append(row)
    endpoints = {
        (r["scenario"], int(r["replicate"]), int(r["horizon"]), r["branch"]): r
        for r in episodes
        if r["kind"] == "endpoint"
    }
    replicates = list(range(config.replicate_start, config.replicate_start + config.replicates))
    curves, contrasts, verdicts, blocks, diagnostics = [], [], [], [], []
    for horizon in config.horizons:
        for metric in ("sum", "discounted"):
            family, family_names = [], []
            for scenario in config.scenarios:
                values = np.array(
                    [
                        [
                            [
                                np.mean(
                                    [
                                        float(row[f"{b}_{metric}"])
                                        for row in grouped[(scenario, rep, horizon, k)]
                                    ]
                                )
                                for b in ("c", "d")
                            ]
                            for k in config.compositions
                        ]
                        for rep in replicates
                    ]
                )
                rng = np.random.default_rng(seed)
                indices = rng.integers(config.replicates, size=(resamples, config.replicates))
                boots = values[indices].mean(axis=1)
                means = values.mean(axis=0)
                for j, k in enumerate(config.compositions):
                    curve = dict(
                        scenario=scenario,
                        horizon=horizon,
                        return_type=metric,
                        k=k,
                        n_replicates=config.replicates,
                    )
                    for label, sample, observed in (
                        ("c", boots[:, j, 0], means[j, 0]),
                        ("d", boots[:, j, 1], means[j, 1]),
                        ("delta", boots[:, j, 1] - boots[:, j, 0], means[j, 1] - means[j, 0]),
                    ):
                        curve.update(
                            {
                                f"{label}_mean": float(observed),
                                f"{label}_low": float(np.quantile(sample, 0.025)),
                                f"{label}_high": float(np.quantile(sample, 0.975)),
                            }
                        )
                    curves.append(curve)
                    for index, rep in enumerate(replicates):
                        blocks.append(
                            dict(
                                scenario=scenario,
                                replicate=rep,
                                horizon=horizon,
                                return_type=metric,
                                k=k,
                                c=values[index, j, 0],
                                d=values[index, j, 1],
                                delta=values[index, j, 1] - values[index, j, 0],
                            )
                        )
                collective = np.array(
                    [
                        float(endpoints[(scenario, rep, horizon, "C")][f"mean_return_{metric}"])
                        - float(endpoints[(scenario, rep, horizon, "D")][f"mean_return_{metric}"])
                        for rep in replicates
                    ]
                )
                for name, column in zip(
                    CONTRASTS,
                    (
                        collective,
                        values[:, -1, 0] - values[:, 0, 0],
                        values[:, -1, 1] - values[:, -1, 0],
                        values[:, 0, 1] - values[:, 0, 0],
                    ),
                ):
                    family.append(column)
                    family_names.append((scenario, name))
            means, lows, highs = simultaneous_intervals(
                np.array(family).T, resamples=resamples, seed=seed
            )
            for index, (scenario, name) in enumerate(family_names):
                contrasts.append(
                    dict(
                        scenario=scenario,
                        horizon=horizon,
                        **{"return": metric},
                        contrast=name,
                        mean=float(means[index]),
                        low=float(lows[index]),
                        high=float(highs[index]),
                        status=condition_status(lows[index], highs[index]),
                        n_replicates=config.replicates,
                        replicate_sd=float(np.std(family[index], ddof=1)),
                        standard_error=float(
                            np.std(family[index], ddof=1) / np.sqrt(config.replicates)
                        ),
                        projected_se_100=float(np.std(family[index], ddof=1) / 10),
                        family_size=len(family_names),
                        interval="95% simultaneous centered max-standardized bootstrap",
                    )
                )
            for scenario in config.scenarios:
                bounds = [
                    (lows[i], highs[i]) for i, (s, _) in enumerate(family_names) if s == scenario
                ]
                verdicts.append(
                    dict(
                        scenario=scenario,
                        horizon=horizon,
                        return_type=metric,
                        verdict=classify(bounds),
                    )
                )
        for scenario in config.scenarios:
            for branch in ("C", "D"):
                rows = [endpoints[(scenario, rep, horizon, branch)] for rep in replicates]
                diagnostics.append(
                    dict(
                        scenario=scenario,
                        horizon=horizon,
                        branch=branch,
                        **{
                            name: float(np.mean([float(r[name]) for r in rows]))
                            for name in (
                                "mean_return_sum",
                                "mean_return_discounted",
                                "mean_resource_fraction",
                                "final_resource_fraction",
                                "mean_reserve_welfare",
                                "late_harvest_rate",
                            )
                        },
                    )
                )
    return curves, contrasts, verdicts, blocks, diagnostics


def make_figures(config, curves, diagnostics, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payoff_label = "harvest" if config.reward_mode == "harvest" else "capped-harvest utility"
    for horizon in config.horizons:
        for metric in ("sum", "discounted"):
            fig, axes = plt.subplots(
                2, len(config.scenarios), figsize=(5 * len(config.scenarios), 7), squeeze=False
            )
            for column, scenario in enumerate(config.scenarios):
                rows = [
                    r
                    for r in curves
                    if (r["scenario"], r["horizon"], r["return_type"])
                    == (scenario, horizon, metric)
                ]
                x = [r["k"] for r in rows]

                def draw(ax, prefix, label, color):
                    means = [r[f"{prefix}_mean"] for r in rows]
                    lows = [r[f"{prefix}_low"] for r in rows]
                    highs = [r[f"{prefix}_high"] for r in rows]
                    if len(rows) == 2:
                        # Do not imply that unmeasured interior compositions were evaluated.
                        ax.scatter(x, means, label=label, color=color)
                        ax.vlines(x, lows, highs, color=color, linewidth=2)
                    else:
                        ax.plot(x, means, label=label, color=color, marker="o")
                        ax.fill_between(x, lows, highs, color=color, alpha=0.16)

                draw(axes[0, column], "c", "Focal C", "#267c5b")
                draw(axes[0, column], "d", "Focal D", "#c35d36")
                axes[0, column].set_title(SCENARIO_LABELS.get(scenario, scenario))
                axes[0, column].set_ylabel(f"Individual {payoff_label} return")
                axes[0, column].legend()
                axes[1, column].axhline(0, color="gray", linewidth=1)
                draw(axes[1, column], "delta", "D - C", "#405885")
                axes[1, column].set_ylabel("Unilateral incentive: D − C")
                axes[1, column].set_xlabel(
                    f"Cooperative co-players (out of {config.population - 1})"
                )
            title = f"{metric.capitalize()} {payoff_label} | H={horizon} | C={config.policy_c}, D={config.policy_d}"
            if metric == "discounted":
                title += f" | gamma={config.gamma}"
            intervals = "Pointwise 95% replicate-bootstrap intervals"
            if len(config.compositions) == 2:
                intervals += "; endpoints only, interior unmeasured"
            fig.suptitle(title + "\n" + intervals)
            fig.tight_layout(rect=(0, 0, 1, 0.91))
            for extension in ("png", "pdf", "svg"):
                fig.savefig(output / f"schelling_{metric}_{horizon}.{extension}", dpi=160)
            plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, field, title in zip(
            axes,
            ("mean_return_sum", "mean_resource_fraction", "mean_reserve_welfare"),
            (f"Total {payoff_label} per agent", "Mean resource fraction", "Mean reserve welfare"),
        ):
            for offset, branch, color in [(-0.18, "C", "#267c5b"), (0.18, "D", "#c35d36")]:
                data = [
                    next(
                        r[field]
                        for r in diagnostics
                        if (r["scenario"], r["horizon"], r["branch"]) == (s, horizon, branch)
                    )
                    for s in config.scenarios
                ]
                ax.bar(
                    np.arange(len(data)) + offset,
                    data,
                    width=0.36,
                    label=f"All {branch}",
                    color=color,
                )
            ax.set_xticks(
                range(len(config.scenarios)),
                [SCENARIO_LABELS.get(s, s) for s in config.scenarios],
                rotation=15,
            )
            ax.set_title(title)
            ax.legend()
        fig.suptitle(f"Homogeneous endpoints, H={horizon}; descriptive replicate means")
        fig.tight_layout()
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output / f"endpoints_{horizon}.{extension}", dpi=160)
        plt.close(fig)


SCENARIO_LABELS = {
    "uniform_high": "Uniform high",
    "patchy_high": "Patchy high",
    "split_high_low": "Split high/low",
}


def analyze(directory, output, *, resamples=5000, seed=1729, figures=True):
    manifest, config, pairs, episodes = load_run(directory)
    curves, contrasts, verdicts, blocks, diagnostics = summarize(
        config, pairs, episodes, resamples=resamples, seed=seed
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [
        ("curve_summary", curves),
        ("contrasts", contrasts),
        ("verdicts", verdicts),
        ("replicate_returns", blocks),
        ("endpoint_summary", diagnostics),
    ]:
        write_csv(output / f"{name}.csv", rows)
    if config.reward_mode == "capped_harvest":
        normalized = []
        for row in contrasts:
            h = row["horizon"]
            weight = (
                h
                if row["return"] == "sum" or config.gamma == 1
                else (1 - config.gamma**h) / (1 - config.gamma)
            )
            maximum = config.metabolism * weight
            normalized.append(
                {
                    **row,
                    "maximum_utility": maximum,
                    **{
                        f"{field}_percent_maximum": 100 * row[field] / maximum
                        for field in ("mean", "low", "high")
                    },
                }
            )
        write_csv(output / "normalized_contrasts.csv", normalized)
    if figures:
        make_figures(config, curves, diagnostics, output)
    report = [
        "# Population payoff validation",
        f"Run purpose: **{manifest['purpose']}**. Policies: C={config.policy_c}, D={config.policy_d} (scarce/moderate/abundant).",
        f"{config.replicates} independent replicates per ecology; {config.focal_count} focal agents and {config.assignments} assignments per replicate.",
        f"Reward: **{config.reward_mode}**, cap={config.metabolism if config.reward_mode == 'capped_harvest' else None}. Primary learner-aligned return discounts actual reward; undiscounted verdicts are separate. k counts OTHER cooperative agents.",
        "Decision bounds within 1e-12 of zero are numerical ties, not positive evidence.",
        "The endpoint criterion is G>0 AND E>0 AND (greed>0 OR fear>0). Verdicts apply only to this policy pair, reset distribution and payoff definition.",
        "\n## Verdicts\n",
        "| Ecology | Horizon | Return | Verdict |",
        "|---|---:|---|---|",
    ]
    report.extend(
        f"| {r['scenario']} | {r['horizon']} | {r['return_type']} | {r['verdict']} |"
        for r in verdicts
    )
    report.extend(
        [
            "\n## Decision contrasts\n",
            "| Ecology | H | Return | Contrast | Mean | Simultaneous 95% interval | Status |",
            "|---|---:|---|---|---:|---|---|",
        ]
    )
    report.extend(
        f"| {r['scenario']} | {r['horizon']} | {r['return']} | {r['contrast']} | {r['mean']:.6g} | [{r['low']:.6g}, {r['high']:.6g}] | {r['status']} |"
        for r in contrasts
    )
    report.extend(
        [
            "\n## Interpretation and limitations\n",
            "Bootstrap resampling uses independent replicate blocks after averaging focal agents and assignments. Decision intervals are simultaneous over four contrasts and all ecologies, separately for each horizon/return. Cross-horizon selection is not covered. Curve intervals are pointwise and descriptive.",
            "Intervals are approximate; small pilots and zero observed variance can understate uncertainty. A pilot verdict is exploratory, not a held-out confirmatory result.",
            "Population means do not establish that every ecological region or agent benefits. Per-agent locations and regions are retained in agent_returns.csv for asymmetric-incentive diagnostics.",
            "If collective advantage fails, test sustainable resource-dependent policies on development seeds before changing incentives. If only long-horizon harvest favors cooperation, examine discounting. If unilateral conflict fails, examine alternative policies and shared-resource externalities. Do not tune rewards merely to force a pass.",
            "A negative result does not rule out all sequential social dilemmas in the environment. A positive result does not establish learning convergence or benefits from social information.",
            f"Runtime: {manifest['elapsed_seconds']:.1f}s; {manifest['episode_count']} physical episodes; {manifest['environment_steps']} environment steps.",
            "[Definition source](https://arxiv.org/html/1702.03037); [population analysis source](https://arxiv.org/html/2503.14576v3).",
        ]
    )
    (output / "validation_report.md").write_text("\n".join(report) + "\n")
    metadata = dict(
        input_manifest_sha256=sha256(Path(directory) / "manifest.json"),
        input_output_sha256=manifest["output_sha256"],
        resamples=resamples,
        seed=seed,
        interval_method="centered max-standardized replicate bootstrap",
        family="four contrasts x configured ecologies, separately by horizon and return",
        analysis_source_sha256=sha256(__file__),
    )
    (output / "analysis_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return verdicts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    analyze(args.run, args.output, resamples=args.resamples, seed=args.seed)


if __name__ == "__main__":
    main()
