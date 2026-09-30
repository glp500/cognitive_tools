"""Supplementary replicate-level evaluation utility and gross-harvest summaries."""

from collections import defaultdict

import numpy as np

from .env import reward_identity

METRICS = (
    "evaluation_mean_utility_sum",
    "evaluation_mean_utility_discounted",
    "evaluation_mean_harvest_sum",
    "evaluation_mean_harvest_discounted",
)


def build_utility_summaries(runs, *, bootstrap_reps, bootstrap_seed):
    from .analysis import bootstrap_mean_ci, primary_evaluation_rows, row_key, stable_seed

    records, summaries = [], []
    for run in runs:
        identity = reward_identity(run.config)
        groups = defaultdict(list)
        seen = set()
        primary = primary_evaluation_rows(run)
        if identity["mode"] == "capped_harvest":
            if not primary:
                raise ValueError(f"Missing capped primary evaluations in {run.run_id}")
            if all(key in run.config for key in ("scenarios", "populations", "replicates")):
                expected = {
                    (scenario, population, replicate)
                    for scenario in run.config["scenarios"]
                    for population in run.config["populations"]
                    for replicate in range(run.config["replicates"])
                }
                if {row_key(row) for row in primary} != expected:
                    raise ValueError(f"Incomplete capped primary evaluations in {run.run_id}")
        legacy = identity["mode"] == "harvest" and not any(
            metric in row for row in primary for metric in METRICS
        )
        for row in primary:
            key = row_key(row)
            if key in seen:
                raise ValueError(f"Duplicate primary evaluation replicate in {run.run_id}: {key}")
            seen.add(key)
            present = [metric in row for metric in METRICS]
            if legacy:
                continue  # Legacy results did not record evaluation reward streams.
            if not all(present):
                raise ValueError(f"Missing evaluation utility accounts in {run.run_id}")
            for metric in METRICS:
                value = float(row[metric])
                if not np.isfinite(value):
                    raise ValueError(f"Nonfinite evaluation account in {run.run_id}: {metric}")
                records.append(
                    dict(
                        run_id=run.run_id,
                        scenario=key[0],
                        population=key[1],
                        replicate=key[2],
                        reward_mode=identity["mode"],
                        reward_cap=identity["cap"],
                        metric=metric,
                        value=value,
                    )
                )
                groups[(key[0], key[1], metric)].append(value)
        for (scenario, population, metric), values in sorted(groups.items()):
            mean, low, high = bootstrap_mean_ci(
                values,
                bootstrap_reps=bootstrap_reps,
                seed=stable_seed(bootstrap_seed, run.run_id, scenario, population, metric),
            )
            summaries.append(
                dict(
                    run_id=run.run_id,
                    scenario=scenario,
                    population=population,
                    reward_mode=identity["mode"],
                    reward_cap=identity["cap"],
                    metric=metric,
                    n_replicates=len(values),
                    mean=mean,
                    low=low,
                    high=high,
                    interval="pointwise 95% percentile bootstrap",
                )
            )
    return records, summaries
