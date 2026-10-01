"""Prospective balanced-landscape hypotheses; independent replicates are units."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .analysis import bootstrap_mean_ci, stable_seed
from .scenarios import BALANCED_SCENARIOS
from .visibility import PROFILES


def _association(xs: np.ndarray, ys: np.ndarray) -> float:
    """Pearson association after centering within each social/ecology cell."""

    dx = xs - xs.mean(axis=1, keepdims=True)
    dy = ys - ys.mean(axis=1, keepdims=True)
    scale = float(np.sqrt(np.sum(dx * dx) * np.sum(dy * dy)))
    return float(np.sum(dx * dy) / scale) if scale > 0 else float("nan")


def _correlation(cross: np.ndarray, x_square: np.ndarray, y_square: np.ndarray) -> float:
    denominator = float(np.sqrt(np.sum(x_square) * np.sum(y_square)))
    return float(np.sum(cross) / denominator) if denominator > 0 else float("nan")


def _h1_checkpoints(runs, *, bootstrap_reps: int, bootstrap_seed: int):
    """Within-run co-movement; fixed graphs have no within-run degree variation."""

    pairs = []
    components = defaultdict(lambda: np.zeros(3, dtype=float))
    cell_coverage = defaultdict(set)
    for run in runs:
        if run.config.get("network_dynamics") != "adaptive_bounded":
            continue
        profile = run.config["visibility_profile"]
        grouped = defaultdict(list)
        for row in run.tables["training_timeseries"]:
            grouped[(row["scenario"], int(row["population"]), int(row["replicate"]))].append(row)
        for (scenario, population, replicate), rows in sorted(grouped.items()):
            if len(rows) < 2:
                raise ValueError("H1 longitudinal analysis requires two training checkpoints")
            times = np.array([int(row["time"]) for row in rows], dtype=float)
            x = np.array([float(row["visibility_gini"]) for row in rows], dtype=float)
            y = np.array([float(row["social_perception_error"]) for row in rows], dtype=float)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("H1 longitudinal checkpoints must have finite metrics")
            x -= x.mean()
            y -= y.mean()
            centered_time = times - times.mean()
            time_square = float(np.dot(centered_time, centered_time))
            if time_square <= 0:
                raise ValueError("H1 training checkpoint times must vary")
            adjusted_x = x - centered_time * float(np.dot(x, centered_time)) / time_square
            adjusted_y = y - centered_time * float(np.dot(y, centered_time)) / time_square
            for label, left, right in (
                ("within_run", x, y),
                ("linear_time_adjusted", adjusted_x, adjusted_y),
            ):
                components[(population, replicate, label)] += np.array(
                    [
                        float(np.dot(left, right)),
                        float(np.dot(left, left)),
                        float(np.dot(right, right)),
                    ]
                )
            cell_coverage[(population, replicate)].add((scenario, profile))
            for index, time in enumerate(times):
                pairs.append(
                    dict(
                        scenario=scenario,
                        profile=profile,
                        dynamics="adaptive_bounded",
                        population=population,
                        replicate=replicate,
                        time=int(time),
                        visibility_gini_deviation=float(x[index]),
                        local_view_error_deviation=float(y[index]),
                    )
                )
    expected = {(scenario, profile) for scenario in BALANCED_SCENARIOS for profile in PROFILES}
    if not cell_coverage or any(cells != expected for cells in cell_coverage.values()):
        raise ValueError("H1 longitudinal treatment coverage is incomplete")
    populations = sorted({population for population, _ in cell_coverage})
    summaries = []
    for population in populations:
        replicates = sorted(rep for n, rep in cell_coverage if n == population)
        for label in ("within_run", "linear_time_adjusted"):
            values = np.stack([components[(population, rep, label)] for rep in replicates])
            estimate = _correlation(values[:, 0], values[:, 1], values[:, 2])
            rng = np.random.default_rng(stable_seed(bootstrap_seed, "h1_checkpoints", label))
            indices = rng.integers(0, len(replicates), size=(bootstrap_reps, len(replicates)))
            totals = values[indices].sum(axis=1)
            denominator = np.sqrt(totals[:, 1] * totals[:, 2])
            draws = np.divide(
                totals[:, 0],
                denominator,
                out=np.full(len(denominator), float("nan")),
                where=denominator > 0,
            )
            draws = draws[np.isfinite(draws)]
            low, high = (
                (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))
                if len(draws)
                else (float("nan"), float("nan"))
            )
            summaries.append(
                dict(
                    hypothesis="H1",
                    population=population,
                    measure=label,
                    estimate=estimate,
                    low=low,
                    high=high,
                    n_replicates=len(replicates),
                    n_treatment_cells=len(expected),
                    n_checkpoints=sum(row["population"] == population for row in pairs),
                    interval="replicate-cluster bootstrap 95%",
                )
            )
    return pairs, summaries


def _h1(runs, *, bootstrap_reps: int, bootstrap_seed: int):
    run_means = []
    for run in runs:
        profile = run.config.get("visibility_profile")
        if profile not in PROFILES:
            continue
        dynamics = run.config["network_dynamics"]
        checkpoints = defaultdict(list)
        for row in run.tables["training_timeseries"]:
            gini = float(row["visibility_gini"])
            error = float(row["social_perception_error"])
            if math.isfinite(gini) and math.isfinite(error):
                checkpoints[
                    (row["scenario"], int(row["population"]), int(row["replicate"]))
                ].append((gini, error))
        for (scenario, population, replicate), values in sorted(checkpoints.items()):
            if len(values) < 2:
                raise ValueError("H1 requires at least two matched training checkpoints per run")
            run_means.append(
                dict(
                    scenario=scenario,
                    profile=profile,
                    dynamics=dynamics,
                    population=population,
                    replicate=replicate,
                    checkpoints=len(values),
                    mean_visibility_gini=float(np.mean([v[0] for v in values])),
                    mean_local_view_error=float(np.mean([v[1] for v in values])),
                )
            )

    by_population = defaultdict(list)
    for row in run_means:
        by_population[row["population"]].append(row)
    summaries = []
    for population, rows in sorted(by_population.items()):
        cells = defaultdict(dict)
        for row in rows:
            key = (row["scenario"], row["profile"], row["dynamics"])
            replicate = row["replicate"]
            if replicate in cells[key]:
                raise ValueError("Duplicate H1 treatment/replicate mean")
            cells[key][replicate] = row
        replicates = sorted({row["replicate"] for row in rows})
        if any(set(cell) != set(replicates) for cell in cells.values()):
            raise ValueError("H1 treatment cells have unmatched replicates")
        ordered = [cells[key] for key in sorted(cells)]
        xs = np.array(
            [[cell[rep]["mean_visibility_gini"] for rep in replicates] for cell in ordered]
        )
        ys = np.array(
            [[cell[rep]["mean_local_view_error"] for rep in replicates] for cell in ordered]
        )
        estimate = _association(xs, ys)
        rng = np.random.default_rng(stable_seed(bootstrap_seed, "h1", population))
        draws = []
        for _ in range(bootstrap_reps):
            indices = rng.integers(0, len(replicates), size=len(replicates))
            value = _association(xs[:, indices], ys[:, indices])
            if math.isfinite(value):
                draws.append(value)
        low, high = (
            (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))
            if draws
            else (float("nan"), float("nan"))
        )
        summaries.append(
            dict(
                hypothesis="H1",
                population=population,
                measure="within_treatment_centered_pearson_r",
                estimate=estimate,
                low=low,
                high=high,
                n_replicates=len(replicates),
                n_treatment_cells=len(ordered),
                interval="replicate-cluster bootstrap 95%",
            )
        )
    return run_means, summaries


def _pooled_contrast(
    rows,
    *,
    hypothesis: str,
    comparison: str,
    expected_cells: set[tuple[str, str]],
    bootstrap_reps: int,
    bootstrap_seed: int,
):
    grouped = defaultdict(dict)
    populations = set()
    for row in rows:
        if row["comparison"] != comparison or row["metric"] != "social_perception_error":
            continue
        key = (
            (row["profile"], row["dynamics"])
            if hypothesis == "H2"
            else (row["scenario"], row["profile"])
        )
        if key not in expected_cells:
            continue
        populations.add(int(row["population"]))
        replicate = int(row["replicate"])
        if key in grouped[replicate]:
            raise ValueError(f"Duplicate {hypothesis} paired contrast cell")
        grouped[replicate][key] = float(row["value"])
    if not grouped:
        raise ValueError(f"Missing {hypothesis} paired contrasts")
    if len(populations) != 1:
        raise ValueError(f"{hypothesis} requires one population size per analysis")
    population = populations.pop()
    replicate_rows = []
    for replicate, cells in sorted(grouped.items()):
        if set(cells) != expected_cells or not all(math.isfinite(v) for v in cells.values()):
            raise ValueError(f"Incomplete {hypothesis} paired contrasts for replicate {replicate}")
        replicate_rows.append(
            dict(
                hypothesis=hypothesis,
                comparison=comparison,
                population=population,
                replicate=replicate,
                metric="social_perception_error",
                value=float(np.mean(list(cells.values()))),
                n_cells=len(cells),
            )
        )
    values = [row["value"] for row in replicate_rows]
    mean, low, high = bootstrap_mean_ci(
        values,
        bootstrap_reps=bootstrap_reps,
        seed=stable_seed(bootstrap_seed, hypothesis, comparison),
    )
    summary = dict(
        hypothesis=hypothesis,
        comparison=comparison,
        population=population,
        metric="social_perception_error",
        mean=mean,
        low=low,
        high=high,
        n_replicates=len(values),
        n_cells=len(expected_cells),
        interval="replicate-bootstrap 95%",
    )
    return replicate_rows, summary


def analyze_balanced_hypotheses(runs, ecology, adaptation, *, bootstrap_reps, bootstrap_seed):
    """Return H1 association and paired, treatment-averaged H2/H3 contrasts."""

    run_means, h1_summary = _h1(runs, bootstrap_reps=bootstrap_reps, bootstrap_seed=bootstrap_seed)
    checkpoint_pairs, checkpoint_summary = _h1_checkpoints(
        runs, bootstrap_reps=bootstrap_reps, bootstrap_seed=bootstrap_seed
    )
    h2_rows, h2_summary = _pooled_contrast(
        ecology,
        hypothesis="H2",
        comparison="balanced_segregated-balanced_dispersed",
        expected_cells={
            (profile, dynamics)
            for profile in PROFILES
            for dynamics in ("fixed", "adaptive_bounded")
        },
        bootstrap_reps=bootstrap_reps,
        bootstrap_seed=bootstrap_seed,
    )
    h3_rows, h3_summary = _pooled_contrast(
        adaptation,
        hypothesis="H3",
        comparison="adaptive-fixed",
        expected_cells={
            (scenario, profile) for scenario in BALANCED_SCENARIOS for profile in PROFILES
        },
        bootstrap_reps=bootstrap_reps,
        bootstrap_seed=bootstrap_seed,
    )
    return dict(
        h1_run_means=run_means,
        h1_association_summary=h1_summary,
        h1_checkpoint_pairs=checkpoint_pairs,
        h1_checkpoint_association_summary=checkpoint_summary,
        pooled_hypothesis_replicates=h2_rows + h3_rows,
        pooled_hypothesis_summary=[h2_summary, h3_summary],
    )
