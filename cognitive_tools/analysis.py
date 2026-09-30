from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .env import reward_identity

ANALYSIS_ROOT = Path("results") / "q_learning_baseline" / "social_analysis"
ECOLOGICAL_STATES = ("scarce", "moderate", "abundant")
SOCIAL_STATES = ("mostly_high", "mixed", "mostly_low")
PRIMARY_EVALUATION_MODE = "fresh_reset"
PRIMARY_STRATEGY = "q_learning"
PRIMARY_RESOURCE_METRIC = "eval_mean_mean_resource_fraction"
FINAL_RESOURCE_METRIC = "final_mean_resource_fraction"

TRAJECTORY_METRICS = (
    "visibility_gini",
    "mean_perception_error",
    "degree_action_correlation",
    "majority_mismatch_rate",
    "majority_tie_rate",
    "reciprocity",
    "degree_assortativity",
)

PAIRED_METRICS = (
    ("eval_mean_mean_resource_fraction", "resource_fraction"),
    ("eval_mean_mean_reserve_welfare", "reserve_welfare"),
    ("eval_mean_mean_need_satisfaction", "need_satisfaction"),
    ("eval_mean_deprivation_rate", "deprivation_rate"),
    ("final_wealth_gini", "wealth_gini"),
    ("eval_mean_social_perception_error", "perception_error"),
    ("eval_mean_visibility_gini", "visibility_gini"),
    ("eval_mean_majority_mismatch_rate", "majority_mismatch_rate"),
)

MEMORY_METRICS = (
    ("eval_mean_mean_resource_fraction", "resource_fraction"),
    ("eval_mean_mean_reserve_welfare", "reserve_welfare"),
    ("final_wealth_gini", "wealth_gini"),
    ("eval_mean_social_perception_error", "perception_error"),
)

STRICT_COMPATIBILITY_KEYS = (
    "seed",
    "width",
    "height",
    "coupling",
    "low_harvest",
    "high_harvest",
    "metabolism",
    "initial_energy",
    "energy_capacity",
    "initial_resource_fraction",
    "alpha",
    "gamma",
    "epsilon",
    "epsilon_min",
    "epsilon_decay",
    "training_steps",
    "evaluation_steps",
    "record_every",
    "record_network_every",
    "rewire_every",
    "rewire_threshold",
    "forecast_alpha",
)


@dataclass
class RunData:
    path: Path
    config: dict
    run_id: str
    treatment: str
    theta: float
    mu: float
    config_sha256: str
    schedule_sha256: str | None
    matched_schedule_sha256: str | None
    tables: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    label: str = ""
    paired_adaptive_run_id: str | None = None


# -----------------------------------------------------------------------------
# Basic file/scalar helpers
# -----------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path, *, required: bool) -> list[dict[str, str]]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required analysis input is missing: {path}")
        return []
    with path.open(newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def as_float(row: dict, name: str, *, default: float = float("nan")) -> float:
    value = row.get(name)
    if value in (None, "", "None", "nan", "NaN"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(row: dict, name: str, *, default: int = 0) -> int:
    value = row.get(name)
    if value in (None, "", "None"):
        return default
    return int(float(value))


def stable_seed(base_seed: int, *parts) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)]).encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % (2**32 - 1)


def numeric_values(rows: list[dict], metric: str) -> list[float]:
    values = [as_float(row, metric) for row in rows]
    return [value for value in values if np.isfinite(value)]


def bootstrap_mean_ci(
    values, *, bootstrap_reps: int, seed: int, confidence: float = 0.95
) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(array))
    if len(array) == 1 or bootstrap_reps <= 0:
        return mean, mean, mean

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(bootstrap_reps, len(array)))
    means = np.mean(array[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return mean, float(low), float(high)


def format_parameter(value: float) -> str:
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def safe_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)


def row_key(row: dict) -> tuple[str, int, int]:
    return str(row.get("scenario")), as_int(row, "population"), as_int(row, "replicate")


# -----------------------------------------------------------------------------
# Loading, labels, compatibility and exact R0 pairing
# -----------------------------------------------------------------------------


def load_run(path: Path) -> RunData:
    run_dir = path.expanduser().resolve()
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run has no config.json: {run_dir}")

    with config_path.open() as file:
        config = json.load(file)

    reward_identity(config)
    if config.get("reward_mode") == "capped_harvest" and "reward_definition" not in config:
        raise ValueError("Capped runs require explicit reward_definition metadata")
    schema = config.get("social_measurement_schema")
    if schema != "stage4_v1":
        raise ValueError(
            f"Run {run_dir.name} does not use the Stage 4 social measurement "
            f"schema. Found {schema!r}. Re-run it with the current experiment runner."
        )

    data_dir = run_dir / "data"
    tables: dict[str, list[dict[str, str]]] = {}
    required_tables = (
        "evaluation_summary",
        "training_timeseries",
        "policy_summary",
        "agent_social_summary",
    )
    optional_tables = (
        "evaluation_timeseries",
        "agent_policies",
        "network_timeseries",
        "rewiring_schedule",
        "network_edges_checkpoints",
    )
    for name in required_tables:
        tables[name] = read_csv_rows(data_dir / f"{name}.csv", required=True)
    for name in optional_tables:
        tables[name] = read_csv_rows(data_dir / f"{name}.csv", required=False)

    schedule_path = data_dir / "rewiring_schedule.csv"
    schedule_sha = sha256_file(schedule_path) if schedule_path.is_file() else None
    return RunData(
        path=run_dir,
        config=config,
        run_id=run_dir.name,
        treatment=str(config.get("treatment", "unknown")),
        theta=float(config.get("rewire_theta", float("nan"))),
        mu=float(config.get("rewire_mu", float("nan"))),
        config_sha256=sha256_file(config_path),
        schedule_sha256=schedule_sha,
        matched_schedule_sha256=config.get("matched_rewire_schedule_sha256"),
        tables=tables,
    )


def assign_run_labels(runs: list[RunData]) -> None:
    mus_by_treatment: dict[tuple[str, float], set[float]] = defaultdict(set)
    for run in runs:
        if run.treatment in {"R1", "R2", "R3", "R_adaptive"}:
            mus_by_treatment[(run.treatment, run.theta)].add(run.mu)

    for run in runs:
        if run.treatment in {"B0", "S1", "S2"}:
            run.label = run.treatment
        elif run.treatment == "R0":
            run.label = f"R0 (theta={format_parameter(run.theta)})"
        elif run.treatment in {"R1", "R2", "R3"}:
            if len(mus_by_treatment[(run.treatment, run.theta)]) > 1:
                run.label = f"{run.treatment} (mu={format_parameter(run.mu)})"
            else:
                run.label = run.treatment
        elif run.treatment == "R_adaptive":
            run.label = f"R (theta={format_parameter(run.theta)}, mu={format_parameter(run.mu)})"
        else:
            run.label = run.treatment


def treatment_sort_key(run: RunData):
    if run.treatment == "B0":
        return 0, 0.0, 0.0, run.label
    if run.treatment == "S1":
        return 1, 0.0, 0.0, run.label
    if run.treatment == "S2":
        return 2, 0.0, 0.0, run.label
    if run.treatment == "R0":
        return 3, run.theta, 0.0, run.label
    if run.treatment in {"R1", "R2", "R3", "R_adaptive"}:
        return 3, run.theta, 1.0 + run.mu, run.label
    return 9, run.theta, run.mu, run.label


def run_label_order(runs: list[RunData]) -> list[str]:
    labels: list[str] = []
    for run in sorted(runs, key=treatment_sort_key):
        if run.label not in labels:
            labels.append(run.label)
    return labels


def validate_compatibility(runs: list[RunData]) -> list[str]:
    if not runs:
        raise ValueError("At least one run directory is required.")

    run_ids = [run.run_id for run in runs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError(
            "Run directory basenames must be unique because they are used as "
            "analysis IDs. Rename or relocate duplicate-basename inputs."
        )

    warnings: list[str] = []
    reference = runs[0]
    expected_reward = reward_identity(reference.config)
    if any(reward_identity(run.config) != expected_reward for run in runs[1:]):
        raise ValueError("Cross-treatment analysis would mix incompatible reward definitions")

    commit_shas = {
        run.config.get("run_metadata", {}).get("git_commit_sha")
        for run in runs
        if run.config.get("run_metadata", {}).get("git_commit_sha")
    }
    if len(commit_shas) > 1:
        raise ValueError(
            "Cross-treatment analysis would mix experiment runs from "
            f"different Git commits: {sorted(commit_shas)}."
        )

    dirty_runs = [
        run.run_id
        for run in runs
        if run.config.get("run_metadata", {}).get("git_worktree_dirty") is True
    ]
    if dirty_runs:
        warnings.append(
            "Input runs recorded a dirty Git worktree: " + ", ".join(sorted(dirty_runs))
        )

    for key in STRICT_COMPATIBILITY_KEYS:
        expected = reference.config.get(key)
        for run in runs[1:]:
            actual = run.config.get(key)
            if actual != expected:
                raise ValueError(
                    "Cross-treatment analysis would mix incompatible settings for "
                    f"{key!r}: {reference.run_id}={expected!r}, "
                    f"{run.run_id}={actual!r}."
                )

    random_k_values = {
        int(run.config.get("social_k", 0))
        for run in runs
        if run.config.get("social_mode") == "fixed"
        and run.config.get("social_network") == "random_k"
    }
    if len(random_k_values) > 1:
        raise ValueError(
            "Cross-treatment analysis would mix different random_k attention "
            f"capacities: {sorted(random_k_values)}."
        )

    ba_values = {
        int(run.config.get("ba_m", 0))
        for run in runs
        if run.config.get("social_mode") == "fixed" and run.config.get("social_network") == "ba"
    }
    if len(ba_values) > 1:
        raise ValueError(
            f"Cross-treatment analysis would mix different BA m values: {sorted(ba_values)}."
        )

    if len({tuple(run.config.get("scenarios", [])) for run in runs}) > 1:
        warnings.append(
            "Input runs do not all contain the same scenarios; available cells will be used."
        )
    if len({tuple(run.config.get("populations", [])) for run in runs}) > 1:
        warnings.append(
            "Input runs do not all contain the same populations; available cells will be used."
        )
    if len({int(run.config.get("replicates", 0)) for run in runs}) > 1:
        warnings.append(
            "Input runs have different configured replicate counts; summaries report actual n."
        )
    return warnings


def resolve_r0_pairs(runs: list[RunData]) -> list[str]:
    """Pair each loaded R0 with its exact adaptive schedule source when possible."""
    warnings: list[str] = []
    adaptive = [run for run in runs if run.config.get("rewiring") == "prediction_error"]
    by_schedule_hash: dict[str, list[RunData]] = defaultdict(list)
    for run in adaptive:
        if run.schedule_sha256:
            by_schedule_hash[run.schedule_sha256].append(run)

    for r0 in [run for run in runs if run.treatment == "R0"]:
        candidates: list[RunData] = []
        if r0.matched_schedule_sha256:
            candidates = by_schedule_hash.get(str(r0.matched_schedule_sha256), [])
        if not candidates:
            candidates = [
                run for run in adaptive if np.isclose(run.theta, r0.theta, rtol=0.0, atol=1e-12)
            ]
        if len(candidates) == 1:
            source = candidates[0]
            r0.paired_adaptive_run_id = source.run_id
            r0.label = (
                f"R0 (theta={format_parameter(r0.theta)}, matched mu={format_parameter(source.mu)})"
            )
        elif not candidates:
            warnings.append(
                f"R0 run {r0.run_id} has no loaded adaptive source run; paired effects are skipped."
            )
        else:
            warnings.append(
                f"R0 run {r0.run_id} has multiple possible adaptive source runs. "
                "Include the exact source run so its schedule hash resolves the pair."
            )
    return warnings


def run_catalog_rows(runs: list[RunData]) -> list[dict]:
    output = []
    for run in sorted(runs, key=treatment_sort_key):
        metadata = run.config.get("run_metadata", {})
        output.append(
            {
                "run_id": run.run_id,
                "run_path": str(run.path),
                "analysis_label": run.label,
                "treatment": run.treatment,
                "rewiring": run.config.get("rewiring"),
                "rewire_theta": run.theta,
                "rewire_mu": run.mu,
                "network_eval": run.config.get("network_eval"),
                "scenarios": "|".join(map(str, run.config.get("scenarios", []))),
                "populations": "|".join(map(str, run.config.get("populations", []))),
                "replicates": run.config.get("replicates"),
                "config_sha256": run.config_sha256,
                "rewiring_schedule_sha256": run.schedule_sha256,
                "matched_source_schedule_sha256": run.matched_schedule_sha256,
                "paired_adaptive_run_id": run.paired_adaptive_run_id,
                "git_commit_sha": metadata.get("git_commit_sha"),
                "git_worktree_dirty": metadata.get("git_worktree_dirty"),
            }
        )
    return output


def annotate_row(run: RunData, row: dict) -> dict:
    return {
        **row,
        "_run_id": run.run_id,
        "_run_path": str(run.path),
        "_treatment": run.treatment,
        "_analysis_label": run.label,
        "_theta": run.theta,
        "_mu": run.mu,
    }


def primary_evaluation_rows(run: RunData) -> list[dict]:
    return [
        annotate_row(run, row)
        for row in run.tables["evaluation_summary"]
        if row.get("strategy") == PRIMARY_STRATEGY
        and row.get("evaluation_mode") == PRIMARY_EVALUATION_MODE
    ]


# -----------------------------------------------------------------------------
# Core analysis tables
# -----------------------------------------------------------------------------


def resource_regime(value: float, *, low_threshold: float, high_threshold: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < low_threshold:
        return "low"
    if value >= high_threshold:
        return "high"
    return "middle"


def build_resource_distribution(
    runs: list[RunData],
    *,
    low_threshold: float,
    high_threshold: float,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    for run in runs:
        for source in primary_evaluation_rows(run):
            final_resource = as_float(source, FINAL_RESOURCE_METRIC)
            rows.append(
                {
                    "run_id": run.run_id,
                    "analysis_label": run.label,
                    "treatment": run.treatment,
                    "rewire_theta": run.theta,
                    "rewire_mu": run.mu,
                    "scenario": source.get("scenario"),
                    "population": as_int(source, "population"),
                    "replicate": as_int(source, "replicate"),
                    "evaluation_mode": PRIMARY_EVALUATION_MODE,
                    "eval_mean_resource_fraction": as_float(source, PRIMARY_RESOURCE_METRIC),
                    "final_resource_fraction": final_resource,
                    "resource_regime": resource_regime(
                        final_resource, low_threshold=low_threshold, high_threshold=high_threshold
                    ),
                }
            )

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["analysis_label"],
            row["treatment"],
            row["rewire_theta"],
            row["rewire_mu"],
            row["scenario"],
            row["population"],
        )
        groups[key].append(row)

    summary: list[dict] = []
    for key, group in groups.items():
        label, treatment, theta, mu, scenario, population = key
        values = [row["eval_mean_resource_fraction"] for row in group]
        mean, low, high = bootstrap_mean_ci(
            values,
            bootstrap_reps=bootstrap_reps,
            seed=stable_seed(bootstrap_seed, "resource", *key),
        )
        finite_final = [
            row["final_resource_fraction"]
            for row in group
            if np.isfinite(row["final_resource_fraction"])
        ]
        counts = {
            regime: sum(row["resource_regime"] == regime for row in group)
            for regime in ("low", "middle", "high")
        }
        n = len(group)
        summary.append(
            {
                "analysis_label": label,
                "treatment": treatment,
                "rewire_theta": theta,
                "rewire_mu": mu,
                "scenario": scenario,
                "population": population,
                "n": n,
                "mean_eval_resource_fraction": mean,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "median_final_resource_fraction": (
                    float(np.median(finite_final)) if finite_final else float("nan")
                ),
                "prob_low_resource": counts["low"] / n if n else float("nan"),
                "prob_middle_resource": counts["middle"] / n if n else float("nan"),
                "prob_high_resource": counts["high"] / n if n else float("nan"),
            }
        )
    return rows, summary


def build_policy_heatmap_summary(
    runs: list[RunData], *, bootstrap_reps: int, bootstrap_seed: int
) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for run in runs:
        if run.config.get("social_mode") != "fixed":
            continue
        for row in run.tables["policy_summary"]:
            key = (
                run.label,
                run.treatment,
                run.theta,
                run.mu,
                row.get("scenario"),
                as_int(row, "population"),
            )
            grouped[key].append(row)

    output: list[dict] = []
    for key, rows in grouped.items():
        label, treatment, theta, mu, scenario, population = key
        for ecological in ECOLOGICAL_STATES:
            for social in SOCIAL_STATES:
                behavior_metric = f"policy_low_{ecological}_{social}"
                occupancy_metric = f"training_visit_fraction_{ecological}_{social}"
                behavior = bootstrap_mean_ci(
                    numeric_values(rows, behavior_metric),
                    bootstrap_reps=bootstrap_reps,
                    seed=stable_seed(bootstrap_seed, "policy", *key, ecological, social),
                )
                occupancy = bootstrap_mean_ci(
                    numeric_values(rows, occupancy_metric),
                    bootstrap_reps=bootstrap_reps,
                    seed=stable_seed(bootstrap_seed, "occupancy", *key, ecological, social),
                )
                output.append(
                    {
                        "analysis_label": label,
                        "treatment": treatment,
                        "rewire_theta": theta,
                        "rewire_mu": mu,
                        "scenario": scenario,
                        "population": population,
                        "ecological_state": ecological,
                        "social_state": social,
                        "n": len(rows),
                        "policy_low_mean": behavior[0],
                        "policy_low_ci_low": behavior[1],
                        "policy_low_ci_high": behavior[2],
                        "occupancy_mean": occupancy[0],
                        "occupancy_ci_low": occupancy[1],
                        "occupancy_ci_high": occupancy[2],
                    }
                )
    return output


def build_trajectory_summary(
    runs: list[RunData], *, bootstrap_reps: int, bootstrap_seed: int
) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    metadata: dict[tuple, tuple] = {}
    for run in runs:
        for row in run.tables["network_timeseries"]:
            scenario = row.get("scenario")
            population = as_int(row, "population")
            time = as_int(row, "time")
            for metric in TRAJECTORY_METRICS:
                value = as_float(row, metric)
                if not np.isfinite(value):
                    continue
                key = (run.label, scenario, population, time, metric)
                grouped[key].append(value)
                metadata[key] = (run.treatment, run.theta, run.mu)

    output: list[dict] = []
    for key, values in grouped.items():
        label, scenario, population, time, metric = key
        treatment, theta, mu = metadata[key]
        mean, low, high = bootstrap_mean_ci(
            values,
            bootstrap_reps=bootstrap_reps,
            seed=stable_seed(bootstrap_seed, "trajectory", *key),
        )
        output.append(
            {
                "analysis_label": label,
                "treatment": treatment,
                "rewire_theta": theta,
                "rewire_mu": mu,
                "scenario": scenario,
                "population": population,
                "time": time,
                "metric": metric,
                "n": len(values),
                "mean": mean,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
            }
        )
    return sorted(
        output,
        key=lambda row: (
            str(row["metric"]),
            str(row["scenario"]),
            int(row["population"]),
            str(row["analysis_label"]),
            int(row["time"]),
        ),
    )


def build_terminal_training_rows(runs: list[RunData], resource_rows: list[dict]) -> list[dict]:
    resource_index = {
        (row["run_id"], row["scenario"], int(row["population"]), int(row["replicate"])): row
        for row in resource_rows
    }
    output: list[dict] = []

    for run in runs:
        training_groups: dict[tuple, list[dict]] = defaultdict(list)
        network_groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in run.tables["training_timeseries"]:
            training_groups[row_key(row)].append(row)
        for row in run.tables["network_timeseries"]:
            network_groups[row_key(row)].append(row)

        for key, rows in training_groups.items():
            scenario, population, replicate = key
            terminal = max(rows, key=lambda row: as_int(row, "time"))
            network_rows = network_groups.get(key, [])
            turnovers = [
                as_float(row, "edge_turnover")
                for row in network_rows
                if as_int(row, "time") > 0 and np.isfinite(as_float(row, "edge_turnover"))
            ]
            final_network = (
                max(network_rows, key=lambda row: as_int(row, "time")) if network_rows else None
            )
            resource = resource_index.get((run.run_id, scenario, population, replicate))
            output.append(
                {
                    "run_id": run.run_id,
                    "analysis_label": run.label,
                    "treatment": run.treatment,
                    "rewire_theta": run.theta,
                    "rewire_mu": run.mu,
                    "scenario": scenario,
                    "population": population,
                    "replicate": replicate,
                    "training_time": as_int(terminal, "time"),
                    "training_resource_fraction": as_float(terminal, "mean_resource_fraction"),
                    "training_wealth_gini": as_float(terminal, "wealth_gini"),
                    "training_visibility_gini": as_float(terminal, "visibility_gini"),
                    "training_perception_error": as_float(terminal, "social_perception_error"),
                    "training_degree_action_correlation": as_float(
                        terminal, "degree_action_correlation"
                    ),
                    "mean_edge_turnover": (
                        float(np.mean(turnovers)) if turnovers else float("nan")
                    ),
                    "sum_edge_turnover": (float(np.sum(turnovers)) if turnovers else float("nan")),
                    "cumulative_rewires": (
                        as_int(final_network, "cumulative_rewires") if final_network else 0
                    ),
                    "eval_mean_resource_fraction": (
                        resource["eval_mean_resource_fraction"] if resource else float("nan")
                    ),
                    "eval_final_resource_fraction": (
                        resource["final_resource_fraction"] if resource else float("nan")
                    ),
                }
            )
    return output


# -----------------------------------------------------------------------------
# Paired R0 effects and network-memory decomposition
# -----------------------------------------------------------------------------


def index_primary_rows(run: RunData) -> dict[tuple[str, int, int], dict]:
    index = {}
    for row in primary_evaluation_rows(run):
        key = row_key(row)
        if key in index:
            raise ValueError(f"Duplicate primary evaluation row in {run.run_id}: {key}")
        index[key] = row
    return index


def build_paired_effects(
    runs: list[RunData], *, bootstrap_reps: int, bootstrap_seed: int
) -> tuple[list[dict], list[dict]]:
    run_by_id = {run.run_id: run for run in runs}
    paired_rows: list[dict] = []

    for r0 in runs:
        if r0.treatment != "R0" or not r0.paired_adaptive_run_id:
            continue
        adaptive = run_by_id.get(r0.paired_adaptive_run_id)
        if adaptive is None:
            continue
        adaptive_index = index_primary_rows(adaptive)
        r0_index = index_primary_rows(r0)

        for key in sorted(set(adaptive_index) & set(r0_index)):
            scenario, population, replicate = key
            for metric, metric_label in PAIRED_METRICS:
                adaptive_value = as_float(adaptive_index[key], metric)
                r0_value = as_float(r0_index[key], metric)
                if not (np.isfinite(adaptive_value) and np.isfinite(r0_value)):
                    continue
                paired_rows.append(
                    {
                        "adaptive_run_id": adaptive.run_id,
                        "r0_run_id": r0.run_id,
                        "adaptive_label": adaptive.label,
                        "r0_label": r0.label,
                        "rewire_theta": adaptive.theta,
                        "rewire_mu": adaptive.mu,
                        "scenario": scenario,
                        "population": population,
                        "replicate": replicate,
                        "metric": metric_label,
                        "adaptive_value": adaptive_value,
                        "r0_value": r0_value,
                        "adaptive_minus_r0": adaptive_value - r0_value,
                    }
                )

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in paired_rows:
        key = (
            row["adaptive_run_id"],
            row["r0_run_id"],
            row["adaptive_label"],
            row["r0_label"],
            row["rewire_theta"],
            row["rewire_mu"],
            row["scenario"],
            row["population"],
            row["metric"],
        )
        groups[key].append(row)

    summary: list[dict] = []
    for key, rows in groups.items():
        mean, low, high = bootstrap_mean_ci(
            [row["adaptive_minus_r0"] for row in rows],
            bootstrap_reps=bootstrap_reps,
            seed=stable_seed(bootstrap_seed, "paired", *key),
        )
        (
            adaptive_run_id,
            r0_run_id,
            adaptive_label,
            r0_label,
            theta,
            mu,
            scenario,
            population,
            metric,
        ) = key
        summary.append(
            {
                "adaptive_run_id": adaptive_run_id,
                "r0_run_id": r0_run_id,
                "adaptive_label": adaptive_label,
                "r0_label": r0_label,
                "rewire_theta": theta,
                "rewire_mu": mu,
                "scenario": scenario,
                "population": population,
                "metric": metric,
                "n_pairs": len(rows),
                "mean_adaptive_minus_r0": mean,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
            }
        )
    return paired_rows, summary


def build_network_memory_effects(
    runs: list[RunData], *, bootstrap_reps: int, bootstrap_seed: int
) -> tuple[list[dict], list[dict], list[str]]:
    rows: list[dict] = []
    warnings: list[str] = []

    for run in runs:
        carried: dict[tuple, dict] = {}
        reset: dict[tuple, dict] = {}
        for row in run.tables["evaluation_summary"]:
            if row.get("strategy") != "q_learning":
                continue
            if row.get("evaluation_mode") == "fresh_reset":
                carried[row_key(row)] = row
            elif row.get("evaluation_mode") == "fresh_reset_network":
                reset[row_key(row)] = row

        for key in sorted(set(carried) & set(reset)):
            scenario, population, replicate = key
            for metric, metric_label in MEMORY_METRICS:
                carried_value = as_float(carried[key], metric)
                reset_value = as_float(reset[key], metric)
                if not (np.isfinite(carried_value) and np.isfinite(reset_value)):
                    continue
                difference = carried_value - reset_value
                rows.append(
                    {
                        "run_id": run.run_id,
                        "analysis_label": run.label,
                        "treatment": run.treatment,
                        "rewire_theta": run.theta,
                        "rewire_mu": run.mu,
                        "scenario": scenario,
                        "population": population,
                        "replicate": replicate,
                        "metric": metric_label,
                        "carried_terminal_value": carried_value,
                        "reset_initial_value": reset_value,
                        "carried_minus_reset": difference,
                    }
                )
                if run.treatment in {"S1", "S2"} and abs(difference) > 1e-12:
                    warnings.append(
                        f"Fixed-network lifecycle check failed for {run.run_id}, "
                        f"{key}, {metric_label}: carried-reset={difference}."
                    )

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["analysis_label"],
            row["treatment"],
            row["rewire_theta"],
            row["rewire_mu"],
            row["scenario"],
            row["population"],
            row["metric"],
        )
        groups[key].append(row)

    summary: list[dict] = []
    for key, group in groups.items():
        mean, low, high = bootstrap_mean_ci(
            [row["carried_minus_reset"] for row in group],
            bootstrap_reps=bootstrap_reps,
            seed=stable_seed(bootstrap_seed, "memory", *key),
        )
        label, treatment, theta, mu, scenario, population, metric = key
        summary.append(
            {
                "analysis_label": label,
                "treatment": treatment,
                "rewire_theta": theta,
                "rewire_mu": mu,
                "scenario": scenario,
                "population": population,
                "metric": metric,
                "n_pairs": len(group),
                "mean_carried_minus_reset": mean,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
            }
        )
    return rows, summary, warnings


# -----------------------------------------------------------------------------
# Theta x mu summaries
# -----------------------------------------------------------------------------


def build_phase_summary(
    runs: list[RunData],
    resource_rows: list[dict],
    terminal_rows: list[dict],
    *,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> list[dict]:
    adaptive_ids = {run.run_id for run in runs if run.config.get("rewiring") == "prediction_error"}
    raw: list[dict] = []

    for row in resource_rows:
        if row["run_id"] in adaptive_ids and np.isfinite(row["eval_mean_resource_fraction"]):
            raw.append(
                {
                    **{
                        key: row[key]
                        for key in (
                            "run_id",
                            "rewire_theta",
                            "rewire_mu",
                            "scenario",
                            "population",
                            "replicate",
                        )
                    },
                    "metric": "resource_fraction",
                    "value": row["eval_mean_resource_fraction"],
                }
            )

    for row in terminal_rows:
        if row["run_id"] in adaptive_ids and np.isfinite(row["training_visibility_gini"]):
            raw.append(
                {
                    **{
                        key: row[key]
                        for key in (
                            "run_id",
                            "rewire_theta",
                            "rewire_mu",
                            "scenario",
                            "population",
                            "replicate",
                        )
                    },
                    "metric": "visibility_gini",
                    "value": row["training_visibility_gini"],
                }
            )

    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in raw:
        key = (
            row["rewire_theta"],
            row["rewire_mu"],
            row["scenario"],
            row["population"],
            row["metric"],
        )
        groups[key].append(row["value"])

    output: list[dict] = []
    for key, values in groups.items():
        theta, mu, scenario, population, metric = key
        mean, low, high = bootstrap_mean_ci(
            values, bootstrap_reps=bootstrap_reps, seed=stable_seed(bootstrap_seed, "phase", *key)
        )
        output.append(
            {
                "rewire_theta": theta,
                "rewire_mu": mu,
                "scenario": scenario,
                "population": population,
                "metric": metric,
                "n": len(values),
                "mean": mean,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
            }
        )
    return output


def build_agent_scatter_rows(runs: list[RunData]) -> list[dict]:
    return [
        {
            **row,
            "analysis_label": run.label,
            "treatment": run.treatment,
            "rewire_theta": run.theta,
            "rewire_mu": run.mu,
        }
        for run in runs
        for row in run.tables["agent_social_summary"]
    ]


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def scenario_population_grid(
    scenarios: list[str],
    populations: list[int],
    *,
    width_per_column: float = 5.0,
    height_per_row: float = 3.8,
):
    return plt.subplots(
        max(1, len(scenarios)),
        max(1, len(populations)),
        figsize=(
            width_per_column * max(1, len(populations)),
            height_per_row * max(1, len(scenarios)),
        ),
        squeeze=False,
        layout="constrained",
    )


def save_resource_distribution_figure(
    rows: list[dict], runs: list[RunData], figures_dir: Path, *, bootstrap_seed: int
) -> None:
    if not rows:
        return
    scenarios = sorted({str(row["scenario"]) for row in rows})
    populations = sorted({int(row["population"]) for row in rows})
    labels = run_label_order(runs)
    figure, axes = scenario_population_grid(
        scenarios, populations, width_per_column=6.0, height_per_row=4.0
    )

    for i, scenario in enumerate(scenarios):
        for j, population in enumerate(populations):
            axis = axes[i, j]
            data, positions = [], []
            for position, label in enumerate(labels, start=1):
                values = [
                    float(row["eval_mean_resource_fraction"])
                    for row in rows
                    if row["scenario"] == scenario
                    and int(row["population"]) == population
                    and row["analysis_label"] == label
                    and np.isfinite(row["eval_mean_resource_fraction"])
                ]
                if not values:
                    continue
                data.append(values)
                positions.append(position)
                jitter = np.random.default_rng(
                    stable_seed(bootstrap_seed, "jitter", scenario, population, label)
                ).normal(0.0, 0.045, size=len(values))
                axis.scatter(
                    np.full(len(values), position, dtype=float) + jitter, values, s=16, alpha=0.65
                )
            if data:
                axis.boxplot(data, positions=positions, widths=0.55, showfliers=False)
            axis.set_xticks(range(1, len(labels) + 1))
            axis.set_xticklabels(labels, rotation=45, ha="right")
            axis.set_ylim(0.0, 1.0)
            axis.grid(alpha=0.25, axis="y")
            axis.set_title(f"{scenario} | N={population}")
            if j == 0:
                axis.set_ylabel("Evaluation mean resource fraction")

    path = figures_dir / "01_resource_outcome_distributions.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {path}")


def heatmap_matrix(
    rows: list[dict], *, label: str, scenario: str, population: int, value_field: str
) -> np.ndarray:
    matrix = np.full((3, 3), np.nan, dtype=float)
    for row in rows:
        if (
            row["analysis_label"] == label
            and row["scenario"] == scenario
            and int(row["population"]) == population
        ):
            matrix[
                ECOLOGICAL_STATES.index(row["ecological_state"]),
                SOCIAL_STATES.index(row["social_state"]),
            ] = float(row[value_field])
    return matrix


def save_policy_heatmaps(
    rows: list[dict],
    runs: list[RunData],
    figures_dir: Path,
    *,
    value_field: str,
    filename_prefix: str,
    title_prefix: str,
) -> None:
    if not rows:
        return
    labels = [
        label
        for label in run_label_order(runs)
        if any(row["analysis_label"] == label for row in rows)
    ]
    scenarios = sorted({row["scenario"] for row in rows})
    populations = sorted({int(row["population"]) for row in rows})

    for scenario in scenarios:
        for population in populations:
            available = [
                label
                for label in labels
                if any(
                    row["analysis_label"] == label
                    and row["scenario"] == scenario
                    and int(row["population"]) == population
                    for row in rows
                )
            ]
            if not available:
                continue
            columns = min(4, len(available))
            rows_count = math.ceil(len(available) / columns)
            figure, axes = plt.subplots(
                rows_count,
                columns,
                figsize=(4.0 * columns, 3.8 * rows_count),
                squeeze=False,
                layout="constrained",
            )
            image = None
            for index, label in enumerate(available):
                axis = axes.ravel()[index]
                matrix = heatmap_matrix(
                    rows,
                    label=label,
                    scenario=scenario,
                    population=population,
                    value_field=value_field,
                )
                image = axis.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto")
                axis.set_xticks(range(3))
                axis.set_xticklabels(
                    ("mostly high", "mixed", "mostly low"), rotation=30, ha="right"
                )
                axis.set_yticks(range(3))
                axis.set_yticklabels(ECOLOGICAL_STATES)
                axis.set_title(label)
                for eco in range(3):
                    for social in range(3):
                        value = matrix[eco, social]
                        if np.isfinite(value):
                            axis.text(social, eco, f"{value:.2f}", ha="center", va="center")
            for axis in axes.ravel()[len(available) :]:
                axis.axis("off")
            figure.suptitle(f"{title_prefix}: {scenario}, N={population}")
            if image is not None:
                figure.colorbar(image, ax=list(axes.ravel()), shrink=0.75)
            path = figures_dir / (f"{filename_prefix}_{safe_filename(scenario)}_N{population}.png")
            figure.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(figure)
            print(f"Saved {path}")


def save_trajectory_figure(
    summary: list[dict],
    runs: list[RunData],
    figures_dir: Path,
    *,
    metric: str,
    filename: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    rows = [row for row in summary if row["metric"] == metric]
    if not rows:
        return
    scenarios = sorted({row["scenario"] for row in rows})
    populations = sorted({int(row["population"]) for row in rows})
    labels = run_label_order(runs)
    figure, axes = scenario_population_grid(scenarios, populations)
    handles, handle_labels = [], []

    for i, scenario in enumerate(scenarios):
        for j, population in enumerate(populations):
            axis = axes[i, j]
            for label in labels:
                subset = sorted(
                    [
                        row
                        for row in rows
                        if row["analysis_label"] == label
                        and row["scenario"] == scenario
                        and int(row["population"]) == population
                    ],
                    key=lambda row: int(row["time"]),
                )
                if not subset:
                    continue
                x = np.asarray([int(row["time"]) for row in subset], dtype=float)
                y = np.asarray([float(row["mean"]) for row in subset], dtype=float)
                low = np.asarray([float(row["bootstrap_ci_low"]) for row in subset], dtype=float)
                high = np.asarray([float(row["bootstrap_ci_high"]) for row in subset], dtype=float)
                line = axis.plot(x, y, label=label)[0]
                axis.fill_between(x, low, high, alpha=0.15)
                if i == 0 and j == 0 and label not in handle_labels:
                    handles.append(line)
                    handle_labels.append(label)
            axis.set_title(f"{scenario} | N={population}")
            axis.set_xlabel("Training step")
            axis.grid(alpha=0.25)
            if j == 0:
                axis.set_ylabel(ylabel)
            if ylim is not None:
                axis.set_ylim(*ylim)

    if handles:
        figure.legend(
            handles,
            handle_labels,
            loc="lower center",
            ncol=min(4, len(handle_labels)),
            bbox_to_anchor=(0.5, -0.02),
        )
    path = figures_dir / filename
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {path}")


def save_scatter_grid(
    rows: list[dict],
    runs: list[RunData],
    figures_dir: Path,
    *,
    x_field: str,
    y_field: str,
    filename: str,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    finite_rows = [
        row
        for row in rows
        if np.isfinite(as_float(row, x_field)) and np.isfinite(as_float(row, y_field))
    ]
    if not finite_rows:
        return
    scenarios = sorted({row["scenario"] for row in finite_rows})
    populations = sorted({int(row["population"]) for row in finite_rows})
    labels = run_label_order(runs)
    figure, axes = scenario_population_grid(scenarios, populations)
    handles, handle_labels = [], []

    for i, scenario in enumerate(scenarios):
        for j, population in enumerate(populations):
            axis = axes[i, j]
            for label in labels:
                subset = [
                    row
                    for row in finite_rows
                    if row["scenario"] == scenario
                    and int(row["population"]) == population
                    and row["analysis_label"] == label
                ]
                if not subset:
                    continue
                artist = axis.scatter(
                    [as_float(row, x_field) for row in subset],
                    [as_float(row, y_field) for row in subset],
                    s=22,
                    alpha=0.7,
                    label=label,
                )
                if i == 0 and j == 0 and label not in handle_labels:
                    handles.append(artist)
                    handle_labels.append(label)
            axis.set_title(f"{scenario} | N={population}")
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            if xlim is not None:
                axis.set_xlim(*xlim)
            if ylim is not None:
                axis.set_ylim(*ylim)

    if handles:
        figure.legend(
            handles,
            handle_labels,
            loc="lower center",
            ncol=min(4, len(handle_labels)),
            bbox_to_anchor=(0.5, -0.02),
        )
    path = figures_dir / filename
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {path}")


def save_effect_figure(
    rows: list[dict],
    figures_dir: Path,
    *,
    metric: str,
    effect_field: str,
    low_field: str,
    high_field: str,
    label_field: str,
    filename: str,
    ylabel: str,
) -> None:
    subset = [
        row
        for row in rows
        if row.get("metric") == metric and np.isfinite(as_float(row, effect_field))
    ]
    if not subset:
        return
    scenarios = sorted({row["scenario"] for row in subset})
    populations = sorted({int(row["population"]) for row in subset})
    figure, axes = scenario_population_grid(scenarios, populations)

    for i, scenario in enumerate(scenarios):
        for j, population in enumerate(populations):
            axis = axes[i, j]
            panel = sorted(
                [
                    row
                    for row in subset
                    if row["scenario"] == scenario and int(row["population"]) == population
                ],
                key=lambda row: (as_float(row, "rewire_theta"), str(row[label_field])),
            )
            x = np.arange(len(panel), dtype=float)
            y = np.asarray([as_float(row, effect_field) for row in panel], dtype=float)
            low = np.asarray([as_float(row, low_field) for row in panel], dtype=float)
            high = np.asarray([as_float(row, high_field) for row in panel], dtype=float)
            if panel:
                lower_error = np.maximum(0.0, y - low)
                upper_error = np.maximum(0.0, high - y)
                axis.errorbar(x, y, yerr=np.vstack([lower_error, upper_error]), fmt="o", capsize=3)
            axis.axhline(0.0, linewidth=1.0)
            axis.set_xticks(x)
            axis.set_xticklabels([str(row[label_field]) for row in panel], rotation=35, ha="right")
            axis.set_title(f"{scenario} | N={population}")
            axis.grid(alpha=0.25, axis="y")
            if j == 0:
                axis.set_ylabel(ylabel)

    path = figures_dir / filename
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {path}")


def save_phase_heatmaps(
    rows: list[dict], figures_dir: Path, *, metric: str, filename_prefix: str, title: str
) -> None:
    subset = [row for row in rows if row["metric"] == metric]
    if not subset:
        return
    scenarios = sorted({row["scenario"] for row in subset})
    populations = sorted({int(row["population"]) for row in subset})

    for scenario in scenarios:
        for population in populations:
            panel = [
                row
                for row in subset
                if row["scenario"] == scenario and int(row["population"]) == population
            ]
            if not panel:
                continue
            thetas = sorted({float(row["rewire_theta"]) for row in panel})
            mus = sorted({float(row["rewire_mu"]) for row in panel})
            matrix = np.full((len(mus), len(thetas)), np.nan, dtype=float)
            for row in panel:
                matrix[
                    mus.index(float(row["rewire_mu"])), thetas.index(float(row["rewire_theta"]))
                ] = float(row["mean"])

            figure, axis = plt.subplots(
                figsize=(max(5.0, 1.2 * len(thetas)), max(4.0, 1.0 * len(mus))),
                layout="constrained",
            )
            image = axis.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto")
            axis.set_xticks(range(len(thetas)))
            axis.set_xticklabels([format_parameter(value) for value in thetas])
            axis.set_yticks(range(len(mus)))
            axis.set_yticklabels([format_parameter(value) for value in mus])
            axis.set_xlabel("theta (global-search probability)")
            axis.set_ylabel("mu (eligible-rewire probability)")
            axis.set_title(f"{title}: {scenario}, N={population}")
            for mu_i in range(len(mus)):
                for theta_i in range(len(thetas)):
                    value = matrix[mu_i, theta_i]
                    if np.isfinite(value):
                        axis.text(theta_i, mu_i, f"{value:.3f}", ha="center", va="center")
            figure.colorbar(image, ax=axis)
            path = figures_dir / (f"{filename_prefix}_{safe_filename(scenario)}_N{population}.png")
            figure.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(figure)
            print(f"Saved {path}")


def save_local_population_scatter(rows: list[dict], runs: list[RunData], figures_dir: Path) -> None:
    finite = [
        row
        for row in rows
        if np.isfinite(as_float(row, "mean_population_low_fraction_excluding"))
        and np.isfinite(as_float(row, "mean_observed_low_fraction"))
    ]
    if not finite:
        return
    scenarios = sorted({row["scenario"] for row in finite})
    populations = sorted({as_int(row, "population") for row in finite})
    labels = run_label_order(runs)
    figure, axes = scenario_population_grid(scenarios, populations)
    handles, handle_labels = [], []

    for i, scenario in enumerate(scenarios):
        for j, population in enumerate(populations):
            axis = axes[i, j]
            axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0)
            for label in labels:
                panel = [
                    row
                    for row in finite
                    if row["scenario"] == scenario
                    and as_int(row, "population") == population
                    and row["analysis_label"] == label
                ]
                if not panel:
                    continue
                artist = axis.scatter(
                    [as_float(row, "mean_population_low_fraction_excluding") for row in panel],
                    [as_float(row, "mean_observed_low_fraction") for row in panel],
                    s=18,
                    alpha=0.55,
                    label=label,
                )
                if i == 0 and j == 0 and label not in handle_labels:
                    handles.append(artist)
                    handle_labels.append(label)
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_title(f"{scenario} | N={population}")
            axis.set_xlabel("Population LOW fraction excluding focal")
            axis.set_ylabel("Locally observed LOW fraction")
            axis.grid(alpha=0.25)

    if handles:
        figure.legend(
            handles,
            handle_labels,
            loc="lower center",
            ncol=min(4, len(handle_labels)),
            bbox_to_anchor=(0.5, -0.02),
        )
    path = figures_dir / "12_local_vs_population_low_fraction.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {path}")


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def analysis_manifest(*, args, runs: list[RunData], warnings: list[str]) -> dict:
    return {
        "schema_version": 1,
        "profile": getattr(args, "profile", "diagnostics"),
        "reward_definition": reward_identity(runs[0].config),
        "primary_training_window": "max(0, T-1000) < recorded step <= T",
        "inference": "replicate-level pointwise percentile intervals; no significance claims",
        "analysis_name": args.analysis_name,
        "command": " ".join(sys.argv),
        "python_version": platform.python_version(),
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed,
        "resource_low_threshold": args.resource_low_threshold,
        "resource_high_threshold": args.resource_high_threshold,
        "primary_evaluation_mode": PRIMARY_EVALUATION_MODE,
        "primary_strategy": PRIMARY_STRATEGY,
        "warnings": warnings,
        "inputs": run_catalog_rows(runs),
    }


def run_analysis(args) -> Path:
    if not (0.0 <= args.resource_low_threshold < args.resource_high_threshold <= 1.0):
        raise ValueError("Resource thresholds must satisfy 0 <= low < high <= 1.")
    if args.bootstrap_reps < 0:
        raise ValueError("--bootstrap-reps must be non-negative.")

    resolved_paths = [Path(value).expanduser().resolve() for value in args.run]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("The same run directory was supplied more than once.")

    runs = [load_run(path) for path in resolved_paths]
    assign_run_labels(runs)
    warnings = validate_compatibility(runs)
    warnings.extend(resolve_r0_pairs(runs))

    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (ANALYSIS_ROOT / args.analysis_name).resolve()
    )
    tables_dir = output_dir / "data"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    from .utility_analysis import build_utility_summaries

    utility_rows, utility_summary = build_utility_summaries(
        runs, bootstrap_reps=args.bootstrap_reps, bootstrap_seed=args.bootstrap_seed
    )
    write_csv_rows(tables_dir / "utility_evaluation.csv", utility_rows)
    write_csv_rows(tables_dir / "utility_evaluation_summary.csv", utility_summary)

    if getattr(args, "profile", "diagnostics") == "focused":
        from .focused_analysis import run_focused_analysis

        run_focused_analysis(
            runs,
            tables_dir,
            figures_dir,
            bootstrap_reps=args.bootstrap_reps,
            bootstrap_seed=args.bootstrap_seed,
        )
        write_csv_rows(tables_dir / "run_catalog.csv", run_catalog_rows(runs))
        manifest = analysis_manifest(args=args, runs=runs, warnings=warnings)
        (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"Focused analysis complete: {output_dir}")
        for warning in warnings:
            print(f"Warning: {warning}")
        return output_dir

    resource_rows, resource_summary = build_resource_distribution(
        runs,
        low_threshold=args.resource_low_threshold,
        high_threshold=args.resource_high_threshold,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
    )
    policy_rows = build_policy_heatmap_summary(
        runs, bootstrap_reps=args.bootstrap_reps, bootstrap_seed=args.bootstrap_seed
    )
    trajectory_rows = build_trajectory_summary(
        runs, bootstrap_reps=args.bootstrap_reps, bootstrap_seed=args.bootstrap_seed
    )
    terminal_rows = build_terminal_training_rows(runs, resource_rows)
    paired_rows, paired_summary = build_paired_effects(
        runs, bootstrap_reps=args.bootstrap_reps, bootstrap_seed=args.bootstrap_seed
    )
    memory_rows, memory_summary, memory_warnings = build_network_memory_effects(
        runs, bootstrap_reps=args.bootstrap_reps, bootstrap_seed=args.bootstrap_seed
    )
    warnings.extend(memory_warnings)
    phase_rows = build_phase_summary(
        runs,
        resource_rows,
        terminal_rows,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
    )
    agent_rows = build_agent_scatter_rows(runs)

    write_csv_rows(tables_dir / "run_catalog.csv", run_catalog_rows(runs))
    write_csv_rows(tables_dir / "resource_distribution.csv", resource_rows)
    write_csv_rows(tables_dir / "resource_distribution_summary.csv", resource_summary)
    write_csv_rows(tables_dir / "policy_heatmap_summary.csv", policy_rows)
    write_csv_rows(tables_dir / "trajectory_summary.csv", trajectory_rows)
    write_csv_rows(tables_dir / "terminal_training_summary.csv", terminal_rows)
    write_csv_rows(tables_dir / "paired_adaptive_minus_r0.csv", paired_rows)
    write_csv_rows(tables_dir / "paired_adaptive_minus_r0_summary.csv", paired_summary)
    write_csv_rows(tables_dir / "network_memory_effects.csv", memory_rows)
    write_csv_rows(tables_dir / "network_memory_effects_summary.csv", memory_summary)
    write_csv_rows(tables_dir / "theta_mu_summary.csv", phase_rows)

    save_resource_distribution_figure(
        resource_rows, runs, figures_dir, bootstrap_seed=args.bootstrap_seed
    )
    save_policy_heatmaps(
        policy_rows,
        runs,
        figures_dir,
        value_field="policy_low_mean",
        filename_prefix="02_joint_state_behavior_heatmap",
        title_prefix="P(LOW | ecological, social state)",
    )
    save_policy_heatmaps(
        policy_rows,
        runs,
        figures_dir,
        value_field="occupancy_mean",
        filename_prefix="03_joint_state_occupancy_heatmap",
        title_prefix="Training state occupancy P(ecological, social state)",
    )
    save_trajectory_figure(
        trajectory_rows,
        runs,
        figures_dir,
        metric="visibility_gini",
        filename="04_visibility_gini_trajectories.png",
        ylabel="Visibility Gini",
        ylim=(0.0, 1.0),
    )
    save_trajectory_figure(
        trajectory_rows,
        runs,
        figures_dir,
        metric="mean_perception_error",
        filename="05_perception_error_trajectories.png",
        ylabel="Mean perception error",
        ylim=(0.0, 1.0),
    )
    save_trajectory_figure(
        trajectory_rows,
        runs,
        figures_dir,
        metric="degree_action_correlation",
        filename="06_degree_action_correlation_trajectories.png",
        ylabel="Degree-action correlation",
        ylim=(-1.0, 1.0),
    )
    save_scatter_grid(
        terminal_rows,
        runs,
        figures_dir,
        x_field="training_visibility_gini",
        y_field="training_wealth_gini",
        filename="07_wealth_gini_vs_visibility_gini.png",
        xlabel="Terminal visibility Gini",
        ylabel="Terminal training wealth Gini",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    save_scatter_grid(
        terminal_rows,
        runs,
        figures_dir,
        x_field="mean_edge_turnover",
        y_field="eval_mean_resource_fraction",
        filename="08_turnover_vs_resource_outcome.png",
        xlabel="Mean recorded edge turnover",
        ylabel="Fresh-ecology mean resource fraction",
        ylim=(0.0, 1.0),
    )
    save_effect_figure(
        paired_summary,
        figures_dir,
        metric="resource_fraction",
        effect_field="mean_adaptive_minus_r0",
        low_field="bootstrap_ci_low",
        high_field="bootstrap_ci_high",
        label_field="adaptive_label",
        filename="09_paired_adaptive_minus_r0_resource_effects.png",
        ylabel="Adaptive - matched R0 resource fraction",
    )
    save_effect_figure(
        memory_summary,
        figures_dir,
        metric="resource_fraction",
        effect_field="mean_carried_minus_reset",
        low_field="bootstrap_ci_low",
        high_field="bootstrap_ci_high",
        label_field="analysis_label",
        filename="10_network_memory_resource_effects.png",
        ylabel="Carried terminal - reset initial resource fraction",
    )
    save_phase_heatmaps(
        phase_rows,
        figures_dir,
        metric="resource_fraction",
        filename_prefix="11_theta_mu_resource_heatmap",
        title="Prediction-error rewiring resource outcome",
    )
    save_phase_heatmaps(
        phase_rows,
        figures_dir,
        metric="visibility_gini",
        filename_prefix="11_theta_mu_visibility_gini_heatmap",
        title="Prediction-error rewiring terminal visibility Gini",
    )
    save_local_population_scatter(agent_rows, runs, figures_dir)

    manifest = analysis_manifest(args=args, runs=runs, warnings=warnings)
    with (output_dir / "analysis_manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)

    if warnings:
        print("\nAnalysis warnings:")
        for warning in warnings:
            print(f"- {warning}")
    print(f"\nCross-treatment analysis complete: {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-treatment analysis and figures for cognitive_tools "
            "social-network experiment outputs."
        )
    )
    parser.add_argument(
        "--profile",
        choices=("focused", "diagnostics"),
        default="focused",
        help="Four study figures by default; legacy diagnostics are opt-in.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Experiment run directory. Repeat once per treatment/parameter setting.",
    )
    parser.add_argument(
        "--analysis-name",
        default="social_core_analysis",
        help=("Output name under results/q_learning_baseline/social_analysis/."),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output directory; overrides --analysis-name.",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=2000,
        help="Bootstrap resamples for run-level means and paired effects.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=1729,
        help="Deterministic bootstrap/display-jitter seed.",
    )
    parser.add_argument(
        "--resource-low-threshold",
        type=float,
        default=1.0 / 3.0,
        help="Final resource fraction below which a run is classified low-resource.",
    )
    parser.add_argument(
        "--resource-high-threshold",
        type=float,
        default=2.0 / 3.0,
        help="Final resource fraction at/above which a run is classified high-resource.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
