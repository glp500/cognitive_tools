"""Factor-based Stage-5 summaries and paired contrasts; replicate is the inferential unit."""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

from .analysis import bootstrap_mean_ci, read_csv_rows, stable_seed, write_csv_rows
from .scenarios import BALANCED_SCENARIOS
from .social import _gini_nonnegative
from .visibility import PROFILES, PROTOCOL, graph_hash

PRIMARY = (
    "social_perception_error",
    "majority_mismatch_rate",
    "majority_tie_rate",
    "visibility_gini",
    "low_extraction_rate",
)
SECONDARY = (
    ("resource_fraction", "eval_mean_mean_resource_fraction"),
    ("reserve_welfare", "eval_mean_mean_reserve_welfare"),
    ("final_wealth_gini", "final_wealth_gini"),
    ("evaluation_utility_sum", "evaluation_mean_utility_sum"),
)
ECOLOGIES = ("uniform_high", "patchy_high", "split_high_low")


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _summary(rows, key_fields, *, reps, seed):
    groups = defaultdict(list)
    for row in rows:
        value = _number(row["value"])
        if math.isfinite(value):
            groups[tuple(row[field] for field in key_fields)].append(value)
    output = []
    for key, values in sorted(groups.items()):
        mean, low, high = bootstrap_mean_ci(
            values, bootstrap_reps=reps, seed=stable_seed(seed, *key)
        )
        output.append(
            dict(
                zip(key_fields, key),
                n_replicates=len(values),
                mean=mean,
                low=low,
                high=high,
                interval="pointwise replicate-bootstrap 95%",
            )
        )
    return output


def _contrast(rows, label, left, right, dimensions, metrics, *, reps, seed):
    indexed = {
        (tuple(row[field] for field in dimensions), row["metric"], int(row["replicate"])): _number(
            row["value"]
        )
        for row in rows
        if row["metric"] in metrics
    }
    result = []
    for row in rows:
        if row["metric"] not in metrics or not left(row):
            continue
        other = right(row)
        other_key = (
            tuple(other[field] for field in dimensions),
            row["metric"],
            int(row["replicate"]),
        )
        difference = _number(row["value"]) - indexed.get(other_key, float("nan"))
        if math.isfinite(difference):
            result.append(
                dict(
                    comparison=label(row),
                    scenario=row["scenario"],
                    profile=row["profile"],
                    dynamics=row["dynamics"],
                    population=row["population"],
                    replicate=row["replicate"],
                    metric=row["metric"],
                    value=difference,
                )
            )
    summary = _summary(
        result,
        ("comparison", "scenario", "profile", "dynamics", "population", "metric"),
        reps=reps,
        seed=seed,
    )
    return result, summary


def run_visibility_analysis(runs, tables_dir, figures_dir, *, bootstrap_reps, bootstrap_seed):
    if not runs or any(run.config.get("study_protocol") != PROTOCOL for run in runs):
        raise ValueError("Visibility analysis accepts only Stage-5 runs")
    balanced = runs[0].config.get("environment_design") == "balanced_capacity_v1"
    ecologies = BALANCED_SCENARIOS if balanced else ECOLOGIES
    secondary = SECONDARY
    if balanced:
        if any(tuple(run.config["scenarios"]) != BALANCED_SCENARIOS for run in runs):
            raise ValueError("Balanced analysis requires all three matched landscapes")
        secondary = (
            ("resource_fraction", "eval_mean_capacity_weighted_resource_fraction"),
            ("local_resource_fraction", "eval_mean_mean_resource_fraction"),
            ("resource_stock", "eval_mean_total_resource"),
            ("total_capacity", "eval_mean_total_capacity"),
            *SECONDARY[1:],
        )
    run_factors = {
        (run.config.get("visibility_profile"), run.config.get("network_dynamics")) for run in runs
    }
    if len(run_factors) != len(runs):
        raise ValueError("Duplicate Stage-5 treatment factors")
    initial_agents, initial_reps, primary, outcomes = [], [], [], []
    trajectory_groups = defaultdict(list)
    graph_identity = {}
    for run in runs:
        loaded_here = not run.tables
        if loaded_here:
            data_dir = run.path / "data"
            run.tables = {
                name: read_csv_rows(data_dir / f"{name}.csv", required=True)
                for name in ("evaluation_summary", "training_timeseries")
            }
            social_run = run.config.get("social_mode") == "fixed"
            for name in ("initial_visibility", "network_timeseries", "network_edges_checkpoints"):
                run.tables[name] = read_csv_rows(data_dir / f"{name}.csv", required=social_run)
        config = run.config
        profile, dynamics = config.get("visibility_profile"), config.get("network_dynamics")
        social = config.get("social_mode") == "fixed"
        if social != (profile in PROFILES and dynamics in ("fixed", "adaptive_bounded")):
            raise ValueError(f"Invalid Stage-5 factors in {run.run_id}")
        expected = {
            (scenario, int(pop), rep)
            for scenario in config["scenarios"]
            for pop in config["populations"]
            for rep in range(int(config["replicates"]))
        }
        initial_by_condition = defaultdict(list)
        for row in run.tables["initial_visibility"]:
            key = (row["scenario"], int(row["population"]), int(row["replicate"]))
            initial_by_condition[key].append(row)
        if social and set(initial_by_condition) != expected:
            raise ValueError(f"Missing Stage-5 initial visibility conditions in {run.run_id}")
        snapshots = defaultdict(list)
        for row in run.tables["network_edges_checkpoints"]:
            if row["checkpoint"] == "initial":
                snapshots[(row["scenario"], int(row["population"]), int(row["replicate"]))].append(
                    row
                )
        for key, agents in initial_by_condition.items():
            n = key[1]
            if len(agents) != n or len({r["agent"] for r in agents}) != n:
                raise ValueError(f"Initial visibility agent coverage failed: {run.run_id} {key}")
            hashes = {(r["initial_network_sha256"], r["initial_propensity_sha256"]) for r in agents}
            if len(hashes) != 1:
                raise ValueError("Inconsistent initial network/propensity hashes")
            graph = {r["agent"]: [] for r in agents}
            for edge in snapshots[key]:
                graph[edge["observer"]].append(edge["source"])
            graph = {name: sorted(values) for name, values in graph.items()}
            # Source-list order has no scientific meaning, so compare edge sets via rows.
            counts = {r["agent"]: int(r["initial_visibility_count"]) for r in agents}
            if len(snapshots[key]) != n * int(config["attention_k"]) or any(
                len(values) != int(config["attention_k"])
                or len(set(values)) != len(values)
                or name in values
                for name, values in graph.items()
            ):
                raise ValueError("Initial graph violates fixed attention capacity")
            if any(
                sum(r["source"] == name for r in snapshots[key]) != count
                for name, count in counts.items()
            ):
                raise ValueError("Initial visibility counts disagree with graph")
            identity_key = (profile, n, key[2])
            pair = next(iter(hashes))
            if graph_hash(graph) != pair[0]:
                raise ValueError("Recorded initial graph hash disagrees with edge snapshot")
            if identity_key in graph_identity and graph_identity[identity_key] != pair:
                raise ValueError("Ecology or dynamics changed paired initial graph/propensity")
            graph_identity[identity_key] = pair
            if key[0] == config["scenarios"][0] and dynamics == "fixed":
                initial_agents.extend(agents)
                degrees = np.array(list(counts.values()), dtype=float)
                for metric, value in (
                    ("visibility_gini", _gini_nonnegative(degrees)),
                    ("zero_visibility_fraction", float(np.mean(degrees == 0))),
                    ("visibility_variance", float(np.var(degrees))),
                    ("maximum_visibility", float(np.max(degrees))),
                ):
                    initial_reps.append(
                        dict(
                            profile=profile,
                            population=n,
                            replicate=key[2],
                            metric=metric,
                            value=value,
                        )
                    )
        training = defaultdict(list)
        for row in run.tables["training_timeseries"]:
            key = (row["scenario"], int(row["population"]), int(row["replicate"]))
            if int(row["time"]) > int(config["training_steps"]) - 1000:
                training[key].append(row)
            if social:
                for metric in ("visibility_gini", "low_extraction_rate"):
                    trajectory_groups[
                        (
                            row["scenario"],
                            profile,
                            dynamics,
                            int(row["population"]),
                            int(row["time"]),
                            metric,
                        )
                    ].append(_number(row.get(metric)))
        if set(training) != expected:
            raise ValueError(f"Missing final training windows in {run.run_id}")
        fresh = {
            (r["scenario"], int(r["population"]), int(r["replicate"])): r
            for r in run.tables["evaluation_summary"]
            if r.get("strategy") == "q_learning" and r.get("evaluation_mode") == "fresh_reset"
        }
        if set(fresh) != expected:
            raise ValueError(f"Missing primary evaluation conditions in {run.run_id}")
        networks = defaultdict(dict)
        for row in run.tables["network_timeseries"]:
            key = (row["scenario"], int(row["population"]), int(row["replicate"]))
            networks[key][int(row["time"])] = row
        for key, windows in training.items():
            scenario, n, rep = key
            base = dict(
                scenario=scenario,
                population=n,
                replicate=rep,
                profile=profile or "none",
                dynamics=dynamics,
                run_id=run.run_id,
            )
            for metric in PRIMARY:
                if not social and metric != "low_extraction_rate":
                    continue
                values = [_number(row.get(metric)) for row in windows]
                finite = [v for v in values if math.isfinite(v)]
                value = float(np.mean(finite)) if finite else float("nan")
                primary.append(
                    dict(
                        **base,
                        metric=metric,
                        value=value,
                        recorded_windows=len(windows),
                        valid_windows=len(finite),
                    )
                )
            for metric, field in secondary:
                value = _number(fresh[key].get(field))
                outcomes.append(dict(**base, metric=metric, value=value))
            if social:
                first = networks[key].get(0)
                last = networks[key].get(int(config["training_steps"]))
                if first is None or last is None:
                    raise ValueError("Missing initial or terminal network snapshot")
                g0, gt = _number(first["visibility_gini"]), _number(last["visibility_gini"])
                for metric, value in (
                    ("initial_visibility_gini", g0),
                    ("terminal_visibility_gini", gt),
                    ("visibility_gini_change", gt - g0),
                    (
                        "terminal_zero_visibility_fraction",
                        _number(last["zero_visibility_fraction"]),
                    ),
                    ("terminal_network_turnover", _number(last["edge_turnover"])),
                    ("successful_rewires", _number(last["cumulative_rewires"])),
                ):
                    outcomes.append(dict(**base, metric=metric, value=value))
    initial_summary = _summary(
        initial_reps, ("profile", "population", "metric"), reps=bootstrap_reps, seed=bootstrap_seed
    )
    primary_summary = _summary(
        primary,
        ("scenario", "profile", "dynamics", "population", "metric"),
        reps=bootstrap_reps,
        seed=bootstrap_seed,
    )
    outcome_summary = _summary(
        outcomes,
        ("scenario", "profile", "dynamics", "population", "metric"),
        reps=bootstrap_reps,
        seed=bootstrap_seed,
    )
    all_rows = primary + outcomes
    if balanced:
        ecology, ecology_sum = [], []
        for left, right in (
            ("balanced_dispersed", "balanced_uniform"),
            ("balanced_segregated", "balanced_dispersed"),
            ("balanced_segregated", "balanced_uniform"),
        ):
            rows, summary = _contrast(
                all_rows,
                lambda r, left=left, right=right: f"{left}-{right}",
                lambda r, left=left: r["scenario"] == left,
                lambda r, right=right: {**r, "scenario": right},
                ("scenario", "profile", "dynamics", "population"),
                PRIMARY,
                reps=bootstrap_reps,
                seed=bootstrap_seed,
            )
            ecology.extend(rows)
            ecology_sum.extend(summary)
    else:
        ecology, ecology_sum = _contrast(
            all_rows,
            lambda r: f"{r['scenario']}-uniform_high",
            lambda r: r["scenario"] in ("patchy_high", "split_high_low"),
            lambda r: {**r, "scenario": "uniform_high"},
            ("scenario", "profile", "dynamics", "population"),
            PRIMARY,
            reps=bootstrap_reps,
            seed=bootstrap_seed,
        )
    visibility, visibility_sum = _contrast(
        all_rows,
        lambda r: "random-equal" if r["profile"] == "random" else f"{r['profile']}-random",
        lambda r: r["dynamics"] == "fixed" and r["profile"] in PROFILES and r["profile"] != "equal",
        lambda r: {**r, "profile": "equal" if r["profile"] == "random" else "random"},
        ("scenario", "profile", "dynamics", "population"),
        ("social_perception_error",),
        reps=bootstrap_reps,
        seed=bootstrap_seed,
    )
    adaptation, adaptation_sum = _contrast(
        all_rows,
        lambda r: "adaptive-fixed",
        lambda r: r["dynamics"] == "adaptive_bounded",
        lambda r: {**r, "dynamics": "fixed"},
        ("scenario", "profile", "dynamics", "population"),
        (
            "social_perception_error",
            "visibility_gini",
            "low_extraction_rate",
            "resource_fraction",
            "reserve_welfare",
            "final_wealth_gini",
        ),
        reps=bootstrap_reps,
        seed=bootstrap_seed,
    )
    trajectory_summary = []
    for key, values in sorted(trajectory_groups.items()):
        finite = [v for v in values if math.isfinite(v)]
        if finite:
            scenario, profile, dynamics, population, time, metric = key
            trajectory_summary.append(
                dict(
                    scenario=scenario,
                    profile=profile,
                    dynamics=dynamics,
                    population=population,
                    time=time,
                    metric=metric,
                    n_replicates=len(finite),
                    mean=float(np.mean(finite)),
                )
            )
    tables = dict(
        initial_visibility_replicates=initial_reps,
        initial_visibility_agents=initial_agents,
        initial_visibility_summary=initial_summary,
        primary_window_replicates=primary,
        primary_window_summary=primary_summary,
        outcome_replicates=outcomes,
        outcome_summary=outcome_summary,
        ecology_contrasts=ecology,
        ecology_contrast_summary=ecology_sum,
        visibility_profile_contrasts=visibility,
        visibility_profile_contrast_summary=visibility_sum,
        adaptive_fixed_replicates=adaptation,
        adaptive_fixed_summary=adaptation_sum,
        training_trajectory_summary=trajectory_summary,
    )
    for name, rows in tables.items():
        write_csv_rows(tables_dir / f"{name}.csv", rows)
    expected_factors = {(None, "none")} | {
        (profile, dynamics) for profile in PROFILES for dynamics in ("fixed", "adaptive_bounded")
    }
    health = dict(
        study_protocol=PROTOCOL,
        run_count=len(runs),
        observed_treatments=len(run_factors),
        missing_treatments=[
            f"{p or 'B0'}:{d}"
            for p, d in sorted(
                expected_factors - run_factors, key=lambda pair: (str(pair[0]), pair[1])
            )
        ],
        initial_graph_pairs_verified=len(graph_identity),
        primary_replicate_rows=len(primary),
        undefined_primary_windows=sum(not math.isfinite(_number(r["value"])) for r in primary),
        ecology_contrast_rows=len(ecology),
        visibility_contrast_rows=len(visibility),
        adaptive_contrast_rows=len(adaptation),
    )
    from .visibility_figures import save_visibility_figures

    save_visibility_figures(
        figures_dir,
        tables,
        ecologies=ecologies,
        width=int(runs[0].config.get("width", 10)),
        height=int(runs[0].config.get("height", 10)),
        seed=int(runs[0].config.get("seed", 20261002)) if balanced else 20261002,
    )
    health["figure_count"] = len(list(figures_dir.glob("0[1-5]_*.png")))
    if health["figure_count"] != 5:
        raise RuntimeError("Stage-5 analysis did not produce all five main figures")
    (tables_dir.parent / "analysis_health.json").write_text(json.dumps(health, indent=2))
    return tables
