"""Independent, paired population-payoff experiment (no training or reward changes)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .env import EcoEnv, reward_definition
from .qlearning import resource_state
from .scenarios import SCENARIOS, build_environment_maps


@dataclass(frozen=True)
class PayoffConfig:
    scenarios: tuple[str, ...] = ("uniform_high", "patchy_high", "split_high_low")
    population: int = 64
    width: int = 10
    height: int = 10
    replicates: int = 10
    focal_count: int = 2
    assignments: int = 1
    compositions: tuple[int, ...] = (0, 1, 16, 32, 48, 62, 63)
    horizons: tuple[int, ...] = (1000,)
    gamma: float = 0.95
    seed: int = 20260928
    replicate_start: int = 0
    policy_c: str = "000"
    policy_d: str = "111"
    coupling: float = 0.1
    low_harvest: float = 0.002
    high_harvest: float = 0.020
    initial_resource_fraction: float = 0.5
    metabolism: float = 0.002
    initial_energy: float = 1.0
    energy_capacity: float = 1.0
    late_window: int = 200
    reward_mode: str = "harvest"

    def validate(self):
        reward_definition(self.reward_mode, self.metabolism)
        for name in ("width", "height", "replicates", "focal_count", "assignments", "late_window"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.population < 2 or self.focal_count > self.population:
            raise ValueError("Require population >= 2 and focal_count <= population")
        if self.seed < 0 or self.replicate_start < 0:
            raise ValueError("Seeds and replicate_start must be nonnegative")
        if not self.scenarios or len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError("Scenarios must be nonempty and unique")
        if any(s not in SCENARIOS for s in self.scenarios):
            raise ValueError("Unknown scenario")
        for values, label in ((self.horizons, "horizons"), (self.compositions, "compositions")):
            if not values or tuple(sorted(set(values))) != tuple(values):
                raise ValueError(f"{label} must be sorted and unique")
            if any(not isinstance(v, int) for v in values):
                raise ValueError(f"{label} must contain integers")
        if min(self.horizons) < 1:
            raise ValueError("Horizons must be positive")
        if self.compositions[0] != 0 or self.compositions[-1] != self.population - 1:
            raise ValueError("Compositions must include 0 and N-1 and stay within these bounds")
        for policy in (self.policy_c, self.policy_d):
            if len(policy) != 3 or set(policy) - {"0", "1"}:
                raise ValueError("Policies must be three binary digits: scarce/moderate/abundant")
        if self.policy_c == self.policy_d:
            raise ValueError("Candidate policies must differ")
        for name in ("gamma", "coupling", "initial_resource_fraction"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be finite and in [0,1]")
        for name in (
            "low_harvest",
            "high_harvest",
            "metabolism",
            "initial_energy",
            "energy_capacity",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            self.high_harvest < self.low_harvest
            or not 0 <= self.initial_energy <= self.energy_capacity
        ):
            raise ValueError("Invalid extraction or energy settings")
        if self.energy_capacity == 0:
            raise ValueError("Energy capacity must be positive")


def make_env(config: PayoffConfig, scenario: str, replicate: int):
    capacity, recovery, equilibrium, regions = build_environment_maps(
        scenario, width=config.width, height=config.height, seed=config.seed + replicate
    )
    env = EcoEnv(
        reward_mode=config.reward_mode,
        width=config.width,
        height=config.height,
        n_agents=config.population,
        max_steps=max(config.horizons),
        capacity_map=capacity,
        regeneration_rate=recovery,
        equilibrium_fraction=equilibrium,
        coupling_rate=config.coupling,
        cooperative_harvest_amount=config.low_harvest,
        defective_harvest_amount=config.high_harvest,
        metabolism_rate=config.metabolism,
        initial_energy=config.initial_energy,
        energy_capacity=config.energy_capacity,
        initial_resource_fraction=config.initial_resource_fraction,
    )
    observations, _ = env.reset(seed=config.seed + 100_000 + replicate)
    return env, observations, regions


def assignment(population, focal, permutation, k, branch):
    if (
        branch not in ("C", "D")
        or not 0 <= focal < population
        or not 0 <= k < population
        or sorted(permutation) != [i for i in range(population) if i != focal]
    ):
        raise ValueError("Invalid focal/co-player assignment")
    labels = ["D"] * population
    for i in permutation[:k]:
        labels[i] = "C"
    labels[focal] = branch
    return tuple(labels)


def policy_actions(observations, labels, policy_c, policy_d):
    policies = {"C": policy_c, "D": policy_d}
    return {
        name: int(policies[labels[int(name.split("_")[1])]][resource_state(obs)])
        for name, obs in observations.items()
    }


def rollout(config, scenario, replicate, labels):
    """Accumulate every reward; snapshots at each horizon share one trajectory."""
    if len(labels) != config.population or set(labels) - {"C", "D"}:
        raise ValueError("One C/D policy label is required per agent")
    env, observations, regions = make_env(config, scenario, replicate)
    names = env.possible_agents
    agents = [env.model.by_name[name] for name in names]
    initial = np.array([a.wealth for a in agents])
    totals = np.zeros(config.population)
    discounted = totals.copy()
    gross_totals = totals.copy()
    gross_discounted = totals.copy()
    surplus_totals = totals.copy()
    surplus_discounted = totals.copy()
    low_counts = totals.copy()
    recent = np.zeros((config.late_window, config.population))
    recent_utility = recent.copy()
    resource_total = reserve_total = 0.0
    rows, summaries = [], []
    for step in range(1, max(config.horizons) + 1):
        actions = policy_actions(observations, labels, config.policy_c, config.policy_d)
        observations, rewards, _, _, infos = env.step(actions)
        reward = np.array([rewards[name] for name in names])
        if not np.isfinite(reward).all():
            raise ValueError("Nonfinite reward")
        gross = np.array([infos[name]["harvested"] for name in names])
        surplus = np.array([infos[name]["uncredited_harvest"] for name in names])
        gross_totals += gross
        gross_discounted += config.gamma ** (step - 1) * gross
        surplus_totals += surplus
        surplus_discounted += config.gamma ** (step - 1) * surplus
        totals += reward
        discounted += config.gamma ** (step - 1) * reward
        low_counts += np.array([actions[name] == 0 for name in names])
        recent[(step - 1) % config.late_window] = gross
        recent_utility[(step - 1) % config.late_window] = reward
        resource = float(np.mean(env.model.resource / env.model.capacity))
        reserve = float(np.mean([a.reserve_welfare for a in agents]))
        resource_total += resource
        reserve_total += reserve
        if step not in config.horizons:
            continue
        wealth = np.array([a.wealth for a in agents]) - initial
        if not np.allclose(gross_totals, wealth, rtol=1e-10, atol=1e-12):
            raise AssertionError("Harvest sum does not equal wealth change")
        if not np.allclose(totals + surplus_totals, gross_totals, rtol=1e-10, atol=1e-12):
            raise AssertionError("Utility plus surplus does not equal harvest")
        late = recent.sum(axis=0) / min(step, config.late_window)
        late_utility = recent_utility.sum(axis=0) / min(step, config.late_window)
        for i, agent in enumerate(agents):
            x, y = map(int, agent.position)
            rows.append(
                dict(
                    horizon=step,
                    agent=i,
                    policy=labels[i],
                    x=x,
                    y=y,
                    region=str(regions[y, x]),
                    return_sum=float(totals[i]),
                    return_discounted=float(discounted[i]),
                    harvest_sum=float(gross_totals[i]),
                    harvest_discounted=float(gross_discounted[i]),
                    uncredited_harvest_sum=float(surplus_totals[i]),
                    uncredited_harvest_discounted=float(surplus_discounted[i]),
                    wealth_delta=float(wealth[i]),
                    late_utility_rate=float(late_utility[i]),
                    late_harvest_rate=float(late[i]),
                    low_fraction=float(low_counts[i] / step),
                )
            )
        summaries.append(
            dict(
                horizon=step,
                cooperators=labels.count("C"),
                mean_return_sum=float(totals.mean()),
                mean_return_discounted=float(discounted.mean()),
                mean_harvest_sum=float(gross_totals.mean()),
                mean_harvest_discounted=float(gross_discounted.mean()),
                late_utility_rate=float(late_utility.mean()),
                late_harvest_rate=float(late.mean()),
                final_resource_fraction=resource,
                mean_resource_fraction=resource_total / step,
                final_reserve_welfare=reserve,
                mean_reserve_welfare=reserve_total / step,
            )
        )
    env.close()
    return rows, summaries


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replicate_job(job):
    config, scenario, replicate, directory = job
    base = dict(scenario=scenario, replicate=replicate)
    agent_rows, episode_rows, paired_rows, assignments = [], [], [], []
    # Homogeneous endpoints are independent of focal choice and reused exactly.
    endpoint = {}
    for branch in ("C", "D"):
        rows, summaries = rollout(config, scenario, replicate, (branch,) * config.population)
        endpoint[branch] = rows, summaries
        identity = dict(
            **base,
            episode=f"all_{branch}",
            kind="endpoint",
            focal=-1,
            assignment=-1,
            k=-1,
            branch=branch,
        )
        agent_rows.extend({**identity, **r} for r in rows)
        episode_rows.extend({**identity, **r} for r in summaries)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, replicate, 71001]))
    focals = sorted(map(int, rng.choice(config.population, config.focal_count, replace=False)))
    for focal in focals:
        for draw in range(config.assignments):
            rng = np.random.default_rng(
                np.random.SeedSequence([config.seed, replicate, focal, draw, 71002])
            )
            permutation = list(
                map(int, rng.permutation([i for i in range(config.population) if i != focal]))
            )
            assignments.append(
                dict(**base, focal=focal, assignment=draw, coplayer_order=json.dumps(permutation))
            )
            for k in config.compositions:
                branches = {}
                for branch in ("C", "D"):
                    homogeneous = (branch == "C" and k == config.population - 1) or (
                        branch == "D" and k == 0
                    )
                    if homogeneous:
                        rows, _ = endpoint[branch]
                        episode = f"all_{branch}"
                    else:
                        episode = f"f{focal}_a{draw}_k{k}_{branch}"
                        labels = assignment(config.population, focal, permutation, k, branch)
                        rows, summaries = rollout(config, scenario, replicate, labels)
                        identity = dict(
                            **base,
                            episode=episode,
                            kind="mixed",
                            focal=focal,
                            assignment=draw,
                            k=k,
                            branch=branch,
                        )
                        agent_rows.extend({**identity, **r} for r in rows)
                        episode_rows.extend({**identity, **r} for r in summaries)
                    branches[branch] = (
                        {r["horizon"]: r for r in rows if r["agent"] == focal},
                        episode,
                    )
                for horizon in config.horizons:
                    c, d = branches["C"][0][horizon], branches["D"][0][horizon]
                    paired_rows.append(
                        dict(
                            **base,
                            focal=focal,
                            assignment=draw,
                            k=k,
                            horizon=horizon,
                            x=c["x"],
                            y=c["y"],
                            region=c["region"],
                            c_episode=branches["C"][1],
                            d_episode=branches["D"][1],
                            c_sum=c["return_sum"],
                            d_sum=d["return_sum"],
                            delta_sum=d["return_sum"] - c["return_sum"],
                            c_discounted=c["return_discounted"],
                            d_discounted=d["return_discounted"],
                            delta_discounted=d["return_discounted"] - c["return_discounted"],
                        )
                    )
    prefix = Path(directory) / f"{scenario}_{replicate}"
    for label, rows in (
        ("agents", agent_rows),
        ("episodes", episode_rows),
        ("pairs", paired_rows),
        ("assignments", assignments),
    ):
        write_csv(f"{prefix}_{label}.csv", rows)
    return scenario, replicate, len(episode_rows) // len(config.horizons)


def run_experiment(config, destination, *, workers=1, purpose="pilot"):
    config.validate()
    if workers < 1 or purpose not in ("pilot", "validation", "diagnostic"):
        raise ValueError("Invalid workers or purpose")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    shards = destination / "shards"
    shards.mkdir()
    source_root = Path(__file__).resolve().parent
    protocol = asdict(config)
    manifest = dict(
        schema="population_payoff_v2",
        reward_definition=reward_definition(config.reward_mode, config.metabolism),
        status="running",
        purpose=purpose,
        config=protocol,
        workers=workers,
        python=platform.python_version(),
        packages={name: version(name) for name in ("numpy", "mesa", "pettingzoo", "gymnasium")},
        source_sha256={
            str(p.relative_to(source_root.parent)): sha256(p)
            for p in sorted(source_root.glob("*.py"))
        },
        protocol_sha256=hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        decision_family="four endpoint contrasts across all configured scenarios, separately per horizon/return",
        seed_semantics="landscape=seed+replicate; position=seed+100000+replicate; focal and permutation use SeedSequence domains 71001 and 71002",
        tail_bounds={
            str(h): (
                min(config.high_harvest, config.metabolism)
                if config.reward_mode == "capped_harvest"
                else config.high_harvest
            )
            * config.gamma**h
            / (1 - config.gamma)
            if config.gamma < 1
            else None
            for h in config.horizons
        },
    )
    for label, command in (
        ("git_revision", ["git", "rev-parse", "HEAD"]),
        ("git_status", ["git", "status", "--short"]),
    ):
        result = subprocess.run(command, cwd=source_root.parent, capture_output=True, text=True)
        manifest[label] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    jobs = [
        (config, scenario, rep, str(shards))
        for scenario in config.scenarios
        for rep in range(config.replicate_start, config.replicate_start + config.replicates)
    ]
    episode_count = 0
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = executor.map(replicate_job, jobs) if executor else map(replicate_job, jobs)
        for scenario, replicate, count in results:
            episode_count += count
            print(f"Completed {scenario} replicate {replicate}: {count} episodes", flush=True)
        output_names = {
            "agents": "agent_returns.csv",
            "episodes": "episode_summary.csv",
            "pairs": "paired_returns.csv",
            "assignments": "assignments.csv",
        }
        for suffix, name in output_names.items():
            with (destination / name).open("w", newline="") as output:
                for index, (_, scenario, replicate, _) in enumerate(jobs):
                    with (shards / f"{scenario}_{replicate}_{suffix}.csv").open() as source:
                        if index:
                            next(source)
                        for line in source:
                            output.write(line)
        manifest.update(
            status="complete",
            elapsed_seconds=time.monotonic() - started,
            episode_count=episode_count,
            environment_steps=episode_count * max(config.horizons),
            output_sha256={name: sha256(destination / name) for name in output_names.values()},
        )
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="PayoffConfig JSON; CLI options override it")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--purpose", choices=("pilot", "validation", "diagnostic"), default="pilot")
    for name in (
        "population",
        "replicates",
        "focal_count",
        "assignments",
        "seed",
        "replicate_start",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("compositions", "horizons"):
        parser.add_argument("--" + name, nargs="+", type=int)
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS))
    parser.add_argument("--reward-mode", choices=("harvest", "capped_harvest"))
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--policy-c")
    parser.add_argument("--policy-d")
    args = vars(parser.parse_args())
    output, path, workers, purpose = (
        args.pop(k) for k in ("output", "config", "workers", "purpose")
    )
    settings = json.loads(path.read_text()) if path else {}
    settings.update({k: v for k, v in args.items() if v is not None})
    for key in ("scenarios", "compositions", "horizons"):
        if key in settings:
            settings[key] = tuple(settings[key])
    run_experiment(PayoffConfig(**settings), output, workers=workers, purpose=purpose)


if __name__ == "__main__":
    main()
