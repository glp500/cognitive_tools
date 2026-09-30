"""Check matched carrying capacity and record no-harvest ecological baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from cognitive_tools.ecology import depletion_from_equilibrium, resource_step
from cognitive_tools.scenarios import BALANCED_SCENARIOS, build_environment_maps


def _mixed_edges(capacity: np.ndarray) -> int:
    return int(np.count_nonzero(capacity[:, 1:] != capacity[:, :-1])) + int(
        np.count_nonzero(capacity[1:, :] != capacity[:-1, :])
    )


def validate(seed: int, replicates: int, steps: int, width: int = 10, height: int = 10):
    if replicates < 1 or steps < 1:
        raise ValueError("replicates and steps must be positive")
    rows = []
    for replicate in range(replicates):
        maps = {
            scenario: build_environment_maps(
                scenario, width=width, height=height, seed=seed + replicate
            )
            for scenario in BALANCED_SCENARIOS
        }
        capacities = [maps[name][0] for name in BALANCED_SCENARIOS]
        if not np.allclose([capacity.sum() for capacity in capacities], 0.75 * width * height):
            raise ValueError(f"Capacity totals differ in replicate {replicate}")
        if not np.array_equal(np.sort(capacities[1].ravel()), np.sort(capacities[2].ravel())):
            raise ValueError(f"Dispersed and segregated capacity histograms differ: {replicate}")
        for capacity, recovery, equilibrium, _ in maps.values():
            if not np.allclose(recovery, 0.05) or not np.allclose(equilibrium, 0.70):
                raise ValueError("Renewal maps differ")
        edges = {name: _mixed_edges(maps[name][0]) for name in BALANCED_SCENARIOS}
        if edges["balanced_dispersed"] <= edges["balanced_segregated"]:
            raise ValueError(f"Dispersed map is not more spatially mixed: {replicate}")
        for scenario, (capacity, recovery, equilibrium, _) in maps.items():
            resource = 0.5 * capacity
            depletion = depletion_from_equilibrium(recovery, equilibrium)
            for time in range(1, steps + 1):
                resource = resource_step(resource, capacity, recovery, depletion, 0.10)
                if time in (100, steps):
                    rows.append(
                        dict(
                            scenario=scenario,
                            replicate=replicate,
                            seed=seed + replicate,
                            time=time,
                            total_capacity=float(capacity.sum()),
                            initial_resource=float(0.5 * capacity.sum()),
                            total_resource=float(resource.sum()),
                            capacity_weighted_resource_fraction=float(
                                resource.sum() / capacity.sum()
                            ),
                            mean_local_resource_fraction=float(np.mean(resource / capacity)),
                            mixed_neighbor_edges=edges[scenario],
                            capacity_sha256=hashlib.sha256(capacity.tobytes()).hexdigest(),
                        )
                    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = validate(args.seed, args.replicates, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "event": "landscape_validation_complete",
                "output": str(args.output),
                "replicates": args.replicates,
                "rows": len(rows),
                "capacity_total": rows[0]["total_capacity"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
