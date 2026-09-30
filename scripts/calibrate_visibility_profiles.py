"""Network-only calibration; no ecology or learned behavior is inspected."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cognitive_tools.social import social_metrics, visibility_counts
from cognitive_tools.visibility import (
    PROFILES,
    canonical_hash,
    graph_hash,
    init_visibility_attention,
    load_visibility_spec,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="configs/visibility/visibility_profiles_v1.json")
    parser.add_argument("--networks", type=int, default=1000)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--attention-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--output", default="results/visibility/calibration_v1")
    args = parser.parse_args()
    if args.networks < 1:
        parser.error("--networks must be positive")
    spec, sha = load_visibility_spec(args.spec)
    names = [f"agent_{i}" for i in range(args.population)]
    output = Path(args.output)
    if output.exists():
        parser.error("Calibration output exists; choose a new directory")
    output.mkdir(parents=True)
    rows, agents = [], []
    for profile in PROFILES:
        for replicate in range(args.networks):
            rng = np.random.default_rng(args.seed + replicate + 100000 * PROFILES.index(profile))
            init = init_visibility_attention(
                names, k=args.attention_k, profile=profile, rng=rng, spec=spec
            )
            counts = visibility_counts(init.sources)
            degrees = np.array(list(counts.values()), dtype=float)
            metrics = social_metrics(init.sources, None)
            rows.append(
                dict(
                    profile=profile,
                    replicate=replicate,
                    network_sha256=graph_hash(init.sources),
                    propensity_sha256=canonical_hash(init.raw_propensity),
                    visibility_gini=metrics["visibility_gini"],
                    zero_visibility_fraction=metrics["zero_visibility_fraction"],
                    max_visibility_count=int(degrees.max()),
                    visibility_variance=float(degrees.var()),
                    visibility_skewness=float(
                        np.mean(((degrees - degrees.mean()) / degrees.std()) ** 3)
                    )
                    if degrees.std()
                    else 0.0,
                    propensity_mean=float(np.mean(list(init.raw_propensity.values()))),
                )
            )
            if replicate == 0:
                agents.extend(
                    dict(
                        profile=profile,
                        agent=name,
                        raw_propensity=init.raw_propensity[name],
                        realized_visibility=count,
                    )
                    for name, count in counts.items()
                )
    for filename, data in [("network_metrics.csv", rows), ("first_network_agents.csv", agents)]:
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (output / "manifest.json").write_text(
        json.dumps(dict(spec_sha256=sha, spec=spec, parameters=vars(args)), indent=2)
    )
    print(f"Calibrated {args.networks} networks per profile: {output}")


if __name__ == "__main__":
    main()
