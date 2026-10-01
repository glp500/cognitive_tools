"""Validate the exact held-out population-payoff gate for the balanced study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

SCENARIOS = ("balanced_uniform", "balanced_dispersed", "balanced_segregated")
RETURNS = ("sum", "discounted")
RAW_FILES = ("agent_returns.csv", "episode_summary.csv", "paired_returns.csv", "assignments.csv")
EXPECTED_CONFIG = {
    "scenarios": list(SCENARIOS),
    "population": 64,
    "replicates": 100,
    "replicate_start": 4000,
    "seed": 20261014,
    "focal_count": 4,
    "assignments": 1,
    "gamma": 0.95,
    "policy_c": "000",
    "policy_d": "111",
    "compositions": [0, 63],
    "horizons": [1000],
    "reward_mode": "capped_harvest",
    "width": 10,
    "height": 10,
    "coupling": 0.1,
    "low_harvest": 0.002,
    "high_harvest": 0.02,
    "initial_resource_fraction": 0.5,
    "metabolism": 0.002,
    "initial_energy": 1.0,
    "energy_capacity": 1.0,
    "late_window": 200,
}


def validate_gate(analysis: Path) -> dict:
    producer_path = analysis.parent / "run" / "manifest.json"
    producer_bytes = producer_path.read_bytes()
    producer = json.loads(producer_bytes)
    metadata = json.loads((analysis / "analysis_manifest.json").read_text())
    if producer.get("status") != "complete" or producer.get("purpose") != "validation":
        raise ValueError("Payoff run is not a complete validation run")
    if metadata.get("input_manifest_sha256") != hashlib.sha256(producer_bytes).hexdigest():
        raise ValueError("Payoff analysis does not match the validation run manifest")
    if (
        metadata.get("resamples") != 5000
        or metadata.get("seed") != 1729
        or metadata.get("interval_method") != "centered max-standardized replicate bootstrap"
    ):
        raise ValueError("Payoff analysis interval protocol differs from the frozen gate")
    output_hashes = producer.get("output_sha256", {})
    if set(output_hashes) != set(RAW_FILES) or metadata.get("input_output_sha256") != output_hashes:
        raise ValueError("Payoff analysis raw-data hashes do not match the validation run")
    for name, expected_digest in output_hashes.items():
        if (
            hashlib.sha256((analysis.parent / "run" / name).read_bytes()).hexdigest()
            != expected_digest
        ):
            raise ValueError(f"Payoff raw data changed after validation: {name}")
    config = producer.get("config", {})
    mismatched = [name for name, value in EXPECTED_CONFIG.items() if config.get(name) != value]
    if mismatched:
        raise ValueError(f"Payoff gate scientific configuration differs: {', '.join(mismatched)}")
    with (analysis / "verdicts.csv").open(newline="") as handle:
        verdicts = list(csv.DictReader(handle))
    expected = {(scenario, ret) for scenario in SCENARIOS for ret in RETURNS}
    observed = {(row["scenario"], row["return_type"]) for row in verdicts}
    if len(verdicts) != len(expected) or observed != expected:
        raise ValueError("Payoff gate must have exactly six unique scenario/return verdicts")
    unsupported = [
        f"{row['scenario']}:{row['return_type']}:{row['verdict']}"
        for row in verdicts
        if row["horizon"] != "1000" or row["verdict"] != "supported"
    ]
    if unsupported:
        raise ValueError(f"Payoff gate has unsupported cells: {', '.join(unsupported)}")
    return {
        "gate": str(analysis),
        "status": "supported",
        "scenario_return_cells": len(verdicts),
        "replicates_per_scenario": config["replicates"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_gate(args.analysis), sort_keys=True))


if __name__ == "__main__":
    main()
