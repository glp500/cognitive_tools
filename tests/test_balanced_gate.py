import csv
import hashlib
import json

import pytest

from scripts.validate_balanced_gate import (
    EXPECTED_CONFIG,
    RAW_FILES,
    RETURNS,
    SCENARIOS,
    validate_gate,
)


def _gate(tmp_path):
    root = tmp_path / "gate"
    run = root / "run"
    analysis = root / "analysis"
    run.mkdir(parents=True)
    analysis.mkdir()
    raw_hashes = {}
    for name in RAW_FILES:
        data = f"fixture {name}".encode()
        (run / name).write_bytes(data)
        raw_hashes[name] = hashlib.sha256(data).hexdigest()
    manifest = {
        "status": "complete",
        "purpose": "validation",
        "config": EXPECTED_CONFIG,
        "output_sha256": raw_hashes,
    }
    producer_bytes = json.dumps(manifest).encode()
    (run / "manifest.json").write_bytes(producer_bytes)
    (analysis / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "input_manifest_sha256": hashlib.sha256(producer_bytes).hexdigest(),
                "input_output_sha256": raw_hashes,
                "resamples": 5000,
                "seed": 1729,
                "interval_method": "centered max-standardized replicate bootstrap",
            }
        )
    )
    with (analysis / "verdicts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("scenario", "horizon", "return_type", "verdict")
        )
        writer.writeheader()
        for scenario in SCENARIOS:
            for return_type in RETURNS:
                writer.writerow(
                    dict(
                        scenario=scenario,
                        horizon=1000,
                        return_type=return_type,
                        verdict="supported",
                    )
                )
    return analysis


def test_balanced_gate_requires_exact_frozen_design(tmp_path):
    analysis = _gate(tmp_path)
    assert validate_gate(analysis)["scenario_return_cells"] == 6
    producer = analysis.parent / "run" / "manifest.json"
    changed = json.loads(producer.read_text())
    changed["config"]["seed"] += 1
    producer_bytes = json.dumps(changed).encode()
    producer.write_bytes(producer_bytes)
    analysis_manifest = analysis / "analysis_manifest.json"
    metadata = json.loads(analysis_manifest.read_text())
    metadata["input_manifest_sha256"] = hashlib.sha256(producer_bytes).hexdigest()
    analysis_manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="seed"):
        validate_gate(analysis)


def test_balanced_gate_rejects_duplicate_or_unsupported_verdict(tmp_path):
    analysis = _gate(tmp_path)
    verdicts_path = analysis / "verdicts.csv"
    with verdicts_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["verdict"] = "inconclusive"
    with verdicts_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="unsupported"):
        validate_gate(analysis)
    rows[0] = dict(rows[1])
    with verdicts_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="unique"):
        validate_gate(analysis)
