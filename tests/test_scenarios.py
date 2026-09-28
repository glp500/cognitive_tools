"""Exact map regressions captured from the pre-cleanup scientific snapshot.

The four hashes cover capacity, regeneration, equilibrium and region arrays,
respectively, for every supported scenario on square and rectangular grids.
"""

import hashlib
import json
from pathlib import Path

import pytest

from cognitive_tools.scenarios import SCENARIOS, build_environment_maps

REFERENCE = json.loads(Path(__file__).with_name("scenario_reference.json").read_text())


@pytest.mark.parametrize(
    "case", REFERENCE, ids=lambda c: f"{c['scenario']}-{c['width']}x{c['height']}-{c['seed']}"
)
def test_scenario_matches_scientific_snapshot(case):
    maps = build_environment_maps(
        case["scenario"], width=case["width"], height=case["height"], seed=case["seed"]
    )
    assert [
        hashlib.sha256(json.dumps(a.tolist(), separators=(",", ":")).encode()).hexdigest()
        for a in maps
    ] == case["hashes"]
    assert all(a.shape == (case["height"], case["width"]) for a in maps)


def test_reference_covers_supported_scenarios():
    assert {case["scenario"] for case in REFERENCE} == set(SCENARIOS)
