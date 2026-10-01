"""Re-render the eight visibility analysis figures from saved analysis tables.

This changes presentation only; it does not recompute estimands or simulations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cognitive_tools.visibility_figures import save_visibility_figures

TABLES = (
    "initial_visibility_agents",
    "initial_visibility_summary",
    "primary_window_summary",
    "adaptive_fixed_summary",
    "outcome_summary",
    "training_trajectory_summary",
)
FIGURES = (
    "01_mechanism_and_design",
    "02_initial_visibility_manipulation",
    "03_population_perception",
    "04_collective_organization",
    "05_secondary_consequences",
    "supplement_gini_decomposition",
    "S2_visibility_gini_trajectories",
    "S2_low_extraction_trajectories",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.analysis_dir.resolve()
    manifest = json.loads((root / "analysis_manifest.json").read_text())
    if manifest.get("profile") != "visibility" or len(manifest.get("inputs", [])) != 11:
        raise ValueError("Expected a complete eleven-treatment visibility analysis")
    config = json.loads((Path(manifest["inputs"][0]["run_path"]) / "config.json").read_text())
    ecologies = tuple(config["scenarios"])
    tables = {}
    for stem in TABLES:
        path = root / "data" / f"{stem}.csv"
        with path.open(newline="") as handle:
            tables[stem] = list(csv.DictReader(handle))
        if not tables[stem]:
            raise ValueError(f"Empty required figure table: {path}")
    output = (args.output or root / "figures").resolve()
    save_visibility_figures(
        output, tables, ecologies=ecologies,
        width=int(config.get("width", 10)), height=int(config.get("height", 10)),
        seed=int(config.get("seed", 20261002)),
    )
    missing = [name for name in FIGURES if not (output / f"{name}.pdf").is_file()]
    if missing:
        raise RuntimeError(f"Missing expected figures: {missing}")
    print(json.dumps({
        "event": "visibility_main_complete", "analysis": root.name,
        "figure_count": len(FIGURES), "output": str(output),
    }))


if __name__ == "__main__":
    main()
