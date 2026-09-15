from __future__ import annotations

import argparse
import shutil
from pathlib import Path


RESULTS_DIR = Path("results")
ARCHIVE_DIR = RESULTS_DIR / "archive"


def classify(
    path: Path,
) -> str:
    name = path.name.lower()

    if name in {
        "agent_animations",
        "agent_frames",
    }:
        return "random_agent_baseline"

    if name in {
        "animations",
        "frames",
    }:
        return "ecology_patch_scan"

    ecology_archetype_terms = (
        "archetype",
        "capacity_equilibrium_response",
        "heterogeneity_gini_response",
        "recovery_response",
        "recovery_rate_vs_recovery_time",
        "resource_fraction_over_time",
        "resource_gini_over_time",
        "scarcity_over_time",
    )

    patch_scan_terms = (
        "patch_scan",
        "capacity_pattern_catalog",
        "final_resource_catalog",
        "abundance_trajectories",
        "resource_fraction_trajectories",
        "resource_gini_trajectories",
        "spatial_organization_trajectories",
        "three_factor_state_space",
        "centralized_vs_decentralized",
        "abundance_vs_inequality",
        "capacity_vs_resource_gini",
    )

    random_agent_terms = (
        "agent_environment",
        "initial_agent",
        "initial_uniform",
        "initial_patchy",
        "initial_centralized",
        "initial_decentralized",
        "initial_fragmented",
        "initial_weak",
        "wealth_gini",
        "welfare",
        "gini_welfare_tradeoff",
    )

    if any(
        term in name
        for term
        in ecology_archetype_terms
    ):
        return "ecology_archetypes"

    if any(
        term in name
        for term
        in patch_scan_terms
    ):
        return "ecology_patch_scan"

    if any(
        term in name
        for term
        in random_agent_terms
    ):
        return "random_agent_baseline"

    return "unclassified"


def unique_destination(
    destination: Path,
) -> Path:
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix

    index = 1

    while True:
        candidate = (
            destination.with_name(
                f"{stem}_{index}"
                f"{suffix}"
            )
        )

        if not candidate.exists():
            return candidate

        index += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Conservatively organize loose files "
            "already inside results/."
        )
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Actually move files. "
            "Without this flag the script "
            "only prints a dry run."
        ),
    )

    args = parser.parse_args()

    if not RESULTS_DIR.exists():
        raise FileNotFoundError(
            "results/ does not exist."
        )

    items = [
        path
        for path
        in RESULTS_DIR.iterdir()
        if path.name
        not in {
            "archive",
            "q_learning_baseline",
        }
    ]

    if not items:
        print(
            "No loose result files found."
        )
        return

    for path in sorted(
        items
    ):
        category = classify(
            path
        )

        destination_dir = (
            ARCHIVE_DIR
            / category
        )

        destination = (
            destination_dir
            / path.name
        )

        destination = (
            unique_destination(
                destination
            )
        )

        if args.apply:
            destination_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.move(
                str(path),
                str(destination),
            )

            print(
                f"MOVED  {path} "
                f"-> {destination}"
            )

        else:
            print(
                f"DRY RUN  {path} "
                f"-> {destination}"
            )

    if not args.apply:
        print(
            "\nNothing was moved. "
            "Run again with --apply "
            "after reviewing the dry run."
        )


if __name__ == "__main__":
    main()
