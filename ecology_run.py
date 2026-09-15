import argparse
import csv
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Cognitive_tools.ecology import (
    EcologyModel,
)


# ---------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------

RESULTS_DIR = Path("results")


def ensure_results_directory():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ---------------------------------------------------------------------
# Running an environment
# ---------------------------------------------------------------------

def run_environment(
    *,
    steps,
    distribution,
    mean_capacity,
    heterogeneity,
    regeneration_rate,
    depletion_rate,
    coupling_rate,
    initial_fraction,
    seed,
):
    model = EcologyModel(
        distribution=distribution,
        mean_capacity=mean_capacity,
        heterogeneity=heterogeneity,
        regeneration_rate=regeneration_rate,
        depletion_rate=depletion_rate,
        coupling_rate=coupling_rate,
        initial_fraction=initial_fraction,
        seed=seed,
    )

    for _ in range(steps):
        model.step()

    return model


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def result_row(
    model,
):
    return {
        "steps":
            model.steps,

        "distribution":
            model.distribution,

        "heterogeneity":
            model.heterogeneity,

        "mean_capacity":
            model.mean_capacity,

        "regeneration_rate":
            model.regeneration_rate,

        "depletion_rate":
            model.depletion_rate,

        "coupling_rate":
            model.coupling_rate,

        "initial_fraction":
            model.initial_fraction,

        "total_resource":
            model.total_resource,

        "mean_resource":
            model.mean_resource,

        "resource_fraction":
            model.resource_fraction,

        "capacity_gini":
            model.capacity_gini,

        "resource_gini":
            model.resource_gini,

        "scarcity_fraction":
            model.scarcity_fraction(),
    }


def print_result(
    result,
):
    print(
        f"{result['distribution']:>12} | "
        f"h={result['heterogeneity']:.2f} | "
        f"regen={result['regeneration_rate']:.3f} | "
        f"depletion={result['depletion_rate']:.3f} | "
        f"resource={result['resource_fraction']:.3f} | "
        f"capacity_gini={result['capacity_gini']:.3f} | "
        f"resource_gini={result['resource_gini']:.3f} | "
        f"scarcity={result['scarcity_fraction']:.3f}"
    )


# ---------------------------------------------------------------------
# Individual environment figures
# ---------------------------------------------------------------------

def save_field_image(
    field,
    title,
    label,
    filename,
):
    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(
        field,
        origin="upper",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title(title)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.colorbar(
        image,
        ax=ax,
        label=label,
    )

    fig.tight_layout()

    path = (
        RESULTS_DIR
        / filename
    )

    fig.savefig(
        path,
        dpi=180,
    )

    plt.close(fig)

    print(
        f"Saved {path}"
    )


def save_environment_images(
    model,
):
    """
    Save the persistent carrying-capacity field
    and the final resource field.
    """

    name = (
        f"{model.distribution}"
        f"_h{model.heterogeneity:.2f}"
        f"_r{model.regeneration_rate:.3f}"
        f"_d{model.depletion_rate:.3f}"
    )

    save_field_image(
        model.capacity,
        title=(
            f"Carrying capacity\n"
            f"{model.distribution}, "
            f"heterogeneity="
            f"{model.heterogeneity:.2f}"
        ),
        label="Carrying capacity",
        filename=(
            f"capacity_{name}.png"
        ),
    )

    save_field_image(
        model.resource,
        title=(
            f"Final resource stock\n"
            f"{model.distribution}, "
            f"heterogeneity="
            f"{model.heterogeneity:.2f}, "
            f"regeneration="
            f"{model.regeneration_rate:.3f}"
        ),
        label="Resource stock",
        filename=(
            f"resource_{name}.png"
        ),
    )


# ---------------------------------------------------------------------
# Phase maps
# ---------------------------------------------------------------------

def save_phase_map(
    distribution,
    models,
    heterogeneity_levels,
    regeneration_rates,
):
    """
    Show final resource fields for one spatial
    distribution across heterogeneity and
    regeneration conditions.
    """

    valid_heterogeneity = [
        level
        for level in heterogeneity_levels
        if not (
            distribution == "uniform"
            and level != 0.0
        )
    ]

    rows = len(
        valid_heterogeneity
    )

    cols = len(
        regeneration_rates
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            3.2 * cols,
            3.2 * rows,
        ),
        squeeze=False,
    )

    image = None

    for row_index, heterogeneity in enumerate(
        valid_heterogeneity
    ):

        for col_index, regeneration in enumerate(
            regeneration_rates
        ):

            ax = axes[
                row_index,
                col_index,
            ]

            model = models[
                (
                    distribution,
                    heterogeneity,
                    regeneration,
                )
            ]

            image = ax.imshow(
                model.resource,
                origin="upper",
                vmin=0.0,
                vmax=1.0,
            )

            ax.set_title(
                f"h={heterogeneity:.2f}\n"
                f"r={regeneration:.2f}"
            )

            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_xlabel(
                f"resource="
                f"{model.resource_fraction:.2f}\n"
                f"Gini="
                f"{model.resource_gini:.2f}"
            )

    fig.suptitle(
        f"{distribution.capitalize()} "
        f"environment phase map",
        fontsize=14,
    )

    if image is not None:

        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label="Resource stock",
            shrink=0.8,
        )

    fig.subplots_adjust(
        top=0.90,
        bottom=0.08,
        left=0.05,
        right=0.90,
        hspace=0.40,
        wspace=0.15,
    )

    path = (
        RESULTS_DIR
        / f"phase_map_{distribution}.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved {path}"
    )


# ---------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------

def save_results_csv(
    rows,
    filename,
):
    path = (
        RESULTS_DIR
        / filename
    )

    with open(
        path,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print(
        f"Saved {path}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Scenario 1 ecology-only "
            "environment experiments"
        )
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--distribution",
        choices=[
            "uniform",
            "gradient",
            "patchy",
            "concentrated",
        ],
        default="uniform",
    )

    parser.add_argument(
        "--mean-capacity",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--heterogeneity",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--regen",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--depletion",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--coupling",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--initial-fraction",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "Run the first environmental "
            "phase-map experiment."
        ),
    )

    parser.add_argument(
        "--save-each",
        action="store_true",
        help=(
            "During a sweep, also save "
            "individual images for every run."
        ),
    )

    args = parser.parse_args()

    ensure_results_directory()

    # -------------------------------------------------------------
    # Single environment
    # -------------------------------------------------------------

    if not args.sweep:

        model = run_environment(
            steps=args.steps,
            distribution=args.distribution,
            mean_capacity=args.mean_capacity,
            heterogeneity=args.heterogeneity,
            regeneration_rate=args.regen,
            depletion_rate=args.depletion,
            coupling_rate=args.coupling,
            initial_fraction=args.initial_fraction,
            seed=args.seed,
        )

        result = result_row(
            model
        )

        print_result(
            result
        )

        save_results_csv(
            [result],
            "ecology_single_run.csv",
        )

        save_environment_images(
            model
        )

        return

    # -------------------------------------------------------------
    # Phase-map parameter sweep
    # -------------------------------------------------------------

    distributions = [
        "uniform",
        "gradient",
        "patchy",
        "concentrated",
    ]

    heterogeneity_levels = [
        0.0,
        0.25,
        0.50,
        0.75,
        1.00,
    ]

    regeneration_rates = [
        0.01,
        0.03,
        0.05,
        0.10,
    ]

    rows = []

    models = {}

    for (
        distribution,
        heterogeneity,
        regeneration,
    ) in product(
        distributions,
        heterogeneity_levels,
        regeneration_rates,
    ):

        # Heterogeneity has no meaning in
        # the perfectly uniform condition.
        if (
            distribution == "uniform"
            and heterogeneity != 0.0
        ):

            continue

        model = run_environment(
            steps=args.steps,
            distribution=distribution,
            mean_capacity=args.mean_capacity,
            heterogeneity=heterogeneity,
            regeneration_rate=regeneration,
            depletion_rate=args.depletion,
            coupling_rate=args.coupling,
            initial_fraction=args.initial_fraction,
            seed=args.seed,
        )

        models[
            (
                distribution,
                heterogeneity,
                regeneration,
            )
        ] = model

        result = result_row(
            model
        )

        rows.append(
            result
        )

        print_result(
            result
        )

        if args.save_each:

            save_environment_images(
                model
            )

    # -------------------------------------------------------------
    # Save numerical results
    # -------------------------------------------------------------

    save_results_csv(
        rows,
        "ecology_phase_map.csv",
    )

    # -------------------------------------------------------------
    # Save one visual phase map per
    # distribution type.
    # -------------------------------------------------------------

    for distribution in distributions:

        save_phase_map(
            distribution,
            models,
            heterogeneity_levels,
            regeneration_rates,
        )


if __name__ == "__main__":
    main()