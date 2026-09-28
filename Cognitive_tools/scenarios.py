"""Supported ecological scenarios and deterministic landscape construction."""
from __future__ import annotations

import numpy as np
from .ecology import make_capacity_map

SCENARIOS = {
    "uniform_high": {
        "label": "Uniform high",
        "group": "single",
        "family": "uniform",
        "mean_capacity": 0.75,
        "recovery": 0.05,
        "equilibrium": 0.70,
    },
    "patchy_high": {
        "label": "Patchy high",
        "group": "single",
        "family": "patchy",
        "mean_capacity": 0.75,
        "recovery": 0.05,
        "equilibrium": 0.70,
    },
    "centralized_low": {
        "label": "Centralized low",
        "group": "single",
        "family": "centralized",
        "mean_capacity": 0.35,
        "recovery": 0.03,
        "equilibrium": 0.60,
    },
    "decentralized_high": {
        "label": "Decentralized high",
        "group": "single",
        "family": "decentralized",
        "mean_capacity": 0.75,
        "recovery": 0.05,
        "equilibrium": 0.70,
    },
    "patchy_high_central_low": {
        "label": "Patchy high + central low",
        "group": "mixed",
        "kind": "central_low",
    },
    "central_high_patchy_low": {
        "label": "Central high + patchy low",
        "group": "mixed",
        "kind": "central_high",
    },
    "split_high_low": {
        "label": "High patchy + low fragmented split",
        "group": "mixed",
        "kind": "split",
    },
    "decentralized_high_in_low": {
        "label": "Decentralized high islands + low background",
        "group": "mixed",
        "kind": "high_islands",
    },
}


def gaussian_mask(
    width: int,
    height: int,
    *,
    centres: list[
        tuple[
            float,
            float,
        ]
    ],
    sigma: float,
    threshold: float,
) -> np.ndarray:
    y, x = np.indices(
        (
            height,
            width,
        )
    )

    field = np.zeros(
        (
            height,
            width,
        ),
        dtype=float,
    )

    for (
        centre_x,
        centre_y,
    ) in centres:
        distance_squared = (
            (
                x
                - centre_x
            )
            ** 2
            + (
                y
                - centre_y
            )
            ** 2
        )

        field += np.exp(
            -distance_squared
            / (
                2.0
                * sigma**2
            )
        )

    if (
        field.max()
        > 0
    ):
        field = (
            field
            / field.max()
        )

    return (
        field
        >= threshold
    )


def build_environment_maps(
    scenario_name: str,
    *,
    width: int,
    height: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Return:

        capacity
        regeneration-rate map
        equilibrium-fraction map
        region labels
    """

    spec = SCENARIOS[
        scenario_name
    ]

    if (
        spec[
            "group"
        ]
        == "single"
    ):
        capacity = (
            make_capacity_map(
                family=spec[
                    "family"
                ],
                mean_capacity=spec[
                    "mean_capacity"
                ],
                width=width,
                height=height,
                seed=seed,
            )
        )

        recovery = np.full(
            (
                height,
                width,
            ),
            spec[
                "recovery"
            ],
            dtype=float,
        )

        equilibrium = (
            np.full(
                (
                    height,
                    width,
                ),
                spec[
                    "equilibrium"
                ],
                dtype=float,
            )
        )

        regions = np.full(
            (
                height,
                width,
            ),
            scenario_name,
            dtype=object,
        )

        return (
            capacity,
            recovery,
            equilibrium,
            regions,
        )

    kind = spec[
        "kind"
    ]

    high_patchy = (
        make_capacity_map(
            family="patchy",
            mean_capacity=0.75,
            width=width,
            height=height,
            seed=seed,
        )
    )

    low_patchy = (
        make_capacity_map(
            family="patchy",
            mean_capacity=0.30,
            width=width,
            height=height,
            seed=(
                seed
                + 17
            ),
        )
    )

    if (
        kind
        == "central_low"
    ):
        mask = gaussian_mask(
            width,
            height,
            centres=[
                (
                    (
                        width
                        - 1
                    )
                    / 2.0,
                    (
                        height
                        - 1
                    )
                    / 2.0,
                )
            ],
            sigma=(
                min(
                    width,
                    height,
                )
                / 4.0
            ),
            threshold=0.52,
        )

        capacity = (
            high_patchy.copy()
        )

        capacity[
            mask
        ] = low_patchy[
            mask
        ]

        recovery = np.full(
            (
                height,
                width,
            ),
            0.07,
        )

        recovery[
            mask
        ] = 0.02

        equilibrium = np.full(
            (
                height,
                width,
            ),
            0.78,
        )

        equilibrium[
            mask
        ] = 0.48

        regions = np.full(
            (
                height,
                width,
            ),
            "patchy_high",
            dtype=object,
        )

        regions[
            mask
        ] = "central_low"

    elif (
        kind
        == "central_high"
    ):
        mask = gaussian_mask(
            width,
            height,
            centres=[
                (
                    (
                        width
                        - 1
                    )
                    / 2.0,
                    (
                        height
                        - 1
                    )
                    / 2.0,
                )
            ],
            sigma=(
                min(
                    width,
                    height,
                )
                / 4.0
            ),
            threshold=0.52,
        )

        capacity = (
            low_patchy.copy()
        )

        high_central = (
            make_capacity_map(
                family=(
                    "centralized"
                ),
                mean_capacity=(
                    0.78
                ),
                width=width,
                height=height,
                seed=(
                    seed
                    + 31
                ),
            )
        )

        capacity[
            mask
        ] = high_central[
            mask
        ]

        recovery = np.full(
            (
                height,
                width,
            ),
            0.025,
        )

        recovery[
            mask
        ] = 0.08

        equilibrium = np.full(
            (
                height,
                width,
            ),
            0.52,
        )

        equilibrium[
            mask
        ] = 0.82

        regions = np.full(
            (
                height,
                width,
            ),
            "patchy_low",
            dtype=object,
        )

        regions[
            mask
        ] = "central_high"

    elif (
        kind
        == "split"
    ):
        fragmented_low = (
            make_capacity_map(
                family=(
                    "fragmented"
                ),
                mean_capacity=(
                    0.30
                ),
                width=width,
                height=height,
                seed=(
                    seed
                    + 47
                ),
            )
        )

        mask = np.zeros(
            (
                height,
                width,
            ),
            dtype=bool,
        )

        mask[
            :,
            width // 2:
        ] = True

        capacity = (
            high_patchy.copy()
        )

        capacity[
            mask
        ] = fragmented_low[
            mask
        ]

        recovery = np.full(
            (
                height,
                width,
            ),
            0.07,
        )

        recovery[
            mask
        ] = 0.02

        equilibrium = np.full(
            (
                height,
                width,
            ),
            0.78,
        )

        equilibrium[
            mask
        ] = 0.48

        regions = np.full(
            (
                height,
                width,
            ),
            "left_high_patchy",
            dtype=object,
        )

        regions[
            mask
        ] = (
            "right_low_fragmented"
        )

    elif (
        kind
        == "high_islands"
    ):
        centres = [
            (
                width
                * 0.25,
                height
                * 0.25,
            ),
            (
                width
                * 0.75,
                height
                * 0.25,
            ),
            (
                width
                * 0.25,
                height
                * 0.75,
            ),
            (
                width
                * 0.75,
                height
                * 0.75,
            ),
        ]

        mask = gaussian_mask(
            width,
            height,
            centres=centres,
            sigma=1.4,
            threshold=0.48,
        )

        capacity = (
            low_patchy.copy()
        )

        high_islands = (
            make_capacity_map(
                family=(
                    "decentralized"
                ),
                mean_capacity=(
                    0.78
                ),
                width=width,
                height=height,
                seed=(
                    seed
                    + 73
                ),
            )
        )

        capacity[
            mask
        ] = high_islands[
            mask
        ]

        recovery = np.full(
            (
                height,
                width,
            ),
            0.025,
        )

        recovery[
            mask
        ] = 0.075

        equilibrium = np.full(
            (
                height,
                width,
            ),
            0.52,
        )

        equilibrium[
            mask
        ] = 0.80

        regions = np.full(
            (
                height,
                width,
            ),
            "low_background",
            dtype=object,
        )

        regions[
            mask
        ] = "high_island"

    else:
        raise ValueError(
            "Unknown mixed "
            "scenario kind: "
            f"{kind}"
        )

    return (
        capacity,
        recovery,
        equilibrium,
        regions,
    )
