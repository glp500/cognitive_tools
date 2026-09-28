"""
Core ecological functions for cognitive_tools.

This module is the single source of truth for:

1. spatial renewable-resource dynamics;
2. ecological parameter maps;
3. carrying-capacity landscape construction.

No Mesa model or experiment loop belongs in this module. Functions here
operate on NumPy arrays and contain no agent-learning or social-network
logic.

Scientific provenance
---------------------
The renewable-resource component is conceptually based on the class of
eco-evolutionary common-pool-resource models in which renewable resources
grow logistically and are depleted by human extraction, including:

    Tilman, A. R., Plotkin, J. B., & Akcay, E. (2020).
    "Evolutionary games with environmental feedbacks."
    Nature Communications, 11, 915.
    https://doi.org/10.1038/s41467-020-14531-6

and:

    Tu, C., Wu, Y., Chen, R., Fan, Y., & Yang, Y. (2025).
    "Balancing Resource and Strategy: Coevolution for Sustainable
    Common-Pool Resource Management."
    Earth Systems and Environment, 9, 1529-1542.
    https://doi.org/10.1007/s41748-024-00489-8

This implementation is NOT a reproduction of either published model.

Project-specific features include:

- a spatial grid;
- heterogeneous carrying capacity K(x);
- heterogeneous regeneration and equilibrium maps;
- four-neighbour spatial resource coupling;
- the parameterization d(x) = r(x) * [1 - q(x)].

Under no harvesting and no spatial coupling, this parameterization gives
the positive local equilibrium:

    R*(x) = q(x) K(x)

Code provenance
---------------
The carrying-capacity pattern functions near the bottom of this file were
moved from this repository's former `ecology_patch_scan.py` implementation
at cognitive_tools commit:

    501cb693bc8a48cb957ea14db9212c2f4e0e453a

They are repository-local code, not copied from an external project.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------
# Existing landscape definitions
# ---------------------------------------------------------------------
#
# These are retained during the refactor so that moving ecological
# functions does not silently change the current experimental baselines.
# The number of ecological treatments can be simplified in a later,
# explicitly scientific commit.

ABUNDANCE_LEVELS = {
    "high": 0.75,
    "low": 0.35,
}


PATTERN_FAMILIES = {
    "uniform": {
        "label": "Uniform",
        "heterogeneity_strength": 0.00,
    },
    "weak_patchy": {
        "label": "Weak patchy",
        "heterogeneity_strength": 0.12,
    },
    "patchy": {
        "label": "Patchy",
        "heterogeneity_strength": 0.28,
    },
    "centralized": {
        "label": "Centralized",
        "heterogeneity_strength": 0.35,
    },
    "decentralized": {
        "label": "Decentralized",
        "heterogeneity_strength": 0.35,
    },
    "fragmented": {
        "label": "Fragmented",
        "heterogeneity_strength": 0.42,
    },
}


# ---------------------------------------------------------------------
# Parameter-map helpers
# ---------------------------------------------------------------------


def coerce_capacity_map(
    capacity: np.ndarray | None,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Return a validated carrying-capacity map.

    Parameters
    ----------
    capacity
        A (height, width) array, or None for a uniform capacity of 1.
    width, height
        Spatial dimensions.

    Returns
    -------
    np.ndarray
        Floating-point carrying-capacity map with values in (0, 1].
    """

    if width <= 0 or height <= 0:
        raise ValueError(
            "width and height must be positive."
        )

    if capacity is None:
        return np.ones(
            (height, width),
            dtype=float,
        )

    field = np.asarray(
        capacity,
        dtype=float,
    )

    expected_shape = (
        height,
        width,
    )

    if field.shape != expected_shape:
        raise ValueError(
            "capacity must have shape "
            f"{expected_shape}, got {field.shape}."
        )

    if np.any(field <= 0.0):
        raise ValueError(
            "capacity values must be > 0."
        )

    if np.any(field > 1.0):
        raise ValueError(
            "capacity values must be <= 1."
        )

    return field.copy()


def coerce_parameter_map(
    value: float | np.ndarray,
    *,
    width: int,
    height: int,
    name: str,
    minimum: float,
    maximum: float | None = None,
) -> np.ndarray:
    """
    Convert a scalar or spatial parameter into a validated map.

    A scalar is broadcast over the entire environment. An array must
    already have shape (height, width).
    """

    if np.isscalar(value):
        field = np.full(
            (height, width),
            float(value),
            dtype=float,
        )

    else:
        field = np.asarray(
            value,
            dtype=float,
        )

        expected_shape = (
            height,
            width,
        )

        if field.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape "
                f"{expected_shape}, got {field.shape}."
            )

        field = field.copy()

    if np.any(field < minimum):
        raise ValueError(
            f"{name} must be >= {minimum}."
        )

    if (
        maximum is not None
        and np.any(field > maximum)
    ):
        raise ValueError(
            f"{name} must be <= {maximum}."
        )

    return field


def depletion_from_equilibrium(
    regeneration_rate: np.ndarray,
    equilibrium_fraction: np.ndarray,
) -> np.ndarray:
    """
    Convert regeneration and target equilibrium fraction to depletion.

    The project parameterization is:

        d(x) = r(x) * [1 - q(x)]

    so that for an isolated, unharvested tile:

        dR/dt = r R (1 - R/K) - d R

    has the positive equilibrium:

        R* = q K.

    This parameterization is specific to cognitive_tools.
    """

    regeneration_rate = np.asarray(
        regeneration_rate,
        dtype=float,
    )

    equilibrium_fraction = np.asarray(
        equilibrium_fraction,
        dtype=float,
    )

    if (
        regeneration_rate.shape
        != equilibrium_fraction.shape
    ):
        raise ValueError(
            "regeneration_rate and equilibrium_fraction "
            "must have the same shape."
        )

    return (
        regeneration_rate
        * (
            1.0
            - equilibrium_fraction
        )
    )


# ---------------------------------------------------------------------
# Spatial relationship
# ---------------------------------------------------------------------


def neighbour_mean(
    field: np.ndarray,
) -> np.ndarray:
    """
    Mean of the four orthogonally adjacent cells.

    Boundaries do not wrap. Edge padding duplicates the boundary value,
    preserving the boundary behavior used by the previous implementation.
    """

    field = np.asarray(
        field,
        dtype=float,
    )

    if field.ndim != 2:
        raise ValueError(
            "field must be a two-dimensional array."
        )

    padded = np.pad(
        field,
        pad_width=1,
        mode="edge",
    )

    up = padded[
        :-2,
        1:-1,
    ]

    down = padded[
        2:,
        1:-1,
    ]

    left = padded[
        1:-1,
        :-2,
    ]

    right = padded[
        1:-1,
        2:,
    ]

    return (
        up
        + down
        + left
        + right
    ) / 4.0


# ---------------------------------------------------------------------
# Single ecological update
# ---------------------------------------------------------------------


def resource_step(
    resource: np.ndarray,
    capacity: np.ndarray,
    regeneration_rate: np.ndarray,
    depletion_rate: np.ndarray,
    coupling_rate: float,
) -> np.ndarray:
    """
    Advance the renewable-resource field by one ecological timestep.

    The update is:

        R(t+1)
        =
        R(t)
        + r R(t) [1 - R(t)/K]
        - d R(t)
        + c [mean_neighbour(R(t)) - R(t)]

    followed by clipping to:

        0 <= R <= K.

    Important
    ---------
    Harvest is deliberately NOT performed in this function.

    `EcoModel` first resolves extraction and subtracts realized harvest
    from the resource field. This function then advances that post-harvest
    resource state.

    That ordering exactly preserves the current cognitive_tools baseline.

    Scientific lineage
    ------------------
    Logistic renewable-resource growth coupled to extraction follows the
    general eco-evolutionary resource framing of Tilman et al. (2020) and
    Tu et al. (2025).

    The discrete spatial formulation, explicit background depletion term,
    and four-neighbour coupling are cognitive_tools-specific extensions.
    """

    resource = np.asarray(
        resource,
        dtype=float,
    )

    capacity = np.asarray(
        capacity,
        dtype=float,
    )

    regeneration_rate = np.asarray(
        regeneration_rate,
        dtype=float,
    )

    depletion_rate = np.asarray(
        depletion_rate,
        dtype=float,
    )

    shape = resource.shape

    if resource.ndim != 2:
        raise ValueError(
            "resource must be a two-dimensional array."
        )

    for name, field in (
        ("capacity", capacity),
        (
            "regeneration_rate",
            regeneration_rate,
        ),
        (
            "depletion_rate",
            depletion_rate,
        ),
    ):
        if field.shape != shape:
            raise ValueError(
                f"{name} must have shape "
                f"{shape}, got {field.shape}."
            )

    if np.any(capacity <= 0.0):
        raise ValueError(
            "capacity values must be > 0."
        )

    if not 0.0 <= coupling_rate <= 1.0:
        raise ValueError(
            "coupling_rate must be between 0 and 1."
        )

    # Local logistic renewal.
    growth = (
        regeneration_rate
        * resource
        * (
            1.0
            - resource / capacity
        )
    )

    # Background ecological depletion.
    depletion = (
        depletion_rate
        * resource
    )

    # Exchange with the four neighbouring cells.
    exchange = (
        coupling_rate
        * (
            neighbour_mean(resource)
            - resource
        )
    )

    updated = (
        resource
        + growth
        - depletion
        + exchange
    )

    return np.clip(
        updated,
        0.0,
        capacity,
    )


# ---------------------------------------------------------------------
# Carrying-capacity landscape construction
# ---------------------------------------------------------------------
#
# CODE PROVENANCE:
# These functions are migrated from the repository-local
# `ecology_patch_scan.py` at cognitive_tools commit
# 501cb693bc8a48cb957ea14db9212c2f4e0e453a.
#
# No external source code is copied here.


def smooth_field(
    field: np.ndarray,
    passes: int,
) -> np.ndarray:
    """
    Neighbour smoothing used to produce spatially autocorrelated patches.
    """

    output = np.asarray(
        field,
        dtype=float,
    ).copy()

    for _ in range(passes):
        padded = np.pad(
            output,
            1,
            mode="edge",
        )

        centre = padded[
            1:-1,
            1:-1,
        ]

        up = padded[
            :-2,
            1:-1,
        ]

        down = padded[
            2:,
            1:-1,
        ]

        left = padded[
            1:-1,
            :-2,
        ]

        right = padded[
            1:-1,
            2:,
        ]

        output = (
            4.0 * centre
            + up
            + down
            + left
            + right
        ) / 8.0

    return output


def rescale_pattern(
    raw: np.ndarray,
    mean_capacity: float,
    heterogeneity_strength: float,
    min_capacity: float = 0.05,
    max_capacity: float = 1.0,
) -> np.ndarray:
    """
    Scale a spatial pattern around the requested mean carrying capacity.
    """

    raw = np.asarray(
        raw,
        dtype=float,
    )

    if not 0.0 < mean_capacity <= 1.0:
        raise ValueError(
            "mean_capacity must be in (0, 1]."
        )

    if heterogeneity_strength < 0.0:
        raise ValueError(
            "heterogeneity_strength must be non-negative."
        )

    if np.allclose(
        raw,
        raw.flat[0],
    ):
        return np.full_like(
            raw,
            mean_capacity,
            dtype=float,
        )

    centered = (
        raw
        - raw.mean()
    )

    maximum = np.max(
        np.abs(centered)
    )

    if maximum > 0.0:
        centered = (
            centered
            / maximum
        )

    amplitude = (
        heterogeneity_strength
        * mean_capacity
    )

    capacity = (
        mean_capacity
        + amplitude * centered
    )

    capacity = np.clip(
        capacity,
        min_capacity,
        max_capacity,
    )

    # Preserve approximately equal mean abundance after clipping.
    capacity += (
        mean_capacity
        - capacity.mean()
    )

    capacity = np.clip(
        capacity,
        min_capacity,
        max_capacity,
    )

    return capacity


def gaussian_hotspot(
    width: int,
    height: int,
    centre_x: float,
    centre_y: float,
    sigma: float,
) -> np.ndarray:
    """Create a two-dimensional Gaussian spatial hotspot."""

    if sigma <= 0.0:
        raise ValueError(
            "sigma must be positive."
        )

    y, x = np.indices(
        (
            height,
            width,
        )
    )

    distance_squared = (
        (x - centre_x) ** 2
        + (y - centre_y) ** 2
    )

    return np.exp(
        -distance_squared
        / (
            2.0
            * sigma**2
        )
    )


def make_capacity_map(
    family: str,
    mean_capacity: float,
    width: int,
    height: int,
    seed: int,
) -> np.ndarray:
    """
    Construct one of the existing carrying-capacity landscape families.

    This function preserves the landscape generator used by the current
    baseline. Simplification of the ecological treatment set should occur
    in a later scientific-design commit, not in this mechanical refactor.
    """

    if family not in PATTERN_FAMILIES:
        raise ValueError(
            "Unknown capacity-map family: "
            f"{family!r}. "
            f"Valid families: "
            f"{', '.join(PATTERN_FAMILIES)}."
        )

    rng = np.random.default_rng(
        seed
    )

    strength = PATTERN_FAMILIES[
        family
    ][
        "heterogeneity_strength"
    ]

    if family == "uniform":
        raw = np.ones(
            (
                height,
                width,
            ),
            dtype=float,
        )

    elif family == "weak_patchy":
        raw = smooth_field(
            rng.random(
                (
                    height,
                    width,
                )
            ),
            passes=10,
        )

    elif family == "patchy":
        raw = smooth_field(
            rng.random(
                (
                    height,
                    width,
                )
            ),
            passes=5,
        )

    elif family == "fragmented":
        raw = smooth_field(
            rng.random(
                (
                    height,
                    width,
                )
            ),
            passes=2,
        )

    elif family == "centralized":
        raw = gaussian_hotspot(
            width=width,
            height=height,
            centre_x=(
                width - 1
            ) / 2.0,
            centre_y=(
                height - 1
            ) / 2.0,
            sigma=(
                min(
                    width,
                    height,
                )
                / 4.0
            ),
        )

    elif family == "decentralized":
        raw = (
            gaussian_hotspot(
                width=width,
                height=height,
                centre_x=width * 0.25,
                centre_y=height * 0.25,
                sigma=1.6,
            )
            + gaussian_hotspot(
                width=width,
                height=height,
                centre_x=width * 0.75,
                centre_y=height * 0.25,
                sigma=1.6,
            )
            + gaussian_hotspot(
                width=width,
                height=height,
                centre_x=width * 0.25,
                centre_y=height * 0.75,
                sigma=1.6,
            )
            + gaussian_hotspot(
                width=width,
                height=height,
                centre_x=width * 0.75,
                centre_y=height * 0.75,
                sigma=1.6,
            )
        )

    else:
        # Kept for type-checking completeness.
        raise ValueError(
            f"Unknown family: {family}"
        )

    return rescale_pattern(
        raw=raw,
        mean_capacity=mean_capacity,
        heterogeneity_strength=strength,
    )