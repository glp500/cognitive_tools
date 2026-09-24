from __future__ import annotations

import numpy as np

from Cognitive_tools.ecology import (
    coerce_capacity_map,
    coerce_parameter_map,
    depletion_from_equilibrium,
    make_capacity_map,
    neighbour_mean,
    resource_step,
)
from Cognitive_tools.model import EcoModel


def legacy_neighbour_mean(
    field: np.ndarray,
) -> np.ndarray:
    """
    Exact pre-refactor neighbour calculation.

    This copy exists only as a regression oracle for commit:
    `refactor: unify ecological dynamics`.
    """

    padded = np.pad(
        field,
        1,
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


def legacy_resource_step(
    resource: np.ndarray,
    capacity: np.ndarray,
    regeneration_rate: np.ndarray,
    depletion_rate: np.ndarray,
    coupling_rate: float,
) -> np.ndarray:
    """
    Exact ecological update used by EcoModel before the refactor.
    """

    growth = (
        regeneration_rate
        * resource
        * (
            1.0
            - resource / capacity
        )
    )

    depletion = (
        depletion_rate
        * resource
    )

    neighbour = (
        legacy_neighbour_mean(
            resource
        )
    )

    diffusion = (
        coupling_rate
        * (
            neighbour
            - resource
        )
    )

    updated = (
        resource
        + growth
        - depletion
        + diffusion
    )

    return np.clip(
        updated,
        0.0,
        capacity,
    )


def test_capacity_none_becomes_uniform_one():
    capacity = coerce_capacity_map(
        None,
        width=4,
        height=3,
    )

    assert capacity.shape == (
        3,
        4,
    )

    np.testing.assert_allclose(
        capacity,
        1.0,
    )


def test_scalar_parameter_becomes_spatial_map():
    field = coerce_parameter_map(
        0.05,
        width=5,
        height=4,
        name="regeneration_rate",
        minimum=0.0,
    )

    assert field.shape == (
        4,
        5,
    )

    np.testing.assert_allclose(
        field,
        0.05,
    )


def test_depletion_parameterization():
    regeneration = np.array(
        [
            [
                0.10,
                0.04,
            ],
            [
                0.02,
                0.08,
            ],
        ],
        dtype=float,
    )

    equilibrium = np.array(
        [
            [
                0.80,
                0.50,
            ],
            [
                0.25,
                1.00,
            ],
        ],
        dtype=float,
    )

    expected = (
        regeneration
        * (
            1.0
            - equilibrium
        )
    )

    actual = (
        depletion_from_equilibrium(
            regeneration,
            equilibrium,
        )
    )

    np.testing.assert_allclose(
        actual,
        expected,
    )


def test_neighbour_mean_matches_legacy_implementation():
    field = np.array(
        [
            [
                0.1,
                0.2,
                0.3,
            ],
            [
                0.4,
                0.5,
                0.6,
            ],
            [
                0.7,
                0.8,
                0.9,
            ],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        neighbour_mean(
            field
        ),
        legacy_neighbour_mean(
            field
        ),
    )


def test_resource_step_matches_pre_refactor_equation():
    rng = np.random.default_rng(
        42
    )

    capacity = rng.uniform(
        0.30,
        1.00,
        size=(
            6,
            7,
        ),
    )

    resource = (
        capacity
        * rng.uniform(
            0.05,
            0.95,
            size=capacity.shape,
        )
    )

    regeneration = rng.uniform(
        0.01,
        0.10,
        size=capacity.shape,
    )

    equilibrium = rng.uniform(
        0.30,
        0.90,
        size=capacity.shape,
    )

    depletion = (
        depletion_from_equilibrium(
            regeneration,
            equilibrium,
        )
    )

    coupling = 0.10

    expected = (
        legacy_resource_step(
            resource=resource,
            capacity=capacity,
            regeneration_rate=regeneration,
            depletion_rate=depletion,
            coupling_rate=coupling,
        )
    )

    actual = resource_step(
        resource=resource,
        capacity=capacity,
        regeneration_rate=regeneration,
        depletion_rate=depletion,
        coupling_rate=coupling,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-14,
    )


def test_uniform_isolated_equilibrium_is_q_times_k():
    capacity = np.full(
        (
            4,
            4,
        ),
        0.80,
    )

    regeneration = np.full(
        (
            4,
            4,
        ),
        0.05,
    )

    equilibrium = np.full(
        (
            4,
            4,
        ),
        0.70,
    )

    depletion = (
        depletion_from_equilibrium(
            regeneration,
            equilibrium,
        )
    )

    resource = (
        equilibrium
        * capacity
    )

    updated = resource_step(
        resource=resource,
        capacity=capacity,
        regeneration_rate=regeneration,
        depletion_rate=depletion,
        coupling_rate=0.0,
    )

    np.testing.assert_allclose(
        updated,
        resource,
        rtol=0.0,
        atol=1e-14,
    )


def test_model_ecology_uses_same_central_equation():
    capacity = np.array(
        [
            [
                0.50,
                0.60,
                0.70,
            ],
            [
                0.55,
                0.65,
                0.75,
            ],
            [
                0.60,
                0.70,
                0.80,
            ],
        ],
        dtype=float,
    )

    regeneration = np.array(
        [
            [
                0.04,
                0.05,
                0.06,
            ],
            [
                0.03,
                0.04,
                0.05,
            ],
            [
                0.02,
                0.03,
                0.04,
            ],
        ],
        dtype=float,
    )

    equilibrium = np.full(
        (
            3,
            3,
        ),
        0.70,
    )

    model = EcoModel(
        width=3,
        height=3,
        n_agents=1,
        regeneration_rate=(
            regeneration
        ),
        equilibrium_fraction=(
            equilibrium
        ),
        coupling_rate=0.10,
        cooperative_harvest_amount=0.0,
        defective_harvest_amount=0.0,
        metabolism_rate=0.0,
        capacity=capacity,
        seed=42,
    )

    initial = (
        0.50
        * capacity
    )

    model.resource = (
        initial.copy()
    )

    expected = resource_step(
        resource=initial,
        capacity=capacity,
        regeneration_rate=(
            model.regeneration_map
        ),
        depletion_rate=(
            model.depletion_map
        ),
        coupling_rate=(
            model.coupling_rate
        ),
    )

    model.step()

    np.testing.assert_allclose(
        model.resource,
        expected,
        rtol=0.0,
        atol=1e-14,
    )


def test_capacity_map_is_reproducible():
    first = make_capacity_map(
        family="patchy",
        mean_capacity=0.75,
        width=10,
        height=10,
        seed=42,
    )

    second = make_capacity_map(
        family="patchy",
        mean_capacity=0.75,
        width=10,
        height=10,
        seed=42,
    )

    np.testing.assert_array_equal(
        first,
        second,
    )


def test_capacity_map_stays_inside_bounds():
    for family in (
        "uniform",
        "weak_patchy",
        "patchy",
        "centralized",
        "decentralized",
        "fragmented",
    ):
        capacity = make_capacity_map(
            family=family,
            mean_capacity=0.60,
            width=10,
            height=10,
            seed=42,
        )

        assert np.all(
            capacity > 0.0
        )

        assert np.all(
            capacity <= 1.0
        )