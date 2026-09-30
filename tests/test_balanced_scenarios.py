import numpy as np
import pytest

from cognitive_tools.scenarios import BALANCED_SCENARIOS, build_environment_maps


@pytest.mark.parametrize("seed", [20261012, 20261013])
def test_balanced_maps_match_capacity_and_differ_only_in_organization(seed):
    maps = {
        name: build_environment_maps(name, width=10, height=10, seed=seed)
        for name in BALANCED_SCENARIOS
    }
    for capacity, recovery, equilibrium, _ in maps.values():
        assert capacity.sum() == pytest.approx(75)
        assert np.allclose(recovery, 0.05)
        assert np.allclose(equilibrium, 0.70)
        assert np.all((capacity > 0) & (capacity <= 1))
    uniform, dispersed, segregated = (maps[name][0] for name in BALANCED_SCENARIOS)
    assert np.allclose(uniform, 0.75)
    assert np.array_equal(np.sort(dispersed.ravel()), np.sort(segregated.ravel()))
    assert np.count_nonzero(dispersed == 0.55) == 50
    assert np.count_nonzero(segregated == 0.95) == 50
    assert not np.array_equal(dispersed, segregated)


def test_balanced_maps_require_even_cell_count():
    with pytest.raises(ValueError, match="even number"):
        build_environment_maps("balanced_dispersed", width=3, height=3, seed=1)


def test_landscape_validation_records_unharvested_supply():
    from scripts.validate_balanced_landscapes import validate

    rows = validate(seed=20261012, replicates=2, steps=100)
    assert len(rows) == 6
    assert all(row["total_capacity"] == pytest.approx(75.0) for row in rows)
    assert all(row["initial_resource"] == pytest.approx(37.5) for row in rows)
    assert all(0 < row["total_resource"] <= 75 for row in rows)
    for replicate in (0, 1):
        by_name = {row["scenario"]: row for row in rows if row["replicate"] == replicate}
        assert (
            by_name["balanced_dispersed"]["mixed_neighbor_edges"]
            > by_name["balanced_segregated"]["mixed_neighbor_edges"]
        )
