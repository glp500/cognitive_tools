import numpy as np
import pytest

from cognitive_tools.social import visibility_counts
from cognitive_tools.visibility import (
    PROFILES,
    canonical_hash,
    init_visibility_attention,
    load_visibility_spec,
)


@pytest.mark.parametrize("profile", PROFILES)
def test_visibility_profiles_are_reproducible_and_fixed_capacity(profile):
    spec, _ = load_visibility_spec()
    names = [f"agent_{i}" for i in range(64)]
    first = init_visibility_attention(
        names, k=4, profile=profile, rng=np.random.default_rng(123), spec=spec
    )
    second = init_visibility_attention(
        names, k=4, profile=profile, rng=np.random.default_rng(123), spec=spec
    )
    assert first == second
    assert canonical_hash(first.sources) == canonical_hash(second.sources)
    counts = visibility_counts(first.sources)
    assert sum(counts.values()) == 256
    assert all(
        len(row) == len(set(row)) == 4 and name not in row for name, row in first.sources.items()
    )
    assert all(0 < value <= 1 for value in first.raw_propensity.values())
    if profile == "equal":
        assert set(counts.values()) == {4}
    elif profile == "random":
        assert len(set(counts.values())) > 1
