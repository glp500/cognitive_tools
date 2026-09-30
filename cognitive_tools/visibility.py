"""Stage-5 visibility-profile specification, sampling, and provenance."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .social import init_random_attention, visibility_counts

PROTOCOL = "visibility_bounded_search_v1"
PROFILES = (
    "equal",
    "random",
    "normal_centered",
    "low_propensity_majority",
    "high_propensity_majority",
)
DEFAULT_SPEC = (
    Path(__file__).resolve().parent.parent / "configs/visibility/visibility_profiles_v1.json"
)


@dataclass
class SocialInitialization:
    sources: dict[str, list[str]]
    raw_propensity: dict[str, float]
    effective_weight: dict[str, float]


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def graph_hash(sources: dict[str, list[str]]) -> str:
    """Hash directed edges independent of source-list order."""
    return canonical_hash({observer: sorted(row) for observer, row in sources.items()})


def load_visibility_spec(path: str | Path = DEFAULT_SPEC) -> tuple[dict, str]:
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    spec = json.loads(raw)
    if spec.get("version") != "v1" or set(spec.get("profiles", {})) != set(PROFILES):
        raise ValueError("Visibility spec must be v1 with exactly the five frozen profiles")
    for profile, definition in spec["profiles"].items():
        distribution = definition.get("distribution")
        network = definition.get("network")
        if (
            (profile == "equal" and (distribution, network) != ("constant", "regular"))
            or (profile == "random" and (distribution, network) != ("constant", "uniform"))
            or (profile not in ("equal", "random") and network != "weighted")
        ):
            raise ValueError(f"Invalid visibility profile definition: {profile}")
        if distribution == "constant":
            if not 0 < float(definition["value"]) <= 1:
                raise ValueError("Constant propensity must lie in (0,1]")
        elif distribution == "truncated_normal":
            mean, sd = float(definition["mean"]), float(definition["sd"])
            if not (0 < mean < 1 and math.isfinite(sd) and 0 < sd <= 1):
                raise ValueError("Invalid truncated-normal parameters")
        elif distribution == "beta":
            alpha, beta = float(definition["alpha"]), float(definition["beta"])
            if not (
                math.isfinite(alpha)
                and math.isfinite(beta)
                and 0 < alpha <= 1000
                and 0 < beta <= 1000
            ):
                raise ValueError("Beta parameters must be finite and positive")
        else:
            raise ValueError(f"Unknown propensity distribution: {distribution}")
    return spec, hashlib.sha256(raw).hexdigest()


def _propensities(names, definition, rng):
    kind = definition["distribution"]
    if kind == "constant":
        return np.full(len(names), float(definition["value"]))
    if kind == "beta":
        return rng.beta(float(definition["alpha"]), float(definition["beta"]), len(names))
    # Rejection sampling gives a true truncated normal, with no mass at 0 or 1.
    values = np.empty(len(names))
    remaining = np.arange(len(names))
    attempts = 0
    while len(remaining):
        attempts += 1
        if attempts > 1000:
            raise ValueError("Visibility normal sampler did not converge")
        draws = rng.normal(float(definition["mean"]), float(definition["sd"]), len(remaining))
        valid = (draws > 0) & (draws < 1)
        values[remaining[valid]] = draws[valid]
        remaining = remaining[~valid]
    return values


def _equal_attention(names, k, rng):
    # A random relabelled circulant graph is k-regular in both directions.
    order = list(rng.permutation(names))
    n = len(names)
    sources = {order[i]: [order[(i + shift) % n] for shift in range(1, k + 1)] for i in range(n)}
    # Degree-preserving swaps mix the graph while keeping each observer's order stable.
    for _ in range(20 * n * k):
        first, second = rng.choice(n, size=2, replace=False)
        a, b = order[int(first)], order[int(second)]
        ai, bi = int(rng.integers(k)), int(rng.integers(k))
        x, y = sources[a][ai], sources[b][bi]
        if x != y and x != b and y != a and y not in sources[a] and x not in sources[b]:
            sources[a][ai], sources[b][bi] = y, x
    return {name: sources[name] for name in names}


def init_visibility_attention(agents, *, k, profile, rng, spec=None) -> SocialInitialization:
    names = list(agents)
    if len(names) != len(set(names)) or not 1 <= k < len(names):
        raise ValueError("Unique agents and 1 <= k < population are required")
    if spec is None:
        spec, _ = load_visibility_spec()
    if profile not in PROFILES:
        raise ValueError(f"Unknown visibility profile: {profile}")
    definition = spec["profiles"][profile]
    weights = _propensities(names, definition, rng)
    if definition["network"] == "regular":
        sources = _equal_attention(names, k, rng)
    elif definition["network"] == "uniform":
        sources = init_random_attention(names, k, rng)
    else:
        sources = {}
        for observer in names:
            candidates = [name for name in names if name != observer]
            probabilities = np.array([weights[names.index(name)] for name in candidates])
            probabilities /= probabilities.sum()
            sources[observer] = list(rng.choice(candidates, size=k, replace=False, p=probabilities))
    counts = visibility_counts(sources)
    if any(len(set(row)) != k or name in row for name, row in sources.items()):
        raise RuntimeError("Invalid initial attention capacity or self-link")
    if sum(counts.values()) != len(names) * k or (
        profile == "equal" and any(v != k for v in counts.values())
    ):
        raise RuntimeError("Invalid initial visibility counts")
    propensity = dict(zip(names, map(float, weights)))
    return SocialInitialization(sources, propensity, propensity.copy())
