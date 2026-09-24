# Scientific and Code Provenance

This document records where the mechanisms implemented in
`cognitive_tools` originate, how they differ from their source models,
and whether source code was reused.

The purpose is to make the simulation mechanically transparent and to
distinguish reproduced, adapted, inspired, and project-original
components.

## Provenance labels

**REPRODUCED**

The mathematical or computational mechanism is intended to reproduce a
source mechanism closely.

**ADAPTED**

A recognizable source mechanism is retained but modified for the current
model.

**INSPIRED**

The source provides the scientific framing or modeling principle, but
the implementation is substantially different.

**ORIGINAL**

The mechanism was introduced specifically for `cognitive_tools`.

---

# Ecology

## Renewable-resource feedback

**Status:** INSPIRED

**Implementation:**

- `Cognitive_tools/ecology.py::resource_step`
- `Cognitive_tools/model.py::EcoModel`

**Scientific sources:**

Tilman, A. R., Plotkin, J. B., & Akcay, E. (2020).
"Evolutionary games with environmental feedbacks."
Nature Communications, 11, 915.
https://doi.org/10.1038/s41467-020-14531-6

Tu, C., Wu, Y., Chen, R., Fan, Y., & Yang, Y. (2025).
"Balancing Resource and Strategy: Coevolution for Sustainable
Common-Pool Resource Management."
Earth Systems and Environment, 9, 1529-1542.
https://doi.org/10.1007/s41748-024-00489-8

**Source principle:**

Renewable common-pool resources regenerate through logistic dynamics,
while human extraction changes the resource state. Resource condition
can in turn alter the consequences and incentives associated with human
behavior.

**cognitive_tools implementation:**

The resource is spatial rather than scalar:

    R(x, t+1)
    =
    R(x, t)
    + r(x) R(x, t) [1 - R(x, t)/K(x)]
    - d(x) R(x, t)
    + c [mean_neighbor(R) - R(x, t)]

after realized harvest has been removed.

The project defines:

    d(x) = r(x) [1 - q(x)]

so that an isolated unharvested tile has positive equilibrium:

    R*(x) = q(x) K(x).

**Project-specific changes:**

- spatial resource grid;
- heterogeneous carrying capacity K(x);
- heterogeneous regeneration r(x);
- heterogeneous equilibrium fraction q(x);
- four-neighbour resource coupling;
- discrete-time implementation;
- explicit separation between harvesting and ecological renewal.

No source code from Tilman et al. or Tu et al. is copied into this
repository. Their work supplies scientific/modeling lineage.

---

# Spatial landscape construction

**Status:** ORIGINAL / repository-local

**Implementation:**

- `Cognitive_tools/ecology.py::make_capacity_map`
- `Cognitive_tools/ecology.py::smooth_field`
- `Cognitive_tools/ecology.py::rescale_pattern`
- `Cognitive_tools/ecology.py::gaussian_hotspot`

These functions were migrated during the ecological-unification
refactor from this repository's former:

    ecology_patch_scan.py

at repository commit:

    501cb693bc8a48cb957ea14db9212c2f4e0e453a

No external source code was copied for these landscape generators.

---

# Commit-level provenance

## refactor: unify ecological dynamics

Purpose:

- eliminate duplicate ecological simulators;
- establish one renewable-resource update function;
- move landscape construction into the package;
- preserve all existing numerical dynamics.

Deleted obsolete ecology experiment implementations:

- `ecology_run.py`
- `ecology_experiment.py`
- `ecology_patch_scan.py`

The historical versions remain available through Git history.