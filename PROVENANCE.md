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
    + r(x) R(x,t) [1 - R(x,t)/K(x)]
    - d(x) R(x,t)
    + c [mean_neighbor(R) - R(x,t)]

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

# Independent tabular Q-learning

## Q-learning algorithm

**Status:** REPRODUCED ALGORITHM / PROJECT IMPLEMENTATION

**Implementation:**

- `Cognitive_tools/qlearning.py::QLearningPolicy`

**Scientific source:**

Watkins, C. J. C. H., & Dayan, P. (1992).
"Q-learning."
Machine Learning, 8, 279-292.
https://doi.org/10.1007/BF00992698

**Source principle:**

The learner applies the standard one-step off-policy Q-learning update:

    Q(s, a)
    <-
    Q(s, a)
    + alpha [
        r
        + gamma max_a' Q(s', a')
        - Q(s, a)
    ]

For a terminal transition, the target is:

    target = r

rather than bootstrapping from the next state.

**cognitive_tools implementation:**

The implementation is an independent minimal tabular implementation.

No source code from Watkins & Dayan is copied.

The learner is deliberately domain-agnostic:

- `n_states` determines the number of discrete states;
- `n_actions` determines the number of discrete actions;
- states are supplied as integer indices;
- actions are returned as integer indices;
- rewards are supplied as scalar values;
- the learner does not know what a state or action means.

The current ecological baseline uses:

    n_states = 3
    n_actions = 2

The planned ecological-social model can use:

    n_states = 9
    n_actions = 2

without modifying the Q-learning algorithm.

The learner uses epsilon-greedy exploration with multiplicative epsilon
decay.

Small random Q-value initialization is retained from the previous
cognitive_tools implementation to avoid deterministic action-0
tie-breaking.

For the default three-state, two-action case, the initialization uses
the same NumPy RNG call and table dimensions as the pre-refactor
implementation.

---

## Ecological state encoding

**Status:** ORIGINAL / project abstraction

**Implementation:**

- `Cognitive_tools/qlearning.py::resource_state_from_fraction`
- `Cognitive_tools/qlearning.py::resource_state`

The ecological learner state is determined from local:

    R / K

using three equal-width bins:

    state 0: scarce
    R/K < 1/3

    state 1: moderate
    1/3 <= R/K < 2/3

    state 2: abundant
    R/K >= 2/3

This discretization is a cognitive_tools modeling choice.

It is not part of the Q-learning algorithm described by Watkins &
Dayan.

The current `resource_state()` function is a compatibility adapter for
the present `EcoEnv` observation structure. It extracts local R/K from:

    observation["local"][0]

and delegates the discretization to:

    resource_state_from_fraction()

This separation is intended to allow the observation representation to
change later without altering `QLearningPolicy`.

---

# Action terminology

The physical model currently has two extraction actions:

    LOW_EXTRACT = 0
    HIGH_EXTRACT = 1

Some existing experiment scripts still use the historical aliases:

    COOPERATE = 0
    DEFECT = 1

Those aliases are retained temporarily for backward compatibility.

They should not be interpreted as implying that low extraction is always
equivalent to social cooperation or that high extraction is always
equivalent to social defection.

The direct modeled variable is extraction intensity.

A later cleanup will remove the historical aliases after the legacy
experiment runner has been retired.

---

# Commit-level provenance

## refactor: unify ecological dynamics

Purpose:

- eliminate duplicate ecological simulators;
- establish one renewable-resource update function;
- move landscape construction into the package;
- preserve existing numerical ecological dynamics.

Deleted obsolete ecology experiment implementations:

- `ecology_run.py`
- `ecology_experiment.py`
- `ecology_patch_scan.py`

The historical versions remain available through Git history.

---

## refactor: make tabular q learner generic

Purpose:

- remove hard-coded three-state assumptions from the Q-learning class;
- remove hard-coded two-action assumptions from action selection;
- parameterize Q-table dimensions with `n_states` and `n_actions`;
- preserve the existing three-state/two-action baseline by default;
- separate ecological state interpretation from the Q-learning algorithm;
- establish support for the planned nine-state ecological-social model;
- add direct unit tests of the Q-learning update;
- preserve legacy action aliases temporarily so this refactor does not
  alter unrelated experiment code.