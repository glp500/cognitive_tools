# Scientific and Code Provenance

This document records where mechanisms implemented in `cognitive_tools`
originate, how the project modifies them, and whether external source
code was reused.

The purpose is to distinguish scientific lineage from implementation
lineage.

## Provenance labels

**REPRODUCED** — the implemented mathematical/computational mechanism is
intended to follow a published source closely.

**ADAPTED** — a recognizable source mechanism is retained but changed for
the present model.

**INSPIRED** — a source motivates the modeling principle, but the
implementation differs substantially.

**ORIGINAL** — the mechanism or experimental construction was introduced
specifically for `cognitive_tools`.

---

# Renewable-resource ecology

**Status:** INSPIRED

**Implementation:**

- `Cognitive_tools/ecology.py::resource_step`
- `Cognitive_tools/model.py::EcoModel`

**Scientific sources:**

- Tilman, Plotkin, & Akcay (2020), *Evolutionary Games with Environmental
  Feedbacks*, Nature Communications 11, 915.
  `doi:10.1038/s41467-020-14531-6`
- Tu et al. (2025), *Balancing Resource and Strategy: Coevolution for
  Sustainable Common-Pool Resource Management*, Earth Systems and
  Environment 9, 1529-1542. `doi:10.1007/s41748-024-00489-8`

The project implements a spatial renewable resource with heterogeneous
carrying capacity, regeneration, equilibrium fraction, four-neighbour
coupling, extraction, and discrete-time renewal.

The project defines depletion so that an isolated unharvested tile has
approximate positive equilibrium:

```text
R*(x) = q(x) K(x)
```

No source code from Tilman et al. or Tu et al. was copied.

---

# Spatial landscape construction

**Status:** ORIGINAL / repository-local

**Implementation:**

- `Cognitive_tools/ecology.py::make_capacity_map`
- `Cognitive_tools/ecology.py::smooth_field`
- `Cognitive_tools/ecology.py::rescale_pattern`
- `Cognitive_tools/ecology.py::gaussian_hotspot`

These functions were migrated from the repository's former
`ecology_patch_scan.py` during ecological unification. No external source
code was copied for these landscape generators.

---

# Extraction actions, reward, and welfare

**Status:** ORIGINAL PROJECT REPRESENTATION

```text
LOW_EXTRACT  = 0
HIGH_EXTRACT = 1
```

Default requested harvest amounts are project calibration choices:

```text
LOW_EXTRACT  -> 0.002
HIGH_EXTRACT -> 0.020
```

Historical `COOPERATE` / `DEFECT` aliases remain for compatibility.

Q-learning reward is realized individual harvest. Resource
sustainability, welfare, wealth inequality, and social/network outcomes
are measured separately rather than inserted directly into reward.

The project also separately records cumulative wealth and bounded
reserve/need-satisfaction welfare measures.

---

# PettingZoo environment interface

**Status:** ORIGINAL INTERFACE / THIRD-PARTY SOFTWARE

**Implementation:** `Cognitive_tools/env.py`

PettingZoo and Gymnasium are software dependencies rather than
scientific mechanism sources.

---

# Tabular Q-learning

**Status:** REPRODUCED ALGORITHM / INDEPENDENT IMPLEMENTATION

**Implementation:** `Cognitive_tools/qlearning.py::QLearningPolicy`

**Scientific source:** Watkins & Dayan (1992), *Q-learning*, Machine
Learning 8, 279-292. `doi:10.1007/BF00992698`

The implementation uses the standard one-step Q-learning update and is
independently written. The learner is state/action-count agnostic.

Current dimensions are:

```text
B0: 3 states x 2 actions
social treatments: 9 states x 2 actions
```

Small random Q-value initialization and the ecological-state
representation are project choices rather than parts of the original
Q-learning algorithm.

---

# Ecological state encoding

**Status:** ORIGINAL PROJECT ABSTRACTION

Local `R/K` is discretized as:

```text
scarce:    R/K < 1/3
moderate:  1/3 <= R/K < 2/3
abundant:  R/K >= 2/3
```

---

# Restricted social observation

**Status:** INSPIRED

**Implementation:**

- `Cognitive_tools/social.py::observed_low_fraction`
- `Cognitive_tools/social.py::social_observations`
- `Cognitive_tools/social.py::social_bin`
- `Cognitive_tools/social.py::joint_state`

**Scientific source:** Schrama, Tilman, & Vasconcelos (2025), *Majority
Illusion Drives the Spontaneous Emergence of Alternative States in
Common-Pool Resource Games with Network-Based Information*, iScience
28(7), 112831. `doi:10.1016/j.isci.2025.112831`

The project adopts the principle that agents see behavior through a
restricted social-information network.

The current implementation does not reproduce Schrama et al.'s full
Heuristics Switching Model, asynchronous strategy update, payoff model,
or social-memory process.

The project instead uses previous source actions as a compact social
signal and combines three social bins with three ecological bins.

No Schrama source code was copied.

---

# S1 random fixed-attention network

**Status:** ORIGINAL PROJECT INITIALIZATION / STRUCTURALLY RELATED TO OH &
SCHAUF

**Implementation:** `Cognitive_tools/social.py::init_random_attention`

Every observer receives exactly `k` distinct sources and cannot observe
itself.

The representation is directed:

```text
source -> observer
```

The fixed-attention architecture is structurally close to the limited
information-source design used by Oh & Schauf, but initialization here is
independently implemented.

---

# S2 fixed Barabasi-Albert-style observational network

**Status:** ADAPTED / INSPIRED

**Implementation:**
`Cognitive_tools/social.py::init_barabasi_albert_attention`

**Scientific source:** Schrama et al. (2025).

Schrama et al. use a Barabasi-Albert social network so direct-neighbour
information access has a skewed degree distribution and highly visible
nodes.

The `cognitive_tools` S2 treatment independently implements a compact
preferential-attachment network:

1. begin with a complete graph on `m + 1` agents;
2. add agents sequentially;
3. connect each newcomer to `m` distinct existing agents with probability
   proportional to current degree;
4. represent every undirected tie as symmetric observation lists.

The default project control uses:

```text
m = 2
```

because its finite-N mean degree is close to four, making it a useful
comparison with S1 `k=4` while introducing strong degree heterogeneity.

This is **not** a reproduction of Schrama et al.'s numerical network
density. In particular, their published default parameter table uses a
much larger BA link-density parameter for their `P=1000` model.

No code from Schrama et al. or their replication repository was copied.

---

# Local/global replacement search

**Status:** ADAPTED

**Implementation:**

- `Cognitive_tools/social.py::local_candidates`
- `Cognitive_tools/social.py::global_candidates`
- `Cognitive_tools/social.py::replace_source`
- `Cognitive_tools/social.py::rewire_epoch`

**Scientific source:** Oh & Schauf (2025), *Self-organizing Group
Structure through Rewiring for Collective Decision-Making in Evolving
Environments*, Scientific Reports 15, 39947.
`doi:10.1038/s41598-025-23634-3`

The project retains these structural principles:

```text
fixed attention capacity
one-source replacement
local sources-of-sources search
global search
theta controls local/global search
```

The implementation adds explicit global fallback when local search has
no legal candidate and uses a pre-checkpoint network snapshot for
candidate construction.

No Oh & Schauf source code was copied.

---

# Local social forecasting

**Status:** ADAPTED IDEA / PROJECT IMPLEMENTATION

**Implementation:** `Cognitive_tools/social.py::update_forecasts`

Each observer maintains one EWMA forecast:

```text
forecast_i(t+1)
=
(1-alpha) forecast_i(t)
+ alpha observed_i(t)
```

This is related to adaptive-expectation ideas in the source literature
but is not a reproduction of Schrama et al.'s HSM.

---

# Prediction-error rewiring

**Status:** ORIGINAL

**Implementation:**

- `Cognitive_tools/social.py::prediction_errors`
- `Cognitive_tools/social.py::rewire_epoch(mode="prediction_error")`

```text
error_i(t) = |observed_i(t) - forecast_i(t)|
```

An observer is eligible when:

```text
error_i(t) > threshold
```

and then rewires with probability `mu`.

This trigger is not Oh & Schauf's objective-correctness rule. CPR
low/high extraction actions do not receive an externally revealed binary
correctness label in this model.

Prediction error therefore measures local social surprise, not truth,
misinformation, deception, competence, or sustainability.

---

# Random rewiring controls

## Probabilistic random rewiring

**Status:** ORIGINAL EXPLORATORY CONTROL

**Implementation:** `rewire_epoch(mode="random")`

Every observer is eligible and independently rewires with probability
`mu`.

This mode is retained for exploratory/backward-compatible runs, but it
does not guarantee the same realized amount of topology change as an
adaptive treatment.

## Event-count-matched random rewiring

**Status:** ORIGINAL CONFIRMATORY CONTROL

**Implementation:** `rewire_epoch(mode="random_matched")` plus the
canonical experiment runner's rewiring-schedule loader.

A paired adaptive run first records the number of successful rewires at
each checkpoint. The matched random control then:

1. uses the same initial random-`k` graph construction and seed;
2. uses the same ecological condition, population, replicate, and
   rewiring checkpoint times;
3. uses the same `theta` search scope;
4. randomly chooses exactly the adaptive run's realized number of
   observers at each checkpoint;
5. gives each selected observer one source replacement.

This isolates adaptive source selection from the amount of network
turnover more cleanly than equal-`mu` random rewiring.

`mu` does not determine the event count in `random_matched` mode.

---

# Temporal ordering

**Status:** ORIGINAL INTEGRATION DESIGN

```text
(G_t, a_(t-1), R_t)
    -> s_t
    -> a_t
    -> R_(t+1)
    -> observe a_t through G_t
    -> prediction error
    -> optional rewiring
    -> G_(t+1)
    -> forecast update from pre-rewire observations
    -> s_(t+1)
    -> Q update
```

The Q target therefore uses the post-rewiring network.

---

# Social diagnostics

**Status:** PROJECT OPERATIONALIZATION

Current diagnostics include:

```text
visibility-degree Gini
maximum visibility share
zero-visibility fraction
reciprocity
visibility-degree assortativity
population low-extraction fraction
edge-weighted visible low-extraction fraction
visible-minus-population low-extraction bias
mean local perception error
signed perception bias
majority mismatch among non-tied comparisons
majority tie rate
degree/action correlation
prediction error
network turnover
rewire counts
requested/used search scope
local-search fallback counts
```

The majority mismatch denominator excludes observations where either the
local sample or the population-excluding-focal comparison is exactly tied.
The project now reports the excluded share directly as
`majority_tie_rate`.

`reciprocity` is the share of directed information edges whose reverse edge
also exists. `degree_assortativity` is the Pearson correlation between
source and observer visibility degree along directed information edges.
For symmetric BA-style graphs this reduces to ordinary degree
assortativity.

`visible_population_bias` is the edge-weighted visible low-extraction
fraction minus the population low-extraction fraction.

Material inequality (`wealth_gini`) and information inequality
(`visibility_gini`) are kept distinct.

The experiment runner also records one normalized per-agent social summary
and exact initial/final directed edge snapshots. These outputs are project
measurement infrastructure rather than mechanisms that alter behavior.

Policy heterogeneity retains the original equal-state Hamming measure and
adds `policy_hamming_visit_weighted_mean`, which weights disagreements by
aggregate training state occupancy. This prevents never-visited Q states
from contributing the same weight as states actually encountered during
learning.

---

# Network evaluation decomposition

**Status:** ORIGINAL EXPERIMENTAL DESIGN

The runner now distinguishes state retained in the Q-tables from state
retained in the social topology.

Default `--network-eval frozen` evaluation contains:

```text
continuation:
    trained Q
    training-end ecology
    terminal training graph
    frozen graph

fresh_reset:
    trained Q
    fresh ecology
    terminal training graph carried forward
    frozen graph

fresh_reset_network:
    trained Q
    fresh ecology
    exact initial training graph restored
    frozen graph
```

`fresh_reset` retains its historical name so the existing baseline figure
script continues to select the same primary fresh-ecology condition. New
rows also record `network_start` explicitly.

The initial graph is stored as an independent snapshot before any training
rewiring occurs; it is not regenerated after training. This guarantees that
the reset-network evaluation uses the exact graph instance from the start
of the corresponding training run.

For prediction-error treatments, optional `--network-eval adaptive` adds:

```text
fresh_adaptive_network:
    trained Q, frozen
    fresh ecology
    terminal training graph
    carried terminal forecast state
    continued network adaptation
```

Adaptive evaluation uses a distinct RNG stream and copies the terminal
graph/forecast state before evaluation. It therefore cannot alter the
training endpoint subsequently used by other evaluation conditions.

Matched-random R0 does not currently support adaptive evaluation because
the project does not yet define a matched evaluation-phase event-count
schedule. R0 confirmatory evaluation remains frozen.

---

# Reproducibility infrastructure

**Status:** ORIGINAL

The canonical runner separates random streams for:

```text
landscape
agent positions
Q-learning
initial social graph
rewiring
evaluation random policy
```

It records in `config.json`:

```text
CLI parameters
scenario definitions
network/evaluation semantics
UTC timestamp
Git commit and branch
Git worktree status
Python/platform information
major package versions
invocation command
```

Active rewiring runs additionally write:

```text
data/rewiring_schedule.csv
```

so matched controls can reproduce the adaptive run's realized rewiring
frequency checkpoint by checkpoint.

Social-measurement hardening additionally writes:

```text
data/agent_social_summary.csv
data/network_edges_checkpoints.csv
```

The edge checkpoint file contains only exact initial and terminal training
graphs rather than every edge at every timestep.

Generated experiment directories are ignored by Git for new runs and
should be frozen later as deliberate research-release artifacts.

---

# External code reuse status

No scientific mechanism source code from the papers listed in
`REFERENCES.bib` has been copied into this repository.

Published work supplies scientific/mechanistic provenance. Current
implementations are independently written.

If direct external code reuse occurs later, record:

```text
source repository
source commit
source path/function
license
specific modifications
```

both in code comments and here.

---

# Commit-level provenance

## `refactor: unify ecological dynamics`

Commit: `b31478295ce7987db72066b592278995e5d297eb`

Centralized ecology and removed duplicate ecological simulators.

## `generic tabular q learner`

Commit: `b8baef4bf041982d10ada37830c187cf6cec821f`

Made the learner state/action-count agnostic while preserving the
3-state baseline.

## scientific provenance documentation

Commit: `298bdbea483b336eb1b8249518f713a7101d2b88`

Established source/code provenance documentation and bibliography.

## `fixed social observation`

Commit: `6e7d7cc9fb89a641e77fd116c646ba7988ee2aad`

Added directed fixed-attention observation and the 9-state
 ecology-social learner.

## `feat: decentralized rewiring`

Commit: `286cb3a5c2c8ae6929dc0ec0109830655012f4a4`

Added local/global source replacement, EWMA prediction, prediction-error
rewiring, random turnover, and social-network diagnostics.

## `chore: experiment hardening`

Commit: `8703f07b84709bfc206ca4539a30236408e2d169`

Added generated-result ignore rules and run-level Git/runtime metadata.
The remaining planned hardening documentation and lifecycle tests were
completed in the subsequent core-social-controls changeset.

## `feat: complete core social controls`

Commit: `be78a42845f75498a6da5e4a42e8399b85d60a8f`

Purpose:

```text
add fixed BA-style S2 network
add matched-random R0
write per-checkpoint rewiring schedules
validate matched theta and run keys
add control-specific tests
repair missing hardening documentation/lifecycle tests
```

## `feat: decompose network evaluation`

Commit: `f159c0ab16f0d0a76e922105d85e73ea0839ceed`

Purpose:

```text
preserve exact initial and terminal social graphs
separate fresh carried-network and reset-network evaluation
retain frozen-network evaluation as the default
add optional adaptive-network robustness evaluation
keep Q-tables frozen during evaluation
```

## Stage 4 social-measurement changeset

Purpose:

```text
report majority tie rate explicitly
add reciprocity and visibility-degree assortativity
add edge-weighted visible-versus-population bias
add per-agent social summary output
add initial/final network edge checkpoints
add visit-weighted policy heterogeneity
```

---

# Current scientific boundary

The repository currently targets:

```text
spatial renewable ecology
+
independent tabular learning
+
network-limited previous-action observation
+
fixed random or fixed BA-style information topology
+
random or prediction-error-driven decentralized rewiring
```

It does not currently implement:

```text
full Schrama HSM
full Oh-Schauf DeGroot model
misinformation
homophily
prestige
payoff-based rewiring
movement
sanctioning
transfers
variable-k adaptive attention
GNN planning
```

See `REFERENCES.bib` for bibliographic records.
