# Stage-5 visibility and bounded-search study

Status: implemented historical protocol; 30 September 2026. The
[capacity-matched visibility study](capacity-matched-visibility-study.md) is the
prospective primary environment design. The earlier Stage-4 design and results remain archived for
reproduction and comparison. The [capped-harvest validation](../reviews/capped-harvest-validation-results-2026-09-30.md)
qualifies the ecology as a population commons dilemma for the tested policies.

## Problem and questions

Ecology and limited social observation may cause agents' local views to differ
from population behavior, and may change visibility concentration and collective
extraction. The experiment manipulates *initial propensity to receive attention*
and fixed versus adaptive attention. Each observer attends exactly four sources.

- **RQ1, perception:** Under which ecological and social-observation conditions
  do local observations misrepresent population behavior?
- **RQ2, organization:** How do ecology and social-observation conditions shape
  visibility concentration and population extraction behavior?

Resource persistence, reserve welfare and wealth inequality are secondary
consequences. No part of the social layer transfers rewards, energy or Q values.

**H1, ecological dependence:** Population misperception differs among uniform
high, patchy high and split high/low ecological settings, holding social factors
fixed. This is nondirectional.

**H2, initial visibility:** Population misperception differs among initial
visibility profiles under fixed networks. This is nondirectional. The planned
comparisons are random−equal, normal-centered−random, low-majority−random and
high-majority−random.

**H3, adaptive observation:** Adaptive bounded search changes population
misperception and visibility Gini relative to the same profile on a fixed graph.
This is nondirectional and identifies the adaptive mechanism as a package.
Low-extraction share is a companion behavior outcome.

## Frozen factors and mechanism

Each of the three ecologies has five profiles × two network dynamics plus B0,
for 33 conditions total. At 100 independent replicates there are 3,300
condition-replicates. B0 has no social perception or visibility values.

The five profiles, distribution parameters and network methods live in
[`visibility_profiles_v1.json`](../../configs/visibility/visibility_profiles_v1.json).
The `equal` network has exactly four observers per source and four sources per
observer. `random` has equal propensity 0.5 but stochastic realized visibility.
`normal_centered` samples a normal distribution truncated to (0,1) with mean
0.5 and SD 0.15. `low_propensity_majority` uses Beta(2,5), and
`high_propensity_majority` uses Beta(5,2). Weighted profiles sample four sources
without replacement, excluding self, proportional to source weights. Propensity
is an **initial source-selection weight**, not an individual's realized
visibility probability. It is used once and never consulted during rewiring.

Realized visibility for source j is its number of observers divided by N−1. At
N=64, mean realized visibility is 4/63. Report both raw propensity and realized
observer count; they answer different questions. The same profile, population
and replicate seed yields the identical propensity vector and graph across all
ecologies and both network dynamics.

Fixed networks do not rewire. Adaptive bounded networks check every 50 steps:
observers with absolute prediction error above 0.25 attempt rewiring with
probability 0.10. Each successful event replaces one source, drawing from local
two-hop candidates with probability 0.75 or all valid population candidates with
probability 0.25, with global fallback if the local set is empty. Selection is
uniform among valid candidates. The global-search probability theta=0.25 is a
fixed assumption, never a Stage-5 treatment axis. The old BA and matched R0
mechanisms remain available only in Stage-4 tools.

Agents optimize the already validated **capped-harvest** reward with cap equal
to metabolism 0.002. The primary evaluation is a fresh ecology with the
terminal network frozen. All learning treatments retrain under this reward.

## Measurement and provenance

`stage5_visibility_v1` stores the profile, dynamics, raw propensity, realized
initial visibility and exact graph/propensity SHA-256 identifiers. Each run's
configuration stores protocol ID, profile spec hash, reward definition,
attention capacity, propensity semantics and rewiring parameters. The campaign
freeze stores code revision, seed, grid and spec hash. Resume rejects changes to
scientific configuration or source revision. The analysis rejects mixed
protocols, reward definitions and profile-spec hashes and checks that fixed and
adaptive paired runs begin from the same graph and propensity realization.

[`initial_visibility.csv`](../../cognitive_tools/experiment.py) has one row per
agent per replicate. Training records perception error, majority mismatch among
non-ties, tie rate, visibility Gini/variance/skewness/quantiles, zero-visible
fraction, low-extraction share, edge turnover and rewiring details. Network
snapshots at step zero and training end yield initial Gini G0, terminal Gini GT
and GT−G0. Evaluation retains separate utility and physical harvest accounts. Secondary
resource and welfare comparisons use fresh-evaluation means; wealth inequality
uses final Gini after fresh evaluation.
Undefined majority mismatch remains missing, with tie rate and valid-window
counts visible. Agent-level distributions are descriptive; inferential intervals
use independent replicates.

The final 1,000 training steps form the primary window; for shorter smoke runs
all recorded steps are used. Estimates average recorded checkpoints in that
window. The primary contrast tables pair results by replicate. H1 uses
patchy−uniform and split−uniform within profile and dynamics. H2 uses the four
fixed-network profile contrasts above. H3 uses adaptive−fixed within ecology and
profile. Pointwise 95% replicate-bootstrap intervals use 5,000 resamples in
the full campaign. They are not familywise significance tests and no significance
stars are applied. Figures and tables are descriptive until the complete grid
and inference protocol are frozen.

## Figure contract

1. Mechanism: actual ecological maps, propensity→graph→realized visibility,
   fixed/adaptive branches and outcomes.
2. Manipulation: agent-level initial observer-count distributions, replicate
   intervals for initial Gini and invisible fraction; no agent-level inference.
3. Perception: error, non-tie mismatch and tie rate by ecology, profile and
   dynamics.
4. Organization: final-window visibility Gini and low-extraction share.
   A supplement shows G0, GT and GT−G0.
5. Consequences: paired Adaptive−Fixed differences in fresh-evaluation resource
   fraction, mean reserve welfare and final wealth Gini, each with zero line.
   B0 absolute outcomes remain in the outcome summary table.

Ecology facets share y-axis limits within each metric row. Two separate
trajectory supplements show visibility Gini and low-extraction share in
profile×ecology panels. Every plotted estimate is read from a
saved replicate or summary table. The original Stage-4 focused figures retain
their historical meaning.

## Calibration and phase gates

Network-only calibration at N=64, k=4 sampled 1,000 networks per profile,
without learning or examining scientific outcomes. Its raw data and manifest
are in `results/visibility/calibration_v1_final/` (generated local artifacts).
At 1,000 networks per profile the observed means were:

| Profile | Mean propensity | Initial Gini | Invisible fraction | Visibility variance |
|---|---:|---:|---:|---:|
| Equal | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| Random | 0.5000 | 0.2652 | 0.0154 | 3.6981 |
| Normal centered | 0.5000 | 0.3099 | 0.0333 | 5.0525 |
| Low majority | 0.2857 | 0.3959 | 0.0841 | 8.3905 |
| High majority | 0.7143 | 0.2916 | 0.0260 | 4.4654 |

Equal gives exact Gini zero. The random, normal, low-majority and high-majority profiles differ in
propensity shape; their realized Gini distributions can overlap, especially
normal and high-majority. Interpret profile contrasts by their actual
manipulation, not their names alone. Changing parameters requires a new spec
version and new held-out study identity.

The mechanical smoke tests all five profiles and both dynamics at N=8, two
replicates, 60 training steps and six evaluation steps. The scientific pilot
uses all three ecologies, N=64, ten replicates, 5,000 training and 1,000
evaluation steps. Inspect runtime, missingness, manipulation separation and
paired graph identities before the full campaign. The full campaign fixes
N=64, k=4, 5,000/1,000 steps, 100 replicates, 5,000 bootstrap resamples, new
seed 20261013 and the code/spec hashes. Run names must be new. Do not use pilot
outcomes to tune visibility profiles.

## Observability questions and signals

1. **Was the intended observation environment created?**
   `initial_visibility.csv`, graph/propensity hashes, calibration tables and
   initial Gini/counts answer this for each replicate.
2. **Did adaptive bounded search operate as specified?**
   `network_timeseries.csv`, rewiring schedule, requested scope/fallback/candidate
   counts, turnover and run-health event totals expose the mechanism.
3. **Are perception and collective outcomes changing, and where?**
   Final-window replicate tables, planned paired contrasts, valid-window counts,
   evaluation utility and the five figures answer this without treating agents
   as independent replicates.
4. **Did a run finish completely and reproducibly?**
   JSON condition events carry a run/scenario/population/replicate correlation ID
   and duration; `run_health.json`, `complete.json`, configuration hashes and
   campaign freeze identify completed work and failed invariants.

These are local batch simulations; HTTP RED metrics, distributed spans and
pager alerts have no relevant endpoint or service. All telemetry fields are
allowlisted run/scenario/profile/count data and contain no private user input.

## Commands

See the [README](../../README.md) for exact smoke, pilot and full campaign
commands and the corresponding analysis outputs. The full campaign is a
separate long-running computation and has not been executed by implementation.
