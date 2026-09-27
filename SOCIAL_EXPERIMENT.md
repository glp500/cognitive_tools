# Social Information and Decentralized Rewiring Experiment

## Status

This document specifies the social-network experiment implemented in
`cognitive_tools`.

Completed core mechanisms:

```text
B0 ecological baseline
S1 fixed random-k social observation
S2 fixed BA-style visibility-skew control
R0 event-count-matched random turnover
R1-R3 prediction-error rewiring
```

The core evaluation decomposition is now implemented. Cross-treatment
social figures and additional measurements still need to be completed
before the final confirmatory experiment campaign.

---

# 1. Research questions

Primary question:

> Can agents that locally predict the behavior of their information
> sources use prediction error to adapt whom they observe, and can that
> decentralized rewiring improve or destabilize common-pool-resource
> sustainability?

Ecological interaction question:

> How do those effects depend on ecological conditions?

The experiment decomposes the mechanism as:

```text
network structure
    -> locally observed behavior
    -> perception / prediction
    -> learned extraction behavior
    -> ecological outcome
    -> welfare and material inequality
```

When rewiring is active, network structure becomes endogenous.

---

# 2. Physical/social separation

The ecological environment owns:

```text
resource dynamics
extraction
realized harvest
reward
wealth
welfare
```

The social layer owns:

```text
attention topology
peer-action observation
social-state construction
forecasting
prediction error
rewiring
network/perception diagnostics
```

Social edges do not alter ecological resource flow directly.

---

# 3. Actions and reward

```text
LOW_EXTRACT  = 0
HIGH_EXTRACT = 1
```

Default requested extraction:

```text
low  = 0.002
high = 0.020
```

Q-learning reward is realized individual harvest.

Sustainability and equality are measured outcomes rather than direct
reward components.

---

# 4. Ecological state

Local resource state uses `R/K`:

```text
0 scarce:    R/K < 1/3
1 moderate:  1/3 <= R/K < 2/3
2 abundant:  R/K >= 2/3
```

B0 therefore has three learner states.

---

# 5. Social state

For observer `i`, let:

```text
o_i(t) = fraction of current sources that chose LOW_EXTRACT
```

The action state at time `t` uses source actions from `t-1`.

The signal is discretized as:

```text
0 mostly_high: o < 1/3
1 mixed:       1/3 <= o < 2/3
2 mostly_low:  o >= 2/3
```

Before previous actions exist:

```text
o_i = 0.5
```

The joint state is:

```text
joint_state = 3 * ecological_state + social_state
```

Social treatments therefore use nine learner states.

---

# 6. Social-network representation

The canonical Python representation is:

```text
sources[observer] = information sources observed by that agent
```

Information direction is:

```text
source -> observer
```

Visibility degree of source `j` is the number of observer lists
containing `j`.

---

# 7. S1 random fixed-k network

CLI:

```bash
--social-mode fixed \
--social-network random_k \
--social-k 4 \
--rewiring none
```

Every observer receives exactly `k` distinct sources.

Properties:

```text
no self-links
no duplicate sources
attention capacity = k for every observer
total directed edges = N * k
```

This is the clean restricted-information control and the initial
condition for all rewiring treatments.

---

# 8. S2 fixed BA-style network

CLI:

```bash
--social-mode fixed \
--social-network ba \
--ba-m 2 \
--rewiring none
```

Construction:

```text
initial clique of m+1 nodes
sequential node addition
m distinct targets per newcomer
target probability proportional to current degree
```

The resulting graph is undirected and is stored as symmetric observation
lists.

Consequences:

```text
attention capacity varies
visibility varies
high-degree hubs exist
network remains fixed
```

The purpose of S2 is to compare:

```text
S1 random-k
vs
S2 skewed fixed visibility
```

without adding adaptive rewiring.

The default `m=2` gives a finite-N mean degree close to four, which makes
it a useful comparison with S1 `k=4` while preserving strong degree
heterogeneity.

This numerical `m` is a project control choice. It is not a reproduction
of the much denser default network in Schrama et al.

---

# 9. Forecast and prediction error

Every social observer starts with:

```text
forecast_i(0) = 0.5
```

Before forecast updating:

```text
error_i(t)
=
|observed_i(t) - forecast_i(t)|
```

Then:

```text
forecast_i(t+1)
=
(1-alpha_f) forecast_i(t)
+ alpha_f observed_i(t)
```

Default:

```text
alpha_f = 0.50
```

Prediction error is local social surprise, not truth or misinformation.

---

# 10. Rewiring search

Rewiring is supported only for `random_k` networks in the current core
experiment.

A successful event:

```text
drop exactly one current source
add exactly one legal replacement
preserve k
```

Search scope:

```text
probability global = theta
probability local  = 1 - theta
```

Local candidates are sources of current sources.

Global candidates are all non-self, not-already-followed agents.

If local search has no legal candidate, the implementation falls back to
global search.

All candidate sets at one checkpoint are built from the network snapshot
at the beginning of that checkpoint.

---

# 11. Adaptive rewiring

CLI:

```bash
--rewiring prediction_error
```

An observer is eligible when:

```text
error_i > rewire_threshold
```

and then rewires with probability:

```text
rewire_mu
```

Default values:

```text
rewire_threshold = 0.25
rewire_mu        = 0.10
rewire_every     = 50
```

Adaptive treatment labels:

```text
R1: theta = 0.00
R2: theta = 0.25
R3: theta = 1.00
```

---

# 12. Rewiring schedule output

Every active-rewiring run writes:

```text
data/rewiring_schedule.csv
```

with one row per rewiring checkpoint and condition.

Required columns:

```text
scenario
population
replicate
time
rewiring
rewire_theta
rewire_every
base_seed
social_network
social_k
training_steps
target_rewires
successful_rewires
```

For adaptive runs, `target_rewires` is blank and
`successful_rewires` is the realized count.

For matched-random runs, `target_rewires` is the imported adaptive count
and `successful_rewires` must equal it.

---

# 13. R0 matched-random control

The confirmatory random-turnover control is:

```bash
--rewiring random_matched
```

and requires:

```bash
--matched-rewire-schedule <adaptive schedule path>
```

At every checkpoint the runner:

1. looks up the adaptive run using `(scenario, population, replicate,
   time)`;
2. verifies that the source row came from `prediction_error` rewiring;
3. verifies that source `theta` equals current `theta`;
4. reads the adaptive run's `successful_rewires`;
5. samples exactly that many legally rewritable observers uniformly
   without replacement;
6. gives each selected observer one random source replacement using the
   same local/global search rule.

Thus the main matched comparison is:

```text
same ecology
same population
same replicate
same initial random-k graph seed
same checkpoint times
same theta
same number of successful rewire events per checkpoint
--------------------------------------------------------
different source-selection trigger
```

`rewire_mu` does not determine the number of events in
`random_matched` mode.

The older:

```bash
--rewiring random
```

mode remains available for exploratory/backward-compatible runs but is
not the preferred confirmatory R0 control.

---

# 14. Required adaptive/R0 run ordering

A matched R0 depends on an adaptive schedule, so run the adaptive member
first.

Example R2 source:

```bash
python baseline_validation_experiment.py \
    --run-name r2_core \
    --social-mode fixed \
    --social-network random_k \
    --social-k 4 \
    --rewiring prediction_error \
    --rewire-theta 0.25 \
    --rewire-mu 0.10 \
    --rewire-every 50 \
    --rewire-threshold 0.25 \
    --forecast-alpha 0.50 \
    --scenarios uniform_high patchy_high \
    --populations 64 \
    --replicates 10 \
    --training-steps 5000 \
    --evaluation-steps 1000
```

Then its paired R0:

```bash
python baseline_validation_experiment.py \
    --run-name r0_matched_r2_core \
    --social-mode fixed \
    --social-network random_k \
    --social-k 4 \
    --rewiring random_matched \
    --rewire-theta 0.25 \
    --matched-rewire-schedule \
        results/q_learning_baseline/experiments/r2_core/data/rewiring_schedule.csv \
    --scenarios uniform_high patchy_high \
    --populations 64 \
    --replicates 10 \
    --training-steps 5000 \
    --evaluation-steps 1000
```

The two runs must use the same base seed, scenarios, populations,
replicates, training length, rewiring interval, and `theta`.

Repeat the same pairing for `theta=0` and `theta=1`.

---

# 15. Temporal ordering

Training uses:

```text
(G_t, a_(t-1), R_t)
        -> s_t
        -> a_t
        -> R_(t+1)
        -> observe a_t through G_t
        -> prediction error
        -> optional rewiring
        -> G_(t+1)
        -> forecast update using observation from G_t
        -> s_(t+1)
        -> Q update
```

The Q target uses the post-rewiring network.

---

# 16. Core treatment matrix

| ID | Social signal | Initial network | Rewiring | Search |
|---|---|---|---|---|
| B0 | none | none | none | none |
| S1 | peer actions | random directed fixed-k | none | none |
| S2 | peer actions | fixed BA-style | none | none |
| R0(theta) | peer actions | random directed fixed-k | matched random | theta matched to paired adaptive run |
| R1 | peer actions | random directed fixed-k | prediction error | theta=0 |
| R2 | peer actions | random directed fixed-k | prediction error | theta=0.25 |
| R3 | peer actions | random directed fixed-k | prediction error | theta=1 |

---

# 17. Main causal contrasts

```text
S1 - B0
```

Restricted social observation relative to ecology-only learning.

```text
S2 - S1
```

Fixed skewed visibility relative to fixed random-k observation.

```text
R0(theta) - S1
```

Effect of dynamic topology with a random source-selection rule.

```text
R1 - matched R0(theta=0)
R2 - matched R0(theta=0.25)
R3 - matched R0(theta=1)
```

Effect of prediction-error-based source selection beyond the same
realized amount of rewiring.

```text
R1 vs R2 vs R3
```

Effect of replacement-search scope.

---

# 18. Current outcomes

Primary sustainability quantity:

```text
eval_mean_mean_resource_fraction
```

Other ecological/behavioral quantities:

```text
low_extraction_rate
collective_order
action_entropy
total_resource
```

Welfare/material quantities:

```text
mean_reserve_welfare
mean_need_satisfaction
deprivation_rate
mean_metabolic_shortfall
mean_energy
wealth_gini
mean_wealth
```

Mechanism quantities:

```text
visibility_gini
max_visibility_share
zero_visibility_fraction
reciprocity
degree_assortativity
population_low_fraction
visible_low_fraction
visible_population_bias
mean_perception_error
signed_perception_bias
majority_mismatch_rate
majority_tie_rate
degree_action_correlation
mean_prediction_error
edge_turnover
rewires_since_record
cumulative_rewires
global_rewire_fraction
requested_global_fraction
local_fallbacks
```

## Stage 4 social-measurement semantics

For focal observer `i`, the local observed fraction remains:

```text
o_i = fraction of current sources choosing LOW_EXTRACT
```

and the comparison population remains:

```text
c_-i = LOW_EXTRACT fraction among all other agents
```

The per-observer perception quantities are:

```text
absolute error = |o_i - c_-i|
signed bias    = o_i - c_-i
```

Majority comparison uses the sign of each fraction relative to `0.5`. A
comparison is marked tied if either the local sample or `c_-i` is exactly
`0.5`. The reported:

```text
majority_mismatch_rate
```

is calculated only over non-tied comparisons, while:

```text
majority_tie_rate
```

is the fraction of focal comparisons excluded for a tie.

`visible_population_bias` is:

```text
edge-weighted visible LOW fraction
- population LOW fraction
```

so it directly records whether visibility weighting over-represents low or
high extraction relative to the population.

Structural measurements are:

```text
reciprocity
    fraction of directed source -> observer edges whose reverse edge exists

degree_assortativity
    Pearson correlation between source visibility degree and observer
    visibility degree along directed information edges
```

For symmetric BA-style graphs, the latter reduces to ordinary degree
assortativity. Undefined correlations remain `NaN`.

Policy heterogeneity now includes:

```text
policy_hamming_mean
policy_hamming_visit_weighted_mean
visited_state_fraction
training_visit_fraction_<state>
```

The visit-weighted Hamming measure uses aggregate training state occupancy
as the weight over Q states. Disagreement in never-visited states therefore
receives zero weight.

The runner additionally writes:

```text
data/agent_social_summary.csv
data/network_edges_checkpoints.csv
```

`agent_social_summary.csv` contains one row per agent/run with training-time
mean observed behavior, comparison-population behavior, perception error,
signed bias, majority tie/mismatch rates, mean/final visibility, rewiring
count, prediction error, and final material/welfare quantities.

`network_edges_checkpoints.csv` contains exact initial and terminal training
edge lists in `source -> observer` orientation, with endpoint visibility
degree and attention size. It is intentionally a selected-checkpoint output
rather than a full edge-level timeseries.

---

# 19. Network evaluation semantics

The default is:

```bash
--network-eval frozen
```

It evaluates:

```text
continuation:
    Q-table: trained
    ecology: training endpoint
    graph: terminal training graph
    network dynamics: frozen

fresh_reset:
    Q-table: trained
    ecology: reset
    graph: terminal training graph carried forward
    network dynamics: frozen

fresh_reset_network:
    Q-table: trained
    ecology: reset
    graph: exact initial training graph restored
    network dynamics: frozen
```

The name `fresh_reset` is retained only for backward compatibility with
the existing baseline figure pipeline. Its meaning is now explicitly
recorded as:

```text
network_start = terminal
network_adaptive = false
```

The reset-topology condition records:

```text
network_start = initial
network_adaptive = false
```

The central Stage 3 comparison is therefore:

```text
fresh_reset
minus
fresh_reset_network
```

with the same trained Q tables and same fresh ecological reset. This
separates performance associated with the terminal learned topology from
performance already encoded in the Q tables.

For S1 and S2, no training rewiring occurs, so the terminal and initial
graphs are identical. Their two fresh evaluations are expected to agree;
this is also a lifecycle validation check.

## Optional adaptive-network robustness

For prediction-error runs, use:

```bash
--network-eval adaptive
```

This retains all frozen conditions and additionally evaluates:

```text
fresh_adaptive_network:
    Q-table: trained and frozen
    ecology: reset
    graph start: terminal training graph
    forecast start: terminal training forecast
    network dynamics: adaptive
```

The graph and forecast dictionaries are copied before the evaluation. A
separate evaluation-rewiring RNG stream is used, so this robustness run
does not alter the stored training endpoint or reuse the training rewiring
stream.

The adaptive option is not available for fixed S1/S2 or confirmatory
`random_matched` R0. For matched R0, an additional evaluation-phase
matching schedule would be required to preserve the event-count control.
The primary R0-vs-adaptive comparisons should therefore use
`--network-eval frozen`.

Evaluation rows add:

```text
network_start
network_adaptive
evaluation_total_rewires
```

and evaluation timeseries rows additionally add:

```text
evaluation_rewires_step
evaluation_rewires_cumulative
```

---

# 20. Reproducibility

The runner uses separate deterministic random streams for:

```text
landscape
agent positions
Q learners
initial social topology
rewiring
evaluation random policy
```

Initial `random_k` topology depends on base seed, replicate, and
population, not ecological scenario or rewiring treatment.

Every run records Git/runtime metadata in `config.json`.

For `random_matched`, the schedule file is an explicit experimental
input and should be archived together with the paired adaptive run.

---

# 21. Pilot guidance

Mechanical smoke testing can continue at `N=8`.

Scientific network pilots should emphasize:

```text
N = 32, 64
k = 4
BA m = 2
```

Suggested initial pilot:

```text
ecology:
    uniform_high
    patchy_high
    one strongly mixed scenario

N:
    32, 64

replicates:
    10

adaptive theta:
    0, 0.25, 1

mu:
    0.10

rewire interval:
    50
```

Do not move to the full 50-100 replicate confirmatory campaign until the
remaining Stage 5 social-figure pipeline has been validated on pilot outputs.

---

# 22. Remaining stages

Completed:

```text
Stage 3 — network evaluation decomposition
Stage 4 — complete social measurements
```

## Stage 5 — cross-treatment analysis and figures

At minimum:

```text
resource outcome distributions
joint state behavior heatmap
joint state occupancy heatmap
visibility-Gini trajectories
perception-error trajectories
degree-action correlation trajectories
wealth Gini vs visibility Gini
turnover vs resource outcome
paired adaptive-minus-R0 effects
theta x mu heatmaps
```

Run-level distributions and paired contrasts should be retained rather
than relying only on mean +/- SEM summaries.

---

# 23. Out of scope for the current core

Do not add these mechanisms before the core experiment is analyzed:

```text
payoff rewiring
homophily
prestige
misinformation
full HSM switching
full DeGroot consensus
movement
sanctions
transfers
variable adaptive k
GNN planning
```
