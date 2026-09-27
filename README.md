# cognitive_tools

`cognitive_tools` is a research codebase for studying how ecological
feedback, individual reinforcement learning, social information, and
decentralized adaptation interact in renewable common-pool-resource
systems.

The central social-network question is:

> Can agents that locally predict the behavior of their information
> sources use prediction error to adapt whom they observe, and can that
> decentralized rewiring improve or destabilize common-pool-resource
> sustainability?

The project keeps ecological dynamics and social information
mechanically separate.

## Current treatments

| ID | Social information | Initial social network | Network dynamics |
|---|---|---|---|
| `B0` | none | none | none |
| `S1` | previous peer actions | random directed fixed-`k` | fixed |
| `S2` | previous peer actions | fixed BA-style network | fixed |
| `R0` | previous peer actions | random directed fixed-`k` | event-count-matched random rewiring |
| `R1` | previous peer actions | random directed fixed-`k` | prediction-error rewiring, `theta=0` |
| `R2` | previous peer actions | random directed fixed-`k` | prediction-error rewiring, `theta=0.25` |
| `R3` | previous peer actions | random directed fixed-`k` | prediction-error rewiring, `theta=1` |

The older probabilistic `--rewiring random` mode is retained for
exploratory/backward-compatible runs. Confirmatory `R0` runs should use
`--rewiring random_matched` and a rewiring schedule produced by the
paired adaptive run.

## Architecture

```text
Cognitive_tools/
    ecology.py       renewable-resource dynamics and landscapes
    model.py         agents and physical resource use
    env.py           PettingZoo ecology interface
    qlearning.py     generic tabular Q learner
    social.py        social observation, topology, prediction, rewiring
    visualization.py

baseline_validation_experiment.py   canonical experiment runner
baseline_validation_figures.py      baseline figures
qlearning_experiment.py              legacy scenario definitions/helpers

BASELINE_EXPERIMENT.md
SOCIAL_EXPERIMENT.md
PROVENANCE.md
REFERENCES.bib

tests/
```

## Actions

```text
LOW_EXTRACT  = 0
HIGH_EXTRACT = 1
```

Historical `COOPERATE` / `DEFECT` aliases remain only for compatibility
with older code.

Default requested extraction amounts are:

```text
low  = 0.002
high = 0.020
```

Reward is realized individual harvest. Sustainability, welfare,
inequality, conformity, prediction accuracy, and network position are
outcomes rather than direct reward terms.

## Ecological learner state

The ecological state is based on local `R/K`:

```text
scarce:    R/K < 1/3
moderate:  1/3 <= R/K < 2/3
abundant:  R/K >= 2/3
```

The no-social baseline therefore uses a `3 x 2` Q-table.

## Social observation

The social representation is:

```text
sources[observer] = agents whose previous actions the observer sees
```

Information flows conceptually:

```text
source -> observer
```

The social signal is the fraction of observed sources that chose
`LOW_EXTRACT` on the previous step.

```text
mostly_high: low fraction < 1/3
mixed:       1/3 <= low fraction < 2/3
mostly_low:  low fraction >= 2/3
```

The joint learner state is:

```text
joint_state = 3 * ecological_state + social_state
```

so social treatments use a `9 x 2` Q-table.

## Social-network modes

### `random_k`

```bash
--social-mode fixed \
--social-network random_k \
--social-k 4
```

Every observer has exactly `k` distinct sources. Self-observation and
duplicate sources are forbidden.

This is the initial network used for `S1` and all rewiring treatments.

### `ba`

```bash
--social-mode fixed \
--social-network ba \
--ba-m 2 \
--rewiring none
```

This creates a fixed Barabasi-Albert-style observational network. The
underlying social ties are undirected and are represented as symmetric
observation lists, so attention and visibility vary with degree.

`m=2` gives mean degree close to four at moderate/large population sizes,
which makes it a useful visibility-skew comparison with the default
`S1` setting `k=4`. This is a project control choice; it is not a claim
that the numerical network density reproduces Schrama et al.

BA networks are fixed in the current core experiment. Rewiring a BA
network is deliberately not supported in this stage.

## Prediction error

Each social observer maintains an EWMA forecast of locally observed
low-extraction frequency.

```text
error_i(t) = |observed_i(t) - forecast_i(t)|
```

```text
forecast_i(t+1)
=
(1 - forecast_alpha) * forecast_i(t)
+ forecast_alpha * observed_i(t)
```

The current defaults are:

```text
forecast_alpha   = 0.50
rewire_threshold = 0.25
rewire_mu        = 0.10
rewire_every     = 50
```

Prediction error measures local social surprise. It does not measure
truth, misinformation, or objective competence.

## Local/global replacement search

A rewire replaces one existing source and therefore preserves attention
capacity in `random_k` treatments.

`theta` is the probability of requesting global search:

```text
theta = 0.00   local search
theta = 0.25   mostly local, some global search
theta = 1.00   global search
```

Local candidates are sources of current sources. If local search has no
legal candidate, the implementation explicitly falls back to global
search.

All agents at one rewiring checkpoint construct candidates from a
snapshot of the network at the start of that checkpoint.

## Matched random-turnover control

The confirmatory random control uses the realized number of successful
rewires from a paired adaptive run.

An adaptive run writes:

```text
data/rewiring_schedule.csv
```

with one row per rewiring checkpoint.

A paired `R0` run then uses:

```bash
--rewiring random_matched \
--matched-rewire-schedule <adaptive-run>/data/rewiring_schedule.csv
```

At every checkpoint, `R0` randomly selects exactly the recorded number
of observers and rewires them using the same `theta` search rule.

The runner checks that the schedule came from
`prediction_error` rewiring and that its `theta` matches the current
run.

The `mu` argument does not determine the event count in
`random_matched` mode. The paired adaptive schedule does.

## Temporal order

Training follows:

```text
(G_t, a_(t-1), R_t)
        -> s_t
        -> a_t
        -> R_(t+1)
        -> observe a_t through G_t
        -> prediction error
        -> optional rewiring
        -> G_(t+1)
        -> forecast update from the pre-rewire observation
        -> s_(t+1)
        -> Q update
```

The Q-learning target therefore uses the post-rewiring graph.

## Network evaluation

The default evaluation is:

```bash
--network-eval frozen
```

For social treatments this decomposes the learned system into:

```text
continuation
    trained Q
    training-end ecology
    terminal training graph
    frozen graph

fresh_reset
    trained Q
    fresh ecology
    terminal training graph carried forward
    frozen graph

fresh_reset_network
    trained Q
    fresh ecology
    exact initial training graph restored
    frozen graph
```

The historical label `fresh_reset` is retained for compatibility with
`baseline_validation_figures.py`; its network semantics are now made
explicit by the output field:

```text
network_start = terminal
```

The reset-network condition instead records:

```text
evaluation_mode = fresh_reset_network
network_start   = initial
```

This comparison isolates learned Q-policy effects from effects stored in
the rewired terminal topology. For S1 and S2, whose graphs never change,
the carried- and reset-network evaluations should coincide up to exact
deterministic evaluation behavior.

For adaptive prediction-error runs, an optional robustness evaluation is
available with:

```bash
--network-eval adaptive
```

This keeps all frozen evaluations and additionally runs:

```text
fresh_adaptive_network
    trained Q, frozen
    fresh ecology
    terminal training graph
    terminal training forecast state carried forward
    network continues adapting
```

Evaluation rewiring uses a random stream separate from training rewiring.
The input terminal graph and forecasts are copied before evaluation so the
training endpoint retained by the runner cannot be mutated by the
robustness evaluation.

Adaptive evaluation is intentionally not available for `random_matched`
R0 yet, because no paired evaluation-phase event-count schedule exists.
The confirmatory R0 comparisons therefore use the default frozen
evaluation decomposition.

## Measurements

Ecological/behavioral outputs include:

```text
low_extraction_rate
collective_order
action_entropy
mean_resource_fraction
total_resource
```

Material/welfare outputs include:

```text
mean_reserve_welfare
mean_need_satisfaction
deprivation_rate
mean_metabolic_shortfall
mean_energy
wealth_gini
mean_wealth
```

Social/network outputs include:

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

`majority_mismatch_rate` is calculated only over comparisons in which
neither the local sample nor the population-excluding-focal comparison
is tied. `majority_tie_rate` reports the excluded share explicitly.

`reciprocity` is the fraction of directed information edges whose reverse
edge also exists. `degree_assortativity` is the Pearson correlation between
source and observer visibility degree along directed information edges. For
symmetric BA-style observation graphs this reduces to ordinary degree
assortativity.

`visible_population_bias` is the edge-weighted visible low-extraction
fraction minus the population low-extraction fraction.

Policy diagnostics retain the unweighted Hamming measure and additionally
report:

```text
policy_hamming_visit_weighted_mean
visited_state_fraction
training_visit_fraction_<state>
```

The visit-weighted Hamming measure weights state disagreements by aggregate
training state occupancy, so arbitrary policy choices in never-visited
states do not contribute to the main heterogeneity diagnostic.

## Output files

Runs are written under:

```text
results/q_learning_baseline/experiments/<run-name>/
```

The canonical runner writes:

```text
config.json

data/
    evaluation_summary.csv
    training_timeseries.csv
    evaluation_timeseries.csv
    policy_summary.csv
    agent_policies.csv
    agent_social_summary.csv
    network_timeseries.csv
    network_edges_checkpoints.csv   # social runs: initial + terminal graphs
    rewiring_schedule.csv           # when rewiring is active
```

`config.json` includes command-line parameters plus Git, Python,
platform, package-version, and invocation metadata.

`agent_social_summary.csv` contains one row per agent and run with training-
time mean exposure, perception error, signed bias, majority tie/mismatch
rates, mean/final visibility, rewiring count, prediction error, and final
wealth/welfare quantities.

`network_edges_checkpoints.csv` stores directed `source -> observer` edges
for the exact initial and terminal training graphs, together with endpoint
visibility degree and attention size. This is intended for structural checks
and network visualizations without writing every edge at every timestep.

## Cross-treatment analysis

Stage 5 adds:

```text
social_experiment_analysis.py
SOCIAL_ANALYSIS.md
```

The analysis script reads multiple Stage-4-compatible experiment directories
and produces run-level distribution tables, bootstrap summaries, paired
adaptive-minus-matched-R0 effects, network-memory contrasts, theta x mu
summaries, and the social figure suite.

Example:

```bash
python social_experiment_analysis.py \
    --analysis-name pilot_core_v1 \
    --run results/q_learning_baseline/experiments/b0_pilot \
    --run results/q_learning_baseline/experiments/s1_pilot \
    --run results/q_learning_baseline/experiments/s2_pilot \
    --run results/q_learning_baseline/experiments/r1_pilot \
    --run results/q_learning_baseline/experiments/r0_matched_r1_pilot \
    --run results/q_learning_baseline/experiments/r2_pilot \
    --run results/q_learning_baseline/experiments/r0_matched_r2_pilot \
    --run results/q_learning_baseline/experiments/r3_pilot \
    --run results/q_learning_baseline/experiments/r0_matched_r3_pilot
```

The default analysis output is:

```text
results/q_learning_baseline/social_analysis/<analysis-name>/
```

Matched R0 effects are paired to the exact loaded adaptive source through the
recorded rewiring-schedule SHA-256 when available. Run-level outcome
distributions are retained, and aggregate confidence intervals use bootstrap
resampling rather than mean +/- SEM.

See `SOCIAL_ANALYSIS.md` for the complete input contract, output tables,
figure definitions, regime thresholds, and pairing semantics.

New experiment and analysis directories are ignored by Git. Final datasets
and derived analysis products should be frozen as deliberate research-release
artifacts rather than accumulated in ordinary source history.

## Tests

```bash
pytest -q
```

Optional syntax check:

```bash
python -m py_compile \
    Cognitive_tools/ecology.py \
    Cognitive_tools/model.py \
    Cognitive_tools/env.py \
    Cognitive_tools/qlearning.py \
    Cognitive_tools/social.py \
    qlearning_experiment.py \
    baseline_validation_experiment.py \
    social_experiment_analysis.py \
    baseline_validation_figures.py
```

## Core smoke tests

### B0

```bash
python baseline_validation_experiment.py \
    --run-name b0_smoke \
    --social-mode none \
    --rewiring none \
    --scenarios uniform_high patchy_high \
    --populations 8 \
    --replicates 1 \
    --training-steps 300 \
    --evaluation-steps 100 \
    --record-every 20
```

### S1

```bash
python baseline_validation_experiment.py \
    --run-name s1_smoke \
    --social-mode fixed \
    --social-network random_k \
    --social-k 4 \
    --rewiring none \
    --scenarios uniform_high patchy_high \
    --populations 8 \
    --replicates 1 \
    --training-steps 300 \
    --evaluation-steps 100 \
    --record-every 20 \
    --record-network-every 20
```

### S2

```bash
python baseline_validation_experiment.py \
    --run-name s2_smoke \
    --social-mode fixed \
    --social-network ba \
    --ba-m 2 \
    --rewiring none \
    --scenarios uniform_high patchy_high \
    --populations 8 \
    --replicates 1 \
    --training-steps 300 \
    --evaluation-steps 100 \
    --record-every 20 \
    --record-network-every 20
```

### Adaptive R2 schedule source

```bash
python baseline_validation_experiment.py \
    --run-name r2_adaptive_smoke \
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
    --populations 8 \
    --replicates 1 \
    --training-steps 500 \
    --evaluation-steps 100 \
    --record-every 20 \
    --record-network-every 50
```

### Matched R0 for that R2 run

```bash
python baseline_validation_experiment.py \
    --run-name r0_matched_r2_smoke \
    --social-mode fixed \
    --social-network random_k \
    --social-k 4 \
    --rewiring random_matched \
    --rewire-theta 0.25 \
    --matched-rewire-schedule \
        results/q_learning_baseline/experiments/r2_adaptive_smoke/data/rewiring_schedule.csv \
    --scenarios uniform_high patchy_high \
    --populations 8 \
    --replicates 1 \
    --training-steps 500 \
    --evaluation-steps 100 \
    --record-every 20 \
    --record-network-every 50
```

The adaptive and matched runs must use the same base seed, scenarios,
populations, replicate count, training length, rewiring interval, and
`theta`.

## Development sequence

Completed:

```text
unified ecology
generic tabular Q learner
scientific provenance foundation
fixed social observation
decentralized rewiring
experiment hardening
core social controls: S2 + matched R0
network evaluation decomposition
social measurements and normalized network/agent outputs
cross-treatment analysis and figures
```

Next:

```text
pilot experiments
confirmatory experiments
```

See `SOCIAL_EXPERIMENT.md` for the experimental specification and
`SOCIAL_ANALYSIS.md` for cross-treatment analysis semantics, and
`PROVENANCE.md` for mechanism-level scientific and code provenance.
