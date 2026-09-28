# Experiment specification

The research questions and treatment overview are in the [README](../README.md).
This is the canonical specification for mechanisms, outcomes, and causal
contrasts. The implementation is `cognitive_tools.experiment`; all treatments
use the same ecological and learning mechanisms.

## Ecology and baseline

B0 is independent ecological Q-learning without social observation. Agents have
fixed random positions on a rectangular grid; co-location is allowed. There is
no movement, communication, sanctioning, transfer, or shared Q table.

At each step, requested extraction is allocated from each occupied tile,
proportionally when demand exceeds its resource. Realized harvest is removed,
credited to wealth and energy, and followed by metabolism. The remaining stock
`R` then undergoes:

```text
R_next = clip(R + r R (1 - R/K) - r (1-q) R + c (neighbor_mean(R) - R), 0, K)
```

Here `K` is capacity, `r` regeneration, `q` equilibrium fraction, and `c`
spatial coupling. This equation acts on post-harvest stock; extraction is not
subtracted a second time. Boundaries do not wrap. An isolated, unharvested tile
has positive equilibrium `q K`. Defaults are a 10×10 grid, initial stock
`0.50 K`, and coupling `0.10`.

`cognitive_tools.scenarios` defines the supported landscapes:

| Scenario | Structure |
|---|---|
| `uniform_high` | Uniform productive ecology |
| `patchy_high` | Patchy productive ecology |
| `decentralized_high` | Multiple productive centers |
| `centralized_low` | Centralized low, slow ecology |
| `patchy_high_central_low` | Productive patchy background, low center |
| `central_high_patchy_low` | Low patchy background, productive center |
| `split_high_low` | Productive and low fragmented halves |
| `decentralized_high_in_low` | Productive islands in a low background |

Single-component high landscapes use mean capacity `0.75`, regeneration `0.05`, and equilibrium
fraction `0.70`; the single low landscape uses `0.35`, `0.03`, and `0.60`. Mixed landscapes have their own regional parameter values and masks, defined
in `scenarios.py`; configurations record scenario identifiers and specifications,
and the recorded Git revision identifies their exact construction. Landscape and position seeds match across scenarios
within each replicate.

## Learning, state, and reward

Actions are `LOW_EXTRACT=0` and `HIGH_EXTRACT=1`, requesting `0.002` and `0.020`
by default. Reward is realized individual harvest only. The independent tabular
update is `Q(s,a) += alpha * (reward + gamma * max Q(s_next,·) - Q(s,a))`;
terminal updates omit the bootstrap term.

Ecological state uses local `R/K`: scarce below `1/3`, moderate from `1/3` to
below `2/3`, abundant at or above `2/3`. B0 has three states. Social treatments
also bin the fraction of observed sources choosing low extraction using those
same thresholds. Before previous actions exist, the social fraction is `0.5`.
The action at time `t` uses source actions from `t-1`.

Social joint state is `3 * ecological_state + social_state`, giving nine states.
Each agent maintains its own Q table. Defaults are learning rate `0.10`, discount
`0.95`, epsilon `0.20`, floor `0.02`, decay `0.9995`, 5,000 training steps, and
1,000 evaluation steps. Evaluation freezes Q tables and uses greedy learned
policies. Always-low, always-high, and random 50/50 policies provide fresh-reset
controls in the same runner.

## Observation networks

`sources[observer]` lists the agents it observes. Information edges point
`source -> observer`. Visibility is the number of observers of a source. Social
edges never directly move resources or change ecological diffusion.

S1 and rewiring treatments start from directed random-k networks. Each observer
has exactly `k` distinct non-self sources; there are `N*k` edges. Default `k=4`.
Initial graphs depend on base seed, replicate, and population, independently of
ecological scenario and rewiring treatment.

S2 starts from a clique of `m+1` nodes and adds newcomers with `m` distinct
existing targets sampled proportional to degree. Undirected ties become
symmetric source lists. Default `m=2` yields mean degree near four but variable
attention and visibility. It remains fixed. This is a project control, not a
numerical reproduction of the source paper's network density.

## Prediction and replacement

Each observer starts with forecast `0.5`. With current observed fraction `o_i`
and forecast `f_i`, prediction error is `abs(o_i-f_i)`, measured before updating
`f_i <- (1-alpha_f)*f_i + alpha_f*o_i`; default `alpha_f=0.50`.

Prediction-error rewiring checks every 50 steps by default. An observer is
eligible when error is strictly greater than the threshold (default `0.25`),
then rewires with probability `mu` (default `0.10`). Error is local social
surprise, not truth, misinformation, competence, or sustainability.

Each successful event randomly drops one source and adds one legal replacement,
preserving attention capacity. Search is global with probability `theta`, local
otherwise. Local candidates are sources of sources; global candidates are all
non-self agents not already followed. Empty local sets fall back to global
search. All candidates use a snapshot from the beginning of the checkpoint.
R1, R2, R3 use `theta=0`, `0.25`, `1`; other values are labeled `R_adaptive`.

## Matched R0 and schedules

R0 uses `--rewiring random_matched --matched-rewire-schedule <path>`. First run
the paired adaptive condition. Its `data/rewiring_schedule.csv` records each
`(scenario, population, replicate, time)` checkpoint, including zero events.

The runner validates source mode, theta, seed, timing, topology, attention
capacity, and training length. It samples exactly the source checkpoint's
successful event count from legally rewritable observers, uniformly without
replacement. Each selected observer then makes one replacement using the same
search rule. R0's `mu` does not set event counts.

The schedule records `scenario`, `population`, `replicate`, `time`, `rewiring`,
`rewire_theta`, `rewire_every`, `base_seed`, `social_network`, `social_k`,
`training_steps`, `target_rewires`, and `successful_rewires`. Adaptive target
counts are blank; R0 target and realized counts must agree. Archive the source
schedule with both runs. Configurations record its SHA-256 for exact analysis
pairing. Probabilistic unmatched random turnover is not a supported treatment.

## Temporal ordering

```text
(G_t, a_(t-1), R_t) -> s_t -> a_t -> R_(t+1)
-> observe a_t through G_t -> prediction error -> optional rewiring
-> G_(t+1) -> forecast update using observation from G_t
-> s_(t+1) -> Q update
```

The Q target uses the post-rewiring network. Forecasts use the pre-rewiring
observation. The ecology, source graph, forecasts, and learners retain separate
responsibilities and deterministic random streams.

## Evaluation and network memory

Default `--network-eval frozen` freezes both policies and graphs:

| Output mode | Ecology | Social graph |
|---|---|---|
| `continuation` | Training endpoint | Terminal training graph |
| `fresh_reset` | Common fresh reset | Terminal graph carried forward |
| `fresh_reset_network` | Same fresh reset | Exact initial training graph |

The last mode applies to social runs. `fresh_reset` is a current analysis-schema
label: it records `network_start=terminal`, `network_adaptive=false`.
`fresh_reset_network` records `network_start=initial`. Their within-run
difference estimates performance associated with learned topology conditional
on the same trained Q tables. S1/S2 graphs never change, so these evaluations
agree. Ecological resets also reset wealth and welfare state.

For prediction-error runs only, `--network-eval adaptive` additionally produces
`fresh_adaptive_network`: fresh ecology, frozen Q, copied terminal graph and
forecast, and a separate evaluation-rewiring RNG stream. It cannot change the
stored training endpoint. Matched R0 lacks an evaluation-phase matching
schedule, so the primary paired analysis uses frozen evaluation. Output fields
include `network_start`, `network_adaptive`, `evaluation_total_rewires`, and
per-step/cumulative evaluation rewires.

## Outcomes and measurement

Primary sustainability is `eval_mean_mean_resource_fraction`. Behavior includes
low-extraction rate, collective order, action entropy, and total resource.
Low extraction is a behavioral proxy, not a claim about intent.

Harvest accumulates as wealth, whose inequality is `wealth_gini`. Energy is a
bounded reserve (default initial value and capacity `1.0`). Metabolic need is
`0.002` per step. Welfare records reserve/capacity, fraction of need met,
deprivation rate, metabolic shortfall, and raw energy. These are outcomes,
not learning rewards.

For each observer, perception compares its local low-action fraction `o_i`
with the fraction among all other agents `c_-i`. Absolute error is
`abs(o_i-c_-i)` and signed bias is `o_i-c_-i`. Majority mismatch uses only
comparisons where neither fraction equals `0.5`; majority tie rate explicitly
records the excluded fraction. Undefined correlations remain `NaN`.

`visible_population_bias` subtracts population low-action fraction from the
edge-weighted visible low-action fraction. Reciprocity is the fraction of
directed information edges whose reverse exists. Degree assortativity is the
Pearson correlation of source and observer visibility degrees along edges.
Other diagnostics include visibility Gini, maximum visibility share, invisible
fraction, degree/action correlation, prediction error, turnover, rewire counts,
global-search fractions, and local fallback counts.

Policy diagnostics include pairwise Hamming disagreement and a version weighted
by aggregate training state occupancy; never-visited states receive zero weight.
The measurement schema identifier `stage4_v1` is retained as a data contract.
`agent_social_summary.csv` contains per-agent training observations, errors,
ties, visibility, rewires, and final welfare/material quantities.
`network_edges_checkpoints.csv` records exact initial/terminal source→observer
edges and endpoint degrees; it is not a full edge timeseries.

## Causal contrasts and limits

- S1 − B0: network-limited social observation relative to ecology-only learning.
- S2 − S1: fixed skewed visibility, including variable attention capacity.
- R0 − S1: random topology turnover at the matched event rate.
- Adaptive − paired R0: error-based observer selection beyond equal event counts.
- R1/R2/R3: replacement-search scope, including local fallback behavior.

Visibility, perception, policy heterogeneity, welfare, and resource outcomes
are endogenous measurements. Correlations do not establish causal mediation.
Network-memory differences condition on learned policies; they do not identify
an independent topology effect across all possible policies. The project does
not implement the full HSM or DeGroot models, misinformation, prestige,
homophily, or payoff-based rewiring. See [provenance](provenance.md).

## Campaigns

`scripts/run_social_campaign.sh` is the sole orchestration path. It refuses a
dirty worktree, records the commit, compiles and tests, runs adaptive conditions
before matched controls, and verifies analysis warnings, pairings, and commits.
`PYTHON_BIN` selects an interpreter; `RESUME=1` reuses completed directories,
subject to the final manifest checks. Choose a unique tag for a new campaign.

| Mode | Scenarios | Populations | Replicates | mu | Training / evaluation |
|---|---|---|---|---|---|
| smoke | uniform high | 8 | 2 | 1 | 12 / 6 |
| pilot | uniform high, patchy high, split high/low | 32, 64 | 10 | 0.10 | 5000 / 1000 |
| full | same as pilot | 64 | 100 by default | 0.05, 0.10, 0.20 | 5000 / 1000 |

All modes include B0, S1, S2 and theta `0, 0.25, 1` with paired R0. Smoke uses
4×4 grids, checkpoints every two steps and threshold zero to exercise rewiring;
it is a mechanical check. Full requires `CONFIRM_FULL=YES` and accepts
`FULL_REPLICATES`. Bootstrap defaults are 20/2000/5000 respectively, overridable
with `BOOTSTRAP_REPS`. Freeze these choices before research execution.

The standalone historical baseline campaign is preserved in the pre-cleanup
snapshot. B0 and its fixed-policy validation remain available through the
canonical runner. Seeds and run metadata are described in
[provenance](provenance.md); output contracts are in [analysis](analysis.md).
