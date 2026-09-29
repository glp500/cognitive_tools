# Population payoff validation implementation plan

Status: implemented and verified. Pilot and held-out endpoint validation completed.
See docs/reviews/payoff-validation-results-2026-09-29.md for findings and the
prospective endpoint-gate refinement of the provisional full main run.
Date: 2026-09-29.

## Objective and claim boundary

Measure whether specified policy pairs exhibit a population social dilemma in
the existing renewable-resource environment. A result applies to the tested
policies, ecology, initial-state distribution, population, horizon and discount;
failure does not prove that no other policy pair can constitute a dilemma.
Keep the completed campaign and its frozen analysis intact.

References:
- Leibo et al., definitions of sequential social dilemmas:
  https://arxiv.org/html/1702.03037
- SocialJax, population incentive conditions and Schelling diagrams:
  https://arxiv.org/html/2503.14576v3 (sections 3.1 and 4.3).

## Protocol

### Policies and environments

Initial policy pair: C = always-low; D = always-high. These are candidate labels,
not an assertion that low extraction is already validated cooperation.
Evaluate N=64 in uniform_high, patchy_high and split_high_low with the campaign
ecology, actions, placement and reset rules. No training or adaptive rewiring.
Fixed policies ignore social observations, so repeating all network treatments
would add no information for this first pair.

Subsequent candidate policies may use the existing three ecological bins.
There are only 2^3 = 8 deterministic binary policies on this state space.
Screen these on development seeds for sustainable collective returns and
behavioral restraint; fix selected policy definitions before held-out validation.
This explores available behavior without changing the action space or ecology.
Socially conditional or learned policies are a later extension with specified
frozen networks, previous-action initialization and frozen policy artifacts.

### Paired focal-agent design

Let k be the number of cooperative OTHER agents, from 0 through 63.
For each independent ecological replicate, sample a uniform subset of focal
agents without replacement. Begin with eight focal agents per replicate.
For each focal agent, draw a seeded permutation of the other 63 agents; the
first k use C and the remainder D. This nested assignment gives a consistent
composition sequence. Repeat assignments if pilot assignment variance is large.

Run two independent environment instances from identical complete initial states:
one with the focal agent using C and one with it using D. Hold the others'
policy assignments fixed; their realized actions can respond to changed states
when conditional policies are introduced. Never hold future ecology fixed.
Use matched policy randomness if stochastic policies are added later.

Record C(k) and D(k), the focal agent's expected return in these two branches.
Stratified regional estimates are diagnostics; population averages retain the
actual population weights. Do not compare C and D group means from unequal
locations as a substitute for paired unilateral comparisons.

Compute full-population all-C and all-D endpoints separately, using every agent,
and retain per-agent and region results. These endpoints estimate the collective
return gap more precisely than a small focal subsample.

### Payoff and horizon

Accumulate each per-step reward returned by EcoEnv.step, not time-averaged wealth.
Store raw cumulative harvest and discounted harvest, plus resource/reserve
diagnostics. Verify reward sums equal final minus initial wealth.

Primary incentive estimand: sum from t=0 to H-1 of 0.95^t * reward_t,
with H=1000, matching the current learner discount. Also report the H=1000
undiscounted return to reproduce the existing controls. Keep verdicts separate.
The discount tail bound uses maximum per-step reward 0.020 in the current model:
0.020 * gamma^H / (1-gamma). Record the configured bound in outputs.

Prospective horizon diagnostics: cumulative harvest at H=5000 and H=20000,
and late-window harvest rates and resource trends. These are finite-horizon
diagnostics, not proof of an infinite-horizon equilibrium. A longer rollout
alone cannot fix short discounting. Additional discounts are separate estimands.
Initial evaluation uses fresh resets. A later continuation analysis must use a
named saved-state distribution and clone the same state for both branches.

### Population conditions and verdicts

Report these explicit contrasts for each ecology and return definition:

- Collective advantage G = mean individual all-C return minus all-D return.
- Exploitation loss E = C(63) - C(0).
- Greed at the cooperative endpoint = D(63) - C(63).
- Fear at the defective endpoint = D(0) - C(0).
- The entire unilateral incentive curve Delta(k) = D(k) - C(k).

The proposed endpoint validation requires G>0, E>0 and at least one of greed>0
or fear>0. This is a sufficient operational population check, not an exhaustive
test of all mixed-policy incentive patterns. Inspect near-endpoint values and
report interior-only conflict separately. Do not impose the two-player
2R>T+S condition on population averages without deriving its interpretation.

Use three verdicts: supported, contradicted for the tested contrasts, and
inconclusive. A confidence interval overlapping zero is inconclusive, not proof
of absence. Strict theoretical inequalities use zero; any practical minimum
effect is separately justified and fixed before confirmatory analysis.

Average nested focal/assignment observations within independent replicate first.
Bootstrap whole replicate blocks, preserving C/D and k pairing. Construct
simultaneous confidence intervals over the prespecified decision contrasts and
ecologies for each claim family; show pointwise descriptive curve intervals
only when clearly labeled. Freeze the family before inspecting final results.
For an OR condition, support requires at least one positive simultaneous lower
bound; evidence against requires both branches to be ruled out. Report individual
condition verdicts, never conceal mixed outcomes in a single overall label.

## Implementation tasks

### 1. Paired endpoint runner (medium)

Files: cognitive_tools/payoff.py, tests/test_payoff.py, docs/payoff-validation.md.
Reuse EcoEnv and scenarios.build_environment_maps; mirror recorded seed/settings
semantics. Keep the large training runner untouched where possible. Define a
small validated payoff configuration and a per-agent policy mapping.

Acceptance:
- CLI runs all-C, all-D, one focal D among C, and one focal C among D.
- Both branches share positions/maps/stocks and cannot mutate each other.
- Per-step returns, wealth deltas and endpoint controls agree.

Verification: deterministic replay and hand-calculated short rollout tests;
reproduce selected saved 1000-step control rows using their original seeds.
Dependencies: none.

### 2. Composition sweep and auditable data (medium)

Files: cognitive_tools/payoff.py, tests/test_payoff.py, docs/payoff-validation.md.
Add focal selection, nested co-player permutations and composition sweeps.
Write per-agent rows, paired focal rows, episode summaries and manifest.
Manifest records revision, dirty status/source hashes, settings, seeds, policy
definitions, horizons, discounts, sampled focal IDs and assignment identities.
Use a separate results/payoff_validation/<run_name>/ directory; refuse overwrite.

Acceptance:
- k counts OTHER cooperators exactly; focal switching never changes their IDs.
- All return definitions derive from the same rollout, with no retraining.
- Missing pairs, duplicate keys and incomplete runs are detected.

Verification: k=0/63 count tests, heterogeneous-location pairing test, replay and
reward-conservation checks. Dependencies: task 1.

Checkpoint: runner semantics verified before any full sweep.

### 3. Analysis and Schelling figures (medium)

Files: cognitive_tools/payoff_analysis.py, tests/test_payoff_analysis.py,
docs/payoff-validation.md.
Produce paired_returns.csv, contrasts.csv, validation_report.md, and PNG/PDF/SVG
figures: C(k)/D(k), Delta(k) with zero line, and all-C/all-D collective outcomes.
Keep resource/reserve diagnostics on separate axes from harvest returns.
Do not call a weighted focal curve the population welfare curve: full population
returns must come from recorded episode outcomes with explicit composition.

Acceptance:
- Block resampling preserves paired dependence and weights replicates equally.
- Figures show uncertainty, policy/return definitions and scenario labels.
- Synthetic known-dilemma, no-collective-benefit, no-conflict and inconclusive
  fixtures produce correct condition-specific verdicts.

Verification: hand-calculated synthetic fixtures, malformed-data rejection, and
visual inspection of exported figures. Dependencies: task 2.

### 4. Pilot and fixed validation run (small configuration/documentation task)

Pilot: 10 independent seeds, 2 focal agents, k={0,1,16,32,48,62,63}, three
ecologies, H=1000. Measure throughput and variance components.
Provisional main run: 100 independent seeds, 8 focal agents,
k={0,1,8,16,24,32,40,48,56,62,63}. Finalize precision target and computational
budget from pilot variance/runtime; do not choose sample size by significance.
Long-horizon diagnostics begin with endpoints, then expand if warranted.
Use disjoint development/held-out seeds for any policy or parameter selection.

Acceptance:
- Protocol, decision family and sample size are frozen before held-out results.
- Baseline and any revised environment get distinct immutable run identities.
- Every claimed result identifies its policy, ecology, horizon and utility.

Verification: manifest/data completeness audit, endpoint reproduction, focused
tests plus existing environment tests, and report review. Dependencies: task 3.

Checkpoint: inspect scientific findings before proposing a model revision.

## Failure diagnosis and prospective revision

1. G fails: determine whether constant low is an inefficient policy or whether
   depletion losses occur beyond the payoff horizon. Screen the eight existing
   ecological-bin policies on development seeds. Diagnose per-step harvest,
   resource stocks and regrowth. Do not relabel reserve welfare as harvest utility.
2. G passes but unilateral conflict is absent: these policies may form a common-
   interest problem. Test alternative exploitative policies before changing
   mechanics. If theory calls for a commons dilemma, consider greater sharing
   of renewable patches or competition exposure; quantify private gains and
   harms to others separately from self-depletion.
3. Depletion occurs even under restraint: compare demand against local net
   regeneration and occupancy. Revise extraction scale, recovery or the action
   set only when justified. A zero-harvest action permits resting but is a new
   model. Greater coupling is not guaranteed to strengthen a dilemma: it can
   rescue depleted cells as well as spread depletion.
4. Long-run benefit exists but discounted benefit fails: report horizon dependence.
   A larger gamma changes agent preferences and requires a new learning campaign;
   a longer simulation alone does not change those preferences.
5. Only reserve welfare favors restraint: decide whether welfare is truly the
   intended agent utility. A survival/need-sensitive reward is a substantive
   alternative objective, to be justified and validated as a new experiment.
6. Effects depend on ecological region: report asymmetric incentives; a positive
   pooled gap does not mean every type benefits. Revise or narrow the claim.
7. Wide intervals: improve independent replication or focal/assignment coverage
   according to the dominant variance component before modifying mechanics.

For any revision, change one mechanism at a time; use a bounded, theoretically
motivated development parameter sweep, then freeze a candidate and evaluate on
held-out seeds. Record failures. Never add a direct cooperation bonus solely to
force positive results: it changes the incentives and can remove the conflict.

A useful ecological diagnostic is local net growth g(R)=rR(q-R/K), maximized
at R=qK/2 with g_max=rKq^2/4 for an isolated cell before clipping. For uniform-high
parameters this is about 0.00459 per step per cell. Current low requests are
0.002 per occupant and high requests 0.020. This suggests plausible restraint
and over-extraction regimes but is not a proof: co-location, post-harvest update
ordering and spatial coupling determine actual sustainable yields.

## Scope and limitations

This plan validates an environmental incentive structure; it does not establish
learning convergence, equilibrium, all-policy robustness, or that social
information improves decisions. Those remain separate research questions.
No migration is needed to run this analysis. Preserve the paired-policy protocol
if a subsequent study moves to SocialJax or Melting Pot.
