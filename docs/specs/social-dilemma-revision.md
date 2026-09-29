# Specification: a capped-harvest social-dilemma variant

Status: proposed, 29 September 2026. Production implementation has not started.
Decision evidence: [codebase and incentive review](../reviews/incentive-redesign-review-2026-09-29.md).
Execution order: [plan](../../tasks/plan.md) and [tasks](../../tasks/todo.md).

## Objective and scope

Add a separately named model variant in which individual utility has a daily
satiation threshold. Validate its population incentives before adopting it for
the project's learning and social-perception experiments. Preserve the completed
gross-harvest study and its interpretation.

The cheapest proposed mechanism is:

```text
harvest mode:         reward_i(t) = actual_harvest_i(t)
capped_harvest mode:  reward_i(t) = min(actual_harvest_i(t), metabolism_rate)
```

The candidate threshold is the existing metabolism rate, 0.002. It is fixed
before confirmation, rather than optimized as an extra parameter. Both actions
receive the same utility for the same realized harvest. There is no direct
cooperation bonus, social reward averaging, or penalty attached to an action label.

This is a substantive change from maximizing accumulated harvest to maximizing
satiating current-harvest utility. Harvest above the threshold can still enter
physical energy reserves and gross wealth, but contributes no additional current
reward. The reward is **not** actual metabolic consumption from reserves,
`need_satisfaction`, survival probability, or reserve welfare. Starting with full
energy does not supply reward. This interpretation must be acceptable for the
new research question; otherwise retain gross-harvest utility and pursue the
more uncertain ecological redesign described in the review.

Keep N=64, all three ecologies, positions, prorated allocation, resource dynamics,
actions 0.002/0.020, bins, initial stocks, observations, learner alpha/gamma,
and network rules. In particular gamma remains 0.95. Adoption requires new
training; rescoring old learned policies does not show how agents learn under
the new objective.

## Scientific acceptance

For fixed policies C=000 (always-low) and D=111 (always-high), define k as the
number of C policies among the other 63 agents. Use paired initial states and
co-player assignments for C(k) and D(k). Let:

- G = all-C mean individual utility minus all-D mean individual utility.
- E = C(63) - C(0).
- Greed = D(63) - C(63).
- Fear = D(0) - C(0).

The operational endpoint criterion is G>0 AND E>0 AND (greed>0 OR fear>0).
The proposed variant is expected to satisfy the fear branch. Strict positive
greed is not required; numerical ties must not be advertised as greed. This is
a policy-relative population criterion, not proof of a prisoner's dilemma,
equilibrium, all-policy robustness, or a benefit from social information.

Primary return is sum(gamma^t * reward_t), t=0..999, gamma=0.95. Also require
support for the separately reported undiscounted H=1000 criterion before claiming
this variant is robust to these two payoff definitions. Never select whichever
return passes. Use simultaneous 95% intervals across four endpoint contrasts
and three ecologies within each return family, bootstrapping independent
replicate blocks. Average sampled focal agents within each block first.

Support requires positive lower bounds for G and E and at least one conflict
branch, in every ecology. An interval spanning zero is inconclusive. Report
each condition and each return separately. Treat values within 1e-12 payoff
units of zero as numerical zero for decisions; this is floating-point handling,
not a substantive effect threshold. Report effect sizes as a percentage of
the maximum capped return (0.04 discounted; 2.0 undiscounted), as well as raw
units. Do not substitute preserved resources or reserves for a failed G test.

### Staged protocol

1. **Implementation replay:** reproduce the development probe on indices
   2010–2029, including its recorded focal IDs, with the real reward API.
   Agreement checks implementation, not independent confirmation.
2. **Development pilot:** indices 2100–2119; four uniformly sampled focal agents
   without replacement; one nested co-player permutation each; k={0,1,16,32,48,62,63};
   three ecologies; H=1000. Record assignment and regional diagnostics.
3. **Freeze:** candidate, software hashes, inference, 100 independent replicates,
   four focal agents, and indices 3000–3099. Base seed stays 20260928 with the
   existing separate landscape/position/assignment streams. These indices have
   not been used in this review. Check the seed ledger again before execution.
4. **Held-out endpoint gate:** k={0,63}, 3,000 unique episodes. All-C/all-D use
   all 64 agents; focal sampling applies to unilateral comparisons. If a required
   condition is contradicted or inconclusive, stop adoption and diagnose it.
5. **Optional held-out interior extension:** seven pilot compositions, same
   frozen settings and held-out blocks, 15,000 total unique episodes for a full
   run. Existing CLI may rerun endpoints in a new directory rather than add a
   resume feature. State that these endpoints duplicate the gate, not independent
   confirmation. Plot C(k), D(k), Delta(k) and population outcomes with uncertainty.
   The cheaper deliverable combines explicitly exploratory pilot curves with
   separate confirmatory endpoint panels; do not draw unmeasured held-out curves.

Allocation seeds and policies are fixed across both payoff definitions. The
two returns come from each same rollout. Region-specific contrasts are
diagnostic: the primary claim is population-average, not that every location
benefits. Do not change sample size or tune the cap after seeing held-out results.
A revised candidate uses a new protocol and unused seeds (reserve 4000 onward).

## Software contracts

### Environment and reward accounting

Add `reward_mode="harvest"` to `EcoEnv`. Supported values are `harvest` and
`capped_harvest`. For capped mode require finite, positive `metabolism_rate`.
Resolve the cap from that value; no independent cap CLI knob is needed.

Compute gross realized harvest from wealth increments before transforming reward.
`infos[name]["harvested"]` always means physical harvest, including capped mode.
Add `utility_reward` and `uncredited_harvest` to infos. Preserve gross wealth,
energy, observations and all ecological transitions. For fixed action sequences,
both modes must produce identical physical trajectories and RNG consumption.

Illustrative style, to implement in the existing environment rather than a new
reward framework:

```python
harvested = agent.wealth - wealth_before[name]
reward = harvested
if self.reward_mode == "capped_harvest":
    reward = min(harvested, self.metabolism_rate)
```

Keep Python 3.12, existing pinned dependencies, typed configuration fields,
snake_case names, and Ruff's current 100-column formatting. No new dependency
or generic plugin architecture is needed.

### Independent validation path — implement first

Extend `PayoffConfig` with `reward_mode` and forward it in `make_env`.
`return_sum` and `return_discounted` continue to mean actual rewards returned
by the environment. Add gross `harvest_sum`, `harvest_discounted`, and
`uncredited_harvest_sum` / `uncredited_harvest_discounted` to per-agent records.
Keep `wealth_delta` and `late_harvest_rate` as physical quantities; add
`late_utility_rate` instead of changing the old field's meaning.

For each agent and horizon, audit:

```text
harvest_sum = wealth_delta
utility_sum + uncredited_harvest_sum = harvest_sum
utility_discounted + uncredited_harvest_discounted = harvest_discounted
0 <= utility_sum <= min(harvest_sum, horizon * cap)  [capped mode]
```

Also check the discounted cap bound and finite/nonnegative accounting fields.
Do not cap the cumulative harvest: sum(min(h_t,b)) generally differs from
min(sum(h_t), H*b). Sum stepwise utility before aggregation.

Write `population_payoff_v2` manifests with explicit reward mode, formula version,
resolved cap, and utility units. Include these semantics in the protocol hash.
Use the mode-specific maximum reward for the discount-tail bound. Keep the
loader for original v1 gross-harvest results: hash the original stored config
before normalizing missing reward mode to `harvest`. Do not rewrite old manifests
or pretend v1 has capped reward fields. Unknown schemas/modes fail clearly.

Update audit and plot labels to distinguish gross harvest, capped-harvest
utility, reserve welfare, and resources. The G/E/greed/fear mathematics and
replicate-block design are reused. Verify numerical-zero treatment with an
exact reward tie and tiny floating-point perturbations.

### Learning integration — only after validation

Add `--reward-mode` to `experiment.py`; record explicit resolved semantics in
config and provenance. Validate them before `validate_configuration`'s
`social_mode == "none"` early return. Forward the same mode through every
training, fresh-evaluation and carried-evaluation environment constructor.

Training already updates Q from returned rewards. Keep that path. In
`evaluate_policy`, retain the returned rewards currently discarded and aggregate
each agent's evaluation utility from the first evaluation step, with discount
exponent zero. This also applies to carried-state evaluation: do not include
training wealth or training reward in evaluation utility. Record population
mean utility sum and discounted return in summary outputs, alongside gross
harvest and existing physical metrics. Avoid labeling time-averaged wealth as
payoff. Keep existing wealth Gini defined on gross nonnegative wealth.

Include canonical reward identity in cross-treatment compatibility checks,
checkpoint/resume comparisons, and adaptive/R0 schedule metadata and validation.
Legacy missing identity means harvest only, never capped mode. New R0 runs
must use schedules from their own newly trained adaptive counterpart. Existing
random seeds may be deliberately paired across reward variants for diagnostics,
but reward variants cannot be silently pooled into the original campaign analysis.

Preserve the frozen focused-analysis profile and its original hypotheses.
Add utility summaries as an explicitly named new-study output. A revised full
campaign is a separate experiment with new run names, its own freeze, and all
learners retrained. Validate a small B0/S1/adaptive/R0 lifecycle first. Do not
immediately repeat all 27 conditions × 100 replicates merely because fixed
policies pass the incentive gate.

## Files and commands

First stage: `cognitive_tools/env.py`, `payoff.py`, `payoff_analysis.py`;
`tests/test_reward_modes.py`, `test_payoff.py`, `test_payoff_analysis.py`;
new `configs/payoff/capped_{pilot,gate,curves}.json`; `docs/payoff-validation.md`.
Second stage: `experiment.py`, `analysis.py`, targeted lifecycle/compatibility
tests, and a separately named campaign recipe/document. `model.py`, `ecology.py`,
`scenarios.py`, `social.py`, `qlearning.py` need no mechanism changes.

Commands below assume an activated project environment. Config files and the
new training flag are proposed interfaces, not available yet.

```bash
python -m pytest -q tests/test_reward_modes.py tests/test_payoff.py tests/test_payoff_analysis.py
python -m ruff check .
python -m ruff format --check .
python -m pytest -q

python -m cognitive_tools.payoff --config configs/payoff/capped_pilot.json \
    --output results/payoff_validation/capped_pilot_v1 --workers 14 --purpose pilot
python -m cognitive_tools.payoff_analysis \
    --run results/payoff_validation/capped_pilot_v1 \
    --output results/payoff_validation/capped_pilot_v1/analysis --resamples 5000

python -m cognitive_tools.payoff --config configs/payoff/capped_gate.json \
    --output results/payoff_validation/capped_gate_v1 --workers 14 --purpose validation
python -m cognitive_tools.payoff_analysis \
    --run results/payoff_validation/capped_gate_v1 \
    --output results/payoff_validation/capped_gate_v1/analysis --resamples 5000

# Only after the gate and learning integration:
python -m cognitive_tools.experiment --run-name capped_b0_smoke_v1 \
    --reward-mode capped_harvest --scenarios uniform_high --populations 8 \
    --replicates 2 --training-steps 20 --evaluation-steps 10 --record-every 5
```

Use the existing serial/parallel reproducibility tests; do not build a new
execution framework. Full confirmation is a scientific acceptance run, not a
unit test whose expected result should be hard-coded to pass.

## Testing and completion criteria

1. Hand-calculated one/two-step examples cover below-cap, above-cap and scarce
   prorated harvest; zero harvest gives zero utility despite full energy.
   Both modes reproduce identical ecology, wealth and observations under fixed
   actions. Invalid modes and zero/nonfinite cap inputs fail.
2. Runner records actual per-step utility; aggregate cap counterexamples,
   gamma timing, multiple horizons, paired resets, hashes, missing rows,
   tampered gross/utility accounting and old v1 reads are tested.
3. Known synthetic population games retain correct condition-specific verdicts;
   ties cannot pass from floating-point noise. Exported labels/figures are
   inspected. Replayed development outcomes agree within numerical tolerance.
4. Held-out condition table and full limitations are published even on failure.
   Adoption requires the prespecified scientific gate; working software alone
   does not count as a validated social dilemma.
5. Before learning use, tiny real runs prove Q updates and evaluation summaries
   use the selected utility; mismatched resumes, reward families and R0 schedules
   are rejected. Existing gross-harvest tests and campaign reproduction still pass.

## Boundaries and failure response

Always keep physical accounting separate, defaults backward compatible, reward
identity explicit, and original results immutable. Record failed candidates.
Never relabel the original campaign as validated, select only passing ecologies
without narrowing the claim, or tune against the reserved confirmation set.

The current deliverable is this reviewable plan/spec. Framework migration,
new actions, movement, survival/death mechanics, changing the ecology, and a
full new learning campaign are later scope decisions, not prerequisites for
the first validation slice.

If G fails, inspect all-C shortfall by occupancy/region and whether resource
preservation supplies current utility. If conflict fails, inspect whether high
requests actually improve allocation under scarcity; a common-interest game
is not repaired by changing its label. If intervals overlap zero, use the pilot
variance to design a new fixed replication plan, not repeated significance
checks. If the mechanism needs changing, return to development and reserve new
confirmation seeds. See the review's bounded gross-harvest fallback.
