# Population payoff validation

This is a separate experiment. It reuses `EcoEnv` and the existing ecological
scenarios, without training, rewiring, or changing extraction, rewards or resource
dynamics. Results do not alter the completed ecology/perception campaign.

## Run and analyze

From the repository root, with its dependencies installed:

```bash
python -m cognitive_tools.payoff --config configs/payoff/pilot.json \
  --output results/payoff_validation/my_pilot --workers 4
MPLCONFIGDIR=/tmp/cognitive-mpl python -m cognitive_tools.payoff_analysis \
  --run results/payoff_validation/my_pilot \
  --output results/payoff_validation/my_pilot/analysis
```

Output directories must be new: both commands refuse overwrites. Replicate jobs
run independently in worker processes; each completed job writes CSV shards.
A failed run retains its manifest and shards for diagnosis, but is rejected by
analysis. Resume is not implemented: rerun to a new directory.

The completed implementation and findings are summarized in
[the validation review](reviews/payoff-validation-results-2026-09-29.md).
A 100-seed held-out endpoint gate was run using
`configs/payoff/heldout_endpoint_gate.json`, following the
[prospectively recorded refinement](reviews/payoff-validation-protocol-2026-09-29.md).

The default configuration is the 10-seed pilot. `configs/payoff/validation.json`
is the proposed 100-seed, eight-focal-agent held-out run. Its replicate indices
start at 1000, disjoint from pilot indices 0–9 and the old campaign 0–99.
This configuration is a candidate protocol, not evidence that the main run has
been performed or its precision approved. Use the pilot's runtime and replicate
variance to finalize the protocol before starting it. A run marked `validation`
is a user declaration of purpose, not an automatic scientific certification.

```bash
python -m cognitive_tools.payoff --config configs/payoff/validation.json \
  --purpose validation --output results/payoff_validation/main_v1 --workers 4
```

Configuration is JSON matching `PayoffConfig`; CLI values override JSON values.
`--horizons 1000 5000 20000` records multiple horizons from the same rollout.
For endpoint-focused horizon diagnostics use `--compositions 0 63` and an explicit
small replicate/focal count before attempting long full composition sweeps.
No infinite-horizon stability is claimed from finite trajectories.

## Policy and pairing semantics

`policy_c` and `policy_d` are three binary digits for scarce, moderate and
abundant ecological observations, in that order. Defaults are `000` (always low)
and `111` (always high). All eight deterministic resource-bin policies are
supported. For example, `001` harvests high only in the abundant bin. Policies
use the existing observation binning, including float32 local observations.
C and D are candidate labels, not a declaration of cooperation or defection.

For each independent replicate, sample focal agents uniformly without replacement.
For each focal and assignment draw, permute the other N−1 agents. At composition
k the first k use C; the others use D. Change only the focal policy between the
C and D branches. All maps, positions and initial stocks are reproduced exactly.
Co-player policies remain fixed, while their actions may respond to changed
resources if conditional policies are used. Homogeneous endpoint episodes are
reused exactly, rather than unnecessarily simulated for each focal agent.

Landscape seed = base seed + replicate. Position seed = base seed + 100000 +
replicate, reproducing campaign semantics. Focal and assignment streams use
separate NumPy SeedSequence domains, recorded in the manifest. Focal identities
and permutations are shared across ecologies for matched comparisons. Within a
replicate, individual agents and compositions are dependent observations.

## Returns and scientific conditions

For each agent accumulate every actual reward returned by `EcoEnv.step`:

- Undiscounted: sum of harvest over H steps.
- Discounted: sum of gamma^(t−1) times harvest at step t, default gamma=0.95.
- Check: undiscounted return equals final wealth minus initial wealth.

The discounted return is the primary learner-aligned incentive measure. The
undiscounted H=1000 return matches the prior campaign controls. Do not replace
harvest with reserve welfare when testing this reward structure. Gamma=1 makes
the two returns equal; gamma=0 includes only the first reward. For gamma<1 the
manifest records a truncation tail bound using the maximum possible individual
harvest per step. Each horizon/return combination receives its own verdict.

Let C(k), D(k) be focal returns with k cooperative OTHER agents. Test:

1. G: population mean all-C return minus population mean all-D return.
2. E: C(N−1) minus C(0).
3. Greed: D(N−1) minus C(N−1).
4. Fear: D(0) minus C(0).

The operational endpoint criterion is G>0 AND E>0 AND (greed>0 OR fear>0).
This is a sufficient endpoint check, not exhaustive characterization of all
interior compositions or all policies. The full D(k)−C(k) curve is also shown.
Two-player payoff conditions should not be applied to population averages
without a justified reduction. An asymmetric ecology also requires care:
positive population means do not imply benefit for every region or agent.

## Inference

Average focal/assignment results within replicate first, then weight independent
replicates equally. Whole replicate rows are bootstrap-resampled, preserving
C/D, composition and ecology pairing. Decision intervals use a centered
max-standardized bootstrap over the four contrasts times all configured
ecologies, separately for each horizon/return; the default is 5000 resamples.
They are approximate simultaneous 95% intervals, not exact finite-sample bounds.
Different horizon/return families are not protected against cross-family selection.
Small pilots and zero empirical variance warrant caution.

- Positive lower bound: condition supported.
- Nonpositive upper bound: condition contradicted under the strict inequality.
- Otherwise: inconclusive.
- Overall support requires G and E support, and support for greed OR fear.
- Overall contradiction requires G or E contradiction, or both greed and fear
  contradiction; other combinations remain inconclusive.

Curve shading is pointwise percentile-bootstrap uncertainty for description,
not a substitute for the decision intervals. At least two replicates are needed.
A pilot is exploratory, regardless of its numerical verdict.

## Artifacts and audit

Runner:

- `agent_returns.csv`: individual return, policy, location, region, low-action
  fraction and late-window harvest rate for each episode/horizon.
- `episode_summary.csv`: collective returns, resource and reserve outcomes.
- `paired_returns.csv`: matched focal C/D returns and unilateral differences.
- `assignments.csv`: focal identities and complete co-player permutations.
- `manifest.json`: settings, run purpose/status, seeds, source and output hashes,
  revision, dirty status, package versions, runtime and episode counts.
- `shards/`: completed per-replicate outputs retained for audit.

Analysis verifies output hashes, exact pair/episode/agent coverage, duplicate
keys, policy assignments, consistent positions, reward conservation and
agreement of paired returns with the underlying focal rows before inference.
It produces replicate returns, curve summaries, contrasts, verdicts, endpoint
diagnostics, a report, an analysis manifest, and PNG/PDF/SVG figures.

The proposed full configuration requires 48600 physical episodes per horizon
trajectory (48.6 million environment steps at H=1000). The default pilot requires
780 episodes. Extrapolated runtime must use measured throughput, worker count
and actual horizon; it is not an assurance of a particular completion time.

## If the candidate pair fails

Keep the negative result. First explore the eight existing deterministic policies
on development seeds, using behavioral restraint and sustainable collective
harvest to select candidate C policies. Freeze selected policies before held-out
validation. Neither training with a common reward nor calling a policy
cooperative substitutes for evaluating individual harvest incentives.

If policy choice is not the problem, diagnose regeneration versus extraction,
occupancy, shared-resource externalities, and horizon/discounting. Change one
mechanism at a time in a separately versioned experiment. A cooperation bonus,
new utility, rest action, or revised gamma changes the scientific model; do not
add one merely to produce a passing diagram. Implementation of such changes is
outside this validation experiment.

Sources: [Leibo et al.](https://arxiv.org/html/1702.03037),
[SocialJax sections 3.1 and 4.3](https://arxiv.org/html/2503.14576v3).

## Capped-harvest variant

The opt-in `reward_mode: "capped_harvest"` uses
`min(actual per-step harvest, metabolism)`. Default `harvest` preserves the
original objective. This changes utility, not physical extraction or reserves.

New runs use `population_payoff_v2`: `return_sum`/`return_discounted` record
actual utility rewards, while `harvest_sum`/`harvest_discounted` record gross
harvest. Utility plus `uncredited_harvest_*` equals gross harvest. Gross sum
matches `wealth_delta`. `late_harvest_rate` stays physical;
`late_utility_rate` reports utility. The analyzer still reads original v1 runs
as harvest-only and verifies their original hashes before applying defaults.

```bash
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
```

The [prospective protocol](reviews/capped-harvest-validation-protocol.md) fixes
seeds, sample size and acceptance. Pilot curves are exploratory. Held-out
endpoints determine adoption, requiring support under both return definitions
in all three ecologies. Numerical bounds within 1e-12 of zero do not establish
positive incentives. Outputs refuse overwrite; use distinct analysis names
when intentionally reanalyzing an existing run.
