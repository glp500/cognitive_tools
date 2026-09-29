# Incentive redesign review — 29 September 2026

## Recommendation

**Keep the current codebase and add an opt-in capped-harvest reward variant.**
Reward each agent with `min(actual_harvest, metabolism_rate)` per step, with the
existing rate 0.002. Validate this variant independently before retraining the
larger social-perception campaign. This is the lowest engineering-cost candidate
with positive development evidence from the alternatives examined here.

The qualification matters: this changes the utility from accumulated harvest
to satiating current-harvest utility. It is appropriate if the follow-up studies
competition for meeting a recurring need. It is not an invisible fix to the
original wealth-maximization model, nor a claim that the original experiment
already meets the social-dilemma criterion.

Estimated focused engineering effort is **4–8 hours for the validation-only
slice**, then **another 8–16 hours for learning, evaluation and compatibility
integration** if the gate passes: roughly 1.5–3 working days total. These are
planning estimates for someone familiar with this repository, not measured
delivery guarantees; scientific iteration and full-campaign runtime are extra.

[Implementation specification](../specs/social-dilemma-revision.md) ·
[Ordered plan](../../tasks/plan.md) · [Task checklist](../../tasks/todo.md).

## What the current model establishes

The project is already a sequential multiagent social/resource game: actions
affect shared resources, future states and other agents' opportunities; agents
have partial observations and individual rewards. The missing evidence concerns
the *incentives for a particular policy pair*, not whether interactions occur.
The [original SSD definition](https://arxiv.org/html/1702.03037) is policy- and
return-dependent; a sequential game is not automatically a sequential social
dilemma. The [SocialJax paper](https://arxiv.org/html/2503.14576v3) provides the
population/Schelling-diagram context for the existing operational check.

The [completed validation](payoff-validation-results-2026-09-29.md) found that
always-low earns less harvest than always-high when everyone uses the same
policy. At H=1000, gamma=0.95, collective gaps are -0.26086, -0.26004 and
-0.19593 across uniform, patchy and split ecologies. All three undiscounted
gaps are also negative. Better resource stocks and reserves cannot substitute
for better individual utility when reward is gross harvest.

Several structural facts explain the result:

- High extraction requests ten times as much harvest (0.020 versus 0.002).
  Agents are rewarded for every unit, including surplus beyond metabolic need.
- Gamma=0.95 heavily weights the first few dozen steps; high extraction captures
  initial stock before many future depletion losses matter. A longer rollout
  alone does not change these preferences.
- Always-low is an extremely restrained candidate, not an optimized sustainable
  policy. In the split ecology, even its late harvest rate trails all-high.
- Positions are fixed and allocation is prorated among co-located agents.
  Diffusion also couples cells. Resource competition and information networks
  are distinct mechanisms: changing rewiring cannot directly repair a failed
  fixed-policy payoff condition.

## Scope of the codebase review

Reviewed the active package's ecology-to-analysis path, configuration and
campaign entry points, existing test contracts, documentation and result
provenance at revision `6835e8aecee129266ec2cfc7b23aebdebabfd1c4`.
This is an architecture and incentive review, not a claim of exhaustive formal
verification of every line. Presentation artifacts are not simulation inputs.

| Area | Finding and consequence for the cheapest change |
|---|---|
| `ecology.py`, `scenarios.py` | Central post-harvest resource equation and three supported spatial ecologies. Keep equations and maps; no port is required. |
| `model.py` | Prorated physical allocation, gross wealth and stored-energy welfare are separate state updates. Keep these physical accounts. |
| `env.py` | Reward is currently wealth increment; `infos.harvested` aliases reward. Transform reward here, but first separate actual harvest from utility. |
| `qlearning.py` | The tabular update already consumes numerical rewards. No new learner or state space is needed. |
| `social.py` | Observes previous actions, computes forecast error, and rewires attention. These mechanisms can be retained. |
| `experiment.py` | Training consumes rewards correctly, but evaluation discards them. Record evaluation utility explicitly; forward reward mode through all environments. Validate before the no-social early return. |
| `payoff.py` | Existing paired runner is reusable. Its reward-sum-equals-wealth assertion must become separate utility and physical-harvest audits. |
| `payoff_analysis.py` | Existing paired contrasts/bootstrap remain useful. Schema reader and harvest-specific labels need reward-aware handling; numerical ties must not become positive greed. |
| `analysis.py`, `focused_analysis.py` | Add reward identity to compatibility checks. Preserve original focused outcomes; expose new utility outcomes under a separate study identity. Keep wealth Gini physical. |
| R0 schedules and condition resume | Current metadata is insufficient to distinguish reward variants. Include reward identity and rebuild schedules from new adaptive training. |
| `configs/payoff/`, campaign script | Add separate validation configs first. Preserve the frozen campaign recipe; do not immediately launch its full grid. |
| `tests/`, `pyproject.toml`, CI | Existing ecology, learner, lifecycle, pairing and reproducibility coverage is reusable. Add focused reward/accounting tests; no dependencies or CI changes are needed. |
| README and research docs | Current aims concern perception and organization. The new reward does not guarantee that better information improves outcomes; retain that distinction. |

The main hidden labor is accounting and provenance, not the reward formula.
Changing one line in `env.py` without repairing these paths would produce
misleading harvest labels, invalid validator assertions and incomplete learned
policy evaluation.

## Development experiments performed for this review

No production module, reward default or original campaign was modified.
Scratch simulations used the existing environment and fixed policies, then
computed candidate utility from each actual per-step harvest. This is exact
for evaluating those fixed policies under a reward-only revision because the
reward does not change their actions. It does not simulate retraining.

### Cheap alternatives screened

Initial screen: 153 episodes, 30.1 seconds with 14 workers; three replicates
per ecology (indices 2000–2002), H=1000. This is a small development screen,
not an exhaustive parameter search or a rejection of every possible policy.

| Candidate | Observed evidence | Decision |
|---|---|---|
| Existing eight deterministic resource-bin policies | None had a larger homogeneous discounted harvest than always-high in the three-replicate means. Policies 110 and 111 tied; they differ only in the abundant state. | A simple policy substitution did not solve the tested collective gap. This did not exhaust all ordered C/D pairs, stochastic, history-dependent or learned policies. |
| Raise low extraction to 0.004, 0.006 or 0.010 | All-low discounted harvest remained below the unchanged all-high control in all three ecologies. | A few-hour configuration change has no supporting evidence from this grid. |
| Lower initial stock fraction from 0.50 to 0.10 or 0.05 | All-low remained below all-high at gamma=0.95 in each ecology. | Reducing the initial windfall alone was insufficient at these settings. |
| Raise gamma to 0.99 or 0.995 | Original all-low/all-high homogeneous gaps remained negative at H=1000 in each ecology. | This changes preferences and requires retraining, without a demonstrated fix here. |
| Incremental high-action effort cost | Exact repricing of saved fixed-policy results can satisfy the discounted criterion, but the example cost fails the undiscounted split-ecology conflict condition. | More arbitrary calibration and weaker horizon robustness than the capped candidate. |
| Cap current-harvest utility at 0.002 | Initial screen suggested collective benefit and fear-driven conflict at both return definitions; larger check below supports feasibility. | Recommended for prospective validation, subject to its changed utility interpretation. |

This screen does not prove that parameter-only redesign is impossible. It makes
an open-ended parameter search a less predictable labor investment than the
bounded reward variant. No claim is made that this is the global cheapest
solution across all imaginable models.

The same homogeneous outcomes also screen all 19 distinct pairs with C's action
no higher than D's in each of the three resource bins. None has a positive
collective gap above numerical tolerance in any ecology's development mean at
gamma=0.95. This narrows the simple policy-only route without requiring mixed
rollouts for pairs already missing a necessary condition; it does not cover
policies whose restraint depends on history or social state.

### Larger capped-harvest feasibility check

Used 20 independent starting-state replicates per ecology, indices 2010–2029,
four uniformly sampled focal agents without replacement, and matched all-C,
all-D, one-C and one-D branches. Each homogeneous endpoint uses every agent.
Total: **600 episodes, 121.6 seconds with 14 workers**. Base seed 20260928;
landscape seed=base+replicate; placement seed=base+100000+replicate. Focal IDs
come from NumPy RNG seed 20260929+replicate and are shared across ecologies.
All other settings are the existing `PayoffConfig` defaults.

Intervals below use 5,000 replicate-block bootstrap resamples, seed 1729, and
simultaneous adjustment across 12 contrasts per return family. They quantify
variation in this development sample; candidate selection means they are **not
confirmatory evidence**. Greed is unnecessary when fear is supported.

| Ecology / return | Collective G [95% interval] | Exploitation E [95% interval] | Fear [95% interval] |
|---|---:|---:|---:|
| Uniform, discounted | 0.001172 [0.000853, 0.001492] | 0.003127 [0.001622, 0.004632] | 0.002275 [0.000939, 0.003610] |
| Patchy, discounted | 0.001183 [0.000857, 0.001509] | 0.003388 [0.001774, 0.005002] | 0.002495 [0.001039, 0.003952] |
| Split, discounted | 0.004293 [0.003543, 0.005042] | 0.007107 [0.004286, 0.009927] | 0.004134 [0.001454, 0.006814] |
| Uniform, sum | 1.0302 [0.9263, 1.1341] | 1.2083 [1.0222, 1.3944] | 0.2617 [0.0978, 0.4256] |
| Patchy, sum | 1.0344 [0.9295, 1.1394] | 1.2022 [1.0166, 1.3878] | 0.2489 [0.0839, 0.4139] |
| Split, sum | 0.8270 [0.7152, 0.9387] | 1.1363 [0.9786, 1.2939] | 0.2108 [0.0483, 0.3732] |

Discounted G is about 2.9%, 3.0% and 10.7% of maximum capped return 0.04.
The uniform/patchy greed contrast is exactly zero; split greed is inconclusive
at both return definitions. Thus this is a candidate **fear-driven population
dilemma**, not evidence of strict dominance of defection or a prisoner's dilemma.

The mechanism has a concrete interpretation. When everyone is restrained,
current needs can be supplied repeatedly. Under scarcity, a high request can
secure a larger prorated share against other high requests; being the lone
restrained agent can be costly. When resources are abundant, surplus extraction
does not generate unlimited immediate utility. The measured conflict comes
from the environment's allocation and depletion dynamics, not an action-label
bonus.

Passing this structural test would not ensure learned cooperation. Abundant-state
utility ties and fear-driven incentives can leave high-extraction behavior
attractive or make learning sensitive to experience. Check that in the bounded
learning pilot; do not require the information treatments to produce a benefit.

## Why the effort-cost alternative is not first choice

Consider `u = harvest - c * 1[high action]`. For constant C/D policies define
A=sum(gamma^t), t=0..999. Exactly, G becomes G+cA, E stays unchanged, and both
greed and fear decrease by cA. This permits a cheap algebraic feasibility check.

At c=0.014, discounted G becomes 0.01914, 0.01996 and 0.08407; discounted greed
remains positive in all ecologies, including positive repriced simultaneous
lower bounds. But undiscounted split-ecology greed becomes -1.68166 and fear
-11.07445. The model would not meet our endpoint criterion there. Already at
the point estimates, the cost needed for all discounted collective gaps exceeds
the cost preserving either split-ecology undiscounted endpoint conflict.

Moreover 0.014 is a substantial assumed effort cost relative to high harvest
0.020, paid even if no resource is obtained. Selecting it from the old held-out
data turns those data into development evidence for the new model. It needs
an economic interpretation and new confirmation; subtracting an arbitrary
action penalty solely to obtain a label is not validation.

## Relative labor and migration decision

| Route | Indicative labor | Scientific tradeoff |
|---|---|---|
| Change policy/config only | Hours if a candidate works; search duration unknown | Preserves harvest utility, but the bounded screens above did not identify a solution against always-high. |
| Capped reward, separate validator | 4–8 focused hours | Small code change plus necessary accounting; positive development evidence at both return definitions. |
| Adopt capped reward across learning/analysis | Another 8–16 hours after the gate | Retains current ecology, learner and networks; requires retraining and a changed utility interpretation. |
| Change extraction/recovery/sharing mechanics | Several days including design and testing; uncertain iterations | May preserve gross-harvest objective but changes ecology and needs new calibration. |
| Port the study to SocialJax or Melting Pot | Budget roughly 1–3 weeks for a minimal study port, with high uncertainty | Changes environment/action semantics and requires observation, behavior-label and analysis adaptation; still needs policy-level validation. |

Migration estimates are architectural judgments, not timed installations or
benchmarks. The existing [platform review](full-run-review-2026-09-29.md) and
official [SocialJax repository](https://github.com/cooperativex/SocialJax) and
[Melting Pot Harvest configuration](https://github.com/google-deepmind/meltingpot/blob/main/meltingpot/configs/substrates/commons_harvest__open.py)
show that migration entails more than changing an environment import. Choose
one later for benchmark comparability or acceleration if those are requirements;
neither name certifies the incentives of a modified policy/observation experiment.

## Bounded fallback if utility must remain gross harvest

Do not implement the capped variant if accumulated material extraction is the
essential individual objective. First finish a bounded policy-pair search over
the existing eight policies on development seeds, requiring C to represent
meaningful restraint and testing matched unilateral returns rather than only
homogeneous ranking. Our initial screen compared homogeneous outcomes against
always-high; it did not establish that no other ordered pair works.

If that search fails, examine one ecological lever at a time: extraction relative
to local net regeneration, then resource-sharing exposure. A useful scale is
local maximum net growth r*K*q^2/4, about 0.00459 per isolated uniform-high cell
per step, before coupling and actual update-order effects. Demand depends on
occupancy, so setting a harvest amount equal to this number is not a proof of
sustainability. A larger low request can also destroy the benefit it was meant
to capture. Increasing coupling can rescue high extractors as well as spread
depletion; do not assume it creates a dilemma monotonically.

Set the development grid and compute budget before searching. Retain only
candidates with collective benefit and matched unilateral conflict at the
intended learner discount. Freeze one candidate and use unused confirmation
seeds. If none qualifies within the budget, report the current game honestly
or undertake a deliberately larger redesign; do not keep moving the criterion.

## Artifacts and reproduction

The [development artifact directory](../../results/review/incentive_design_2026-09-29/)
contains the two scratch scripts, original screen rows, per-episode and
per-replicate capped results, bootstrap summary, source hashes/configuration,
and effort-cost repricing. Generated results are ignored by Git; the central
numeric evidence is preserved in the table above. The scratch scripts are
analysis tools, not the proposed production implementation.

To reproduce the development simulations from the repository root with its
dependencies installed (scripts overwrite their own review outputs):

```bash
python results/review/incentive_design_2026-09-29/development_probe.py
python results/review/incentive_design_2026-09-29/capped_design_probe.py
```

The local run used `/tmp/cognitive-tools-venv/bin/python`; the shell's default
Python lacked NumPy. The capped probe reuses the existing simultaneous-interval
function. No new production tests were run for this documentation-only proposal;
the previous implementation reported 183 passing tests. Implementation-specific
acceptance and fresh held-out validation remain outstanding, as recorded in
the task checklist.
