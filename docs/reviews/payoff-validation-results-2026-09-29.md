# Population payoff validation results — 29 September 2026

**The always-low/always-high candidate pair does not meet the operational population social-dilemma criterion in any of the three tested ecologies at H=1000.** This holds for both discounted harvest (gamma=0.95) and undiscounted harvest. The failed necessary condition is collective advantage: everyone always-low earns less harvest than everyone always-high.

This is a policy-pair result, not proof that the environment contains no sequential social dilemma for other policies, starting states or horizons. No ecological mechanism, action, reward or learner setting was changed.

## Design and implementation

The independent runner reuses EcoEnv and the existing scenario construction. It records each actual individual reward, matched C/D focal comparisons, complete co-player permutations, per-agent returns, locations, diagnostics, source hashes and output hashes. It accepts all eight deterministic policies over the existing resource bins.

- Pilot: 10 seeds per ecology, two focal agents, seven compositions; 780 episodes and 780,000 environment steps. Runtime 453.85 seconds with four workers.
- Held-out endpoint validation: 100 new seeds per ecology (replicates 1000–1099), one focal agent, k=0 and 63; 1,200 episodes and 1.2 million environment steps. Runtime 255.56 seconds with 14 workers.
- All-C/all-D collective returns use all 64 agents; reducing focal sampling does not reduce their precision. Focal-dependent conditions are less precise than the proposed eight-focal main sweep.
- Decision uncertainty: 5,000 replicate-block bootstrap samples; simultaneous intervals across four contrasts and three ecologies, separately per return definition.
- The endpoint gate was [specified before held-out outcomes](payoff-validation-protocol-2026-09-29.md). The larger 48,600-episode interior sweep was not executed because the necessary collective condition failed. The pilot supplies exploratory interior curves.

## Collective advantage: all-C minus all-D

| Ecology | Harvest definition | Difference | Simultaneous 95% interval |
|---|---|---:|---|
| uniform_high | sum | -0.334969 | [-0.441163, -0.228774] |
| patchy_high | sum | -0.332868 | [-0.441732, -0.224005] |
| split_high_low | sum | -1.064536 | [-1.250546, -0.878525] |
| uniform_high | discounted | -0.260863 | [-0.263039, -0.258687] |
| patchy_high | discounted | -0.260037 | [-0.262320, -0.257754] |
| split_high_low | discounted | -0.195930 | [-0.199551, -0.192309] |

All collective-advantage intervals are below zero. Exploitation loss, greed and fear are supported in all three ecologies for both return definitions; their presence cannot compensate for failed collective advantage.

## Why resource preservation did not establish cooperation

| Ecology | All-low total harvest | All-high total harvest | All-low reserve welfare | All-high reserve welfare |
|---|---:|---:|---:|---:|
| uniform_high | 2.000 | 2.335 | 99.8% | 62.8% |
| patchy_high | 2.000 | 2.333 | 99.8% | 62.7% |
| split_high_low | 1.772 | 2.836 | 91.1% | 59.8% |

Reserve welfare and resources improve under restraint, but agents optimize harvest. The validation therefore identifies a mismatch between the candidate cooperation label and the measured utility, rather than a failure to implement social interaction.

In the uniform and patchy ecologies, late-window all-low harvest is 0.002 per agent per step, versus about 0.00146–0.00147 for all-high. This suggests horizon sensitivity worth investigating, but does not establish an eventual crossover. In the split ecology, all-high late-window harvest remains higher (about 0.00231 versus 0.00173). Longer rollouts do not change a gamma=0.95 agent's discounting.

## Next research step

Screen the eight existing resource-dependent policies on development seeds for sustainable collective harvest and behavioral restraint. Fix the policy pair before a new held-out evaluation. Examine horizon dependence separately. Only if the policy search and ecological diagnosis justify it should a new experiment revise extraction, regeneration, competition exposure or utility. Do not change rewards merely to make the diagram pass.

## Verification and artifacts

- Full repository suite: 183 tests passed. Final analysis/audit refinements: nine focused analysis tests passed. Ruff lint/format and diff whitespace checks passed.
- Six saved campaign endpoint controls reproduced to numerical tolerance; serial and parallel outputs have identical data hashes.
- Tests cover branch independence, reward/wealth accounting, gamma timing, multiple horizons, invalid settings, missing pairs, altered files, synthetic verdicts and within-block pseudoreplication.
- Exported PNG figures were visually inspected. Endpoint-only plots do not draw lines across unmeasured interior compositions.
- Completed campaign files and existing model/training code were not modified. Pre-existing presentation edits remain outside this work.

[Usage and commands](../payoff-validation.md) · [Notion plan](https://app.notion.com/p/3ea91786f46281db8d8ec3cac9e93b75)

[Pilot Schelling curves (discounted)](../../results/payoff_validation/pilot_2026-09-29/analysis/schelling_discounted_1000.pdf) · [Pilot curves (undiscounted)](../../results/payoff_validation/pilot_2026-09-29/analysis/schelling_sum_1000.pdf)

[Held-out report](../../results/payoff_validation/heldout_endpoints_2026-09-29/analysis_final/validation_report.md) · [Endpoint outcomes](../../results/payoff_validation/heldout_endpoints_2026-09-29/analysis_final/endpoints_1000.pdf) · [Contrasts CSV](../../results/payoff_validation/heldout_endpoints_2026-09-29/analysis_final/contrasts.csv)

Analysis outputs in `analysis_final` supersede the first held-out figure exports in `analysis`, which are retained for provenance. The data and numerical results are unchanged. Bootstrap inference is approximate, population averages can conceal regional asymmetry, and neither learning convergence nor informational benefits were tested.
