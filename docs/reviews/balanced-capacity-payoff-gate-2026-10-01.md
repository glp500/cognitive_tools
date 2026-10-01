# Balanced-landscape payoff gate — 1 October 2026

The held-out population-payoff validation **supports the commons-dilemma endpoint criterion** in all three capacity-matched landscapes for both summed and discounted capped-harvest utility. This validates the tested C=always-low and D=always-high policy pair, horizon and reset distribution; it does not establish learning convergence or confirm the visibility hypotheses.

| Landscape | Summed return | Discounted return |
|---|---|---|
| Balanced uniform | supported | supported |
| Balanced dispersed | supported | supported |
| Balanced segregated | supported | supported |

The independent gate used 100 replicates per landscape (replicate IDs 4000–4099), N=64, horizon 1000, four focal agents, C=`000`, D=`111`, compositions 0 and 63 other cooperators, capped-harvest reward, and 5,000 simultaneous-bootstrap resamples. For each landscape and return definition, the lower simultaneous bounds for collective gain, exploitation gap, and fear are positive; greed is zero and is not required because fear is supported. The six verdicts and exact design are verified by `scripts.validate_balanced_gate` before the full learning campaign can start.

The simulation recorded 3,000 physical episodes and 3,000,000 environment steps. Its manifest records a dirty Git worktree because the visibility-analysis rewrite was in progress; the payoff simulation source and frozen input manifest are hashed, and no payoff simulation code was edited during the gate. For maximal archival provenance, a future independent reproduction can rerun the same frozen gate from a clean committed checkout.

The revised ten-replicate balanced **learning pilot remains exploratory**:

| Hypothesis | Pilot estimate | Replicate-bootstrap 95% interval | Interpretation |
|---|---:|---:|---|
| H1: within-run checkpoint correlation in adaptive networks | −0.079 | −0.143 to −0.021 | Opposite the predicted sign; associational only |
| H1 sensitivity: remove each run's linear time trend | +0.036 | +0.011 to +0.063 | Sign reverses; H1 is sensitive to time trend |
| H1 secondary: treatment-centered correlation of run means | +0.094 | −0.101 to +0.294 | Between-run direction uncertain |
| H2: pooled segregated−dispersed local-view error | +0.00114 | −0.00158 to +0.00400 | Direction uncertain |
| H3: pooled adaptive−fixed local-view error | −0.00286 | −0.00500 to −0.00102 | Pilot aligns with prediction; not confirmatory |

These H2/H3 directions were chosen after seeing the pilot; only a new independent full learning campaign can test them prospectively. The primary H1 estimate follows observer-count inequality and error together at training checkpoints within adaptive runs. Its sign reverses when a linear time trend is removed, so the pilot cannot support a stable directional H1 interpretation. Neither H1 estimate is a causal effect of visibility. The [revised prospective specification](../specs/capacity-matched-visibility-study.md) defines the estimands and outcomes.

[Payoff verdict report](../../results/payoff_validation/balanced_gate_v1/analysis/validation_report.md) · [Gate verdicts](../../results/payoff_validation/balanced_gate_v1/analysis/verdicts.csv) · [Longitudinal pilot H1](../../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/data/h1_checkpoint_association_summary.csv) · [Run-average pilot H1](../../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/data/h1_association_summary.csv) · [Pilot H2/H3](../../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/data/pooled_hypothesis_summary.csv) · [Five story figures](../../results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/story_candidates/story_manifest.json)
