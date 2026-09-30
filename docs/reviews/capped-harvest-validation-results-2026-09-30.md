# Capped-harvest validation results — 30 September 2026

**The frozen capped-harvest candidate passed the held-out population criterion
in all three ecologies under both summed and discounted utility.** Learning
integration therefore proceeded. The original gross-harvest result is unchanged.

The revision is `reward = min(actual per-step harvest, 0.002)`. It changes the
objective to satiating current-harvest utility; it does not change physical
harvest or ecology and is not consumption from stored reserves or survival.

## Protocol and evidence

The [frozen protocol](capped-harvest-validation-protocol.md) specifies N=64,
H=1000, gamma=0.95, C=always-low and D=always-high. The endpoint gate used
100 independent replicates per ecology (indices 3000–3099), four paired focal
agents, and 3,000 episodes. Simultaneous 95% centered max-standardized bootstrap
intervals use 5,000 resamples (seed 1729), with a 12-contrast family separately
for each return and numerical-zero tolerance 1e-12.

Acceptance requires collective gain G>0, exploitation gap E>0, and either greed
or fear >0. G compares all-C and all-D population means; E=C(63)-C(0);
fear=D(0)-C(0). All three are supported in every ecology/return cell:

| Ecology | Return | G, simultaneous CI | E, simultaneous CI | Fear, simultaneous CI |
|---|---|---|---|---|
| uniform_high | sum | 1.074528 [1.028642, 1.120414] | 1.312056 [1.222826, 1.401286] | 0.237162 [0.169868, 0.304456] |
| uniform_high | discounted | 0.001283 [0.001136, 0.001430] | 0.003914 [0.003213, 0.004614] | 0.002553 [0.001940, 0.003166] |
| patchy_high | sum | 1.077660 [1.031844, 1.123476] | 1.311967 [1.221877, 1.402057] | 0.233669 [0.167960, 0.299378] |
| patchy_high | discounted | 0.001281 [0.001137, 0.001424] | 0.003960 [0.003260, 0.004660] | 0.002620 [0.002006, 0.003234] |
| split_high_low | sum | 0.882473 [0.831061, 0.933885] | 1.078222 [0.978777, 1.177668] | 0.239446 [0.175387, 0.303505] |
| split_high_low | discounted | 0.003883 [0.003576, 0.004189] | 0.007616 [0.006400, 0.008831] | 0.003494 [0.002481, 0.004507] |

Greed is exactly tied in uniform and patchy ecologies. In split high/low it is
supported for summed utility (0.050163 [0.029222, 0.071104]) but inconclusive
for discounted utility (-0.000010 [-0.000048, 0.000027]). Thus the common evidence
is fear-driven; this is not a strict Prisoner's Dilemma claim. Relative to the
maximum discounted return of 0.04, collective gains are about 3.21%, 3.20% and
9.71%, respectively. The summed-return maximum is 2.0.

## Artifacts and provenance

- [Gate report](../../results/payoff_validation/capped_gate_v1/analysis/validation_report.md),
  [contrasts](../../results/payoff_validation/capped_gate_v1/analysis/contrasts.csv),
  [normalized effects](../../results/payoff_validation/capped_gate_v1/analysis/normalized_contrasts.csv).
- [Gate Schelling diagram](../../results/payoff_validation/capped_gate_v1/analysis/schelling_discounted_1000.png)
  shows measured endpoints only; unmeasured interior compositions are not interpolated.
- [Exploratory composition diagram](../../results/payoff_validation/capped_pilot_v1/analysis_final/schelling_discounted_1000.png)
  covers k=0,1,16,32,48,62,63 with 20 development replicates (2100–2119).
- [Endpoint utility and ecology plot](../../results/payoff_validation/capped_gate_v1/analysis/endpoints_1000.png).

The pilot and gate each executed 3,000 episodes, taking 708.06 and 666.35 seconds
respectively with 14 workers. Before those runs, 600 development episodes were
replayed through the implemented reward API and matched the development probe.
The original schema-v1 gross-harvest run also passed the compatibility audit.

The gate was produced from clean commit
`9c29b9c5150e9bbf2311fda2d8481b02f4c3e6cc` with reward semantics bound into the
protocol hash. The pilot began at `1ecec47` before that hash fix. Its manifest
received an explicit metadata-only amendment: exact original manifest bytes
are preserved as `manifest.before_protocol_hash_review.json`, and the amendment
records their SHA, old/new protocol hashes, time and reason. Config, reward
definition, producer revision/source hashes, output hashes and episode data were
unchanged. This pilot is exploratory, not independent confirmation.

Results directories are generated local artifacts ignored by Git; this review
is the tracked numerical summary. The optional 15,000-episode held-out interior
extension was not executed. No framework migration was necessary.

## Interpretation and next step

The validated claim is a population social dilemma for this policy pair, reset
distribution, ecology configuration, horizon and utility. It does not quantify
all possible policies or prove learned cooperation, social-perception benefits,
or strict dominance. New training is necessary under the changed objective.
See the [separate learning-study recipe](../capped-harvest-study.md) for the
bounded smoke test and prospective pilot; the full learning campaign is outside
this implementation.
