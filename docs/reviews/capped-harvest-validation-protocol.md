# Capped-harvest validation protocol — 29 September 2026

Frozen before the held-out data are run. This implements the
[approved specification](../specs/social-dilemma-revision.md).

Reward is `min(actual per-step harvest, 0.002)`, with separate physical harvest,
wealth, energy and resource accounts. C=000 and D=111; N=64; all three existing
ecologies; initial resource fraction 0.5; H=1000; gamma=0.95. No training or
network adaptation occurs in this validation. Remaining model settings are
recorded in the explicit configuration and resolved manifest.

Development replay uses indices 2010–2029 and recorded focal IDs. Pilot uses
indices 2100–2119, four sampled focal agents, one assignment, and
k={0,1,16,32,48,62,63}. These results are exploratory. The held-out gate uses
**100 replicates, indices 3000–3099, four focal agents, one assignment, k={0,63}**,
base seed 20260928, and existing independent RNG domains. These seeds have not
been used by the earlier validation or development probes.

The gate contains 3,000 unique episodes. Both utility returns come from the
same rollouts: undiscounted sum and gamma=0.95 discounted sum. For each ecology
and return require G>0 AND E>0 AND (greed>0 OR fear>0), using simultaneous 95%
centered max-standardized bootstrap intervals over the four contrasts × three
ecologies, separately for each return. Use 5,000 resamples, analysis seed 1729.
Average focal/assignment observations within each independent replicate first.
Bounds within 1e-12 of zero are numerical ties. Report every condition.

Adoption requires support in all three ecologies under **both** returns. An
inconclusive or contradicted required condition stops learning integration.
Do not add seeds, change cap, change policy or select a return after observing
the held-out result. New candidates require new development and confirmation
protocols. Reserve indices 4000 onward for future confirmation if needed.

A full held-out seven-composition sweep is optional, not required for adoption.
If executed with the gate seeds, endpoints are duplicated evidence, not new
replication. The minimal figure set consists of exploratory pilot curves and
separate confirmatory endpoint panels. It must not interpolate unmeasured
held-out interior compositions.

Run names: `capped_pilot_v1`, `capped_gate_v1`. Inputs:
`configs/payoff/capped_pilot.json`, `configs/payoff/capped_gate.json`.
Manifests record the resolved configuration, protocol hash, source hashes,
revision, outputs, focal assignments and run purpose. Original harvest-only
experiments remain separate. No inference about learning convergence, strict
prisoner's-dilemma structure, every location, or informational benefit follows
from this gate.
