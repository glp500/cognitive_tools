# Ecology, Social Perception, and Collective Organization in Artificial Societies

Status: confirmed by the project owner on 2026-09-28. This document records the
agreed scope. The implementation now follows this scope; campaign freeze and
execution records are written under `results/q_learning_baseline/campaigns/`.

## Problem statement

How can a bounded artificial-society experiment explain how ecological
conditions and individual social observations generate population misperception,
collective organization, and resource and welfare outcomes?

## Recommended direction

Make ecology and population perception the starting point, with collective
organization and welfare consequences following. Cognitive tools and social
cybernetics provide the broader motivation; the implemented study concerns
restricted social observation and adaptive information-source replacement.

Use the existing model and treatments, simplify the full campaign, and organize
analysis around two core questions. Adaptive rewiring is a mechanism to compare,
not a mechanism that must prove beneficial. Success includes harmful, negligible,
or uncertain effects when the planned comparisons are reported clearly.

## Research questions and hypotheses

1. **RQ1 — Perception:** Under which ecological conditions do local observations
   misrepresent population-level behavior?
2. **RQ2 — Organization:** How do ecology and observation rules shape visibility
   concentration and population extraction behavior?
3. **Secondary question:** What consequences do these conditions have for
   resource sustainability, individual welfare, and inequality?

- **H1 — Ecological dependence:** Population misperception differs across
  ecological configurations. This is nondirectional; no universal ranking is
  justified by the current evidence.
- **H2 — Search scope:** Local replacement produces greater visibility
  concentration than global replacement. This prediction is informed by related
  work and the pilot, rather than independent of them.
- **H3 — Adaptive selection:** Surprise-triggered rewiring changes population
  misperception relative to equally frequent random turnover. Improvement is
  not assumed.

Whether concentrated visibility increases majority illusions, or whether
misperception accompanies worse resource and welfare outcomes, remains an
explanatory analysis. Associations do not establish causal mediation.

## Minimum experiment scope

| Component | Agreed scope |
|---|---|
| Ecology | Uniform high, patchy high, split high/low |
| Population | 64 agents; existing N=32 pilot results remain supporting evidence |
| Learning | Existing independent tabular Q-learning, actions, rewards, and ecological dynamics |
| Reference treatments | B0: ecological information only; S1: fixed random observation; S2: fixed uneven visibility |
| Rewiring treatments | Local, mixed, and global adaptive replacement, each with its own matched random-turnover control |
| Rewiring probability | mu = 0.10 only; remove the planned full-campaign sweep over 0.05 and 0.20 |
| Duration | 5,000 training and 1,000 evaluation steps |
| Replication | 100 independent replicates per condition, fixed before execution; a practical precision target, not a guarantee of decisive results |
| Evaluation | Frozen policies and terminal networks in a fresh ecological reset |

Total: **27 conditions × 100 replicates = 2,700 training runs**. Use a documented
seed set distinct from the pilot, preserving matching across treatments.

Ecological comparisons concern whole configurations: the split scenario changes
more than spatial heterogeneity. S2 differs in attention capacity as well as
visibility. Neither comparison isolates a single factor.

## Measurements and analysis

| Role | Measures | Use |
|---|---|---|
| Primary: perception | Mean absolute population-perception error; majority-mismatch rate | Answer RQ1; report tie frequency alongside majority mismatch |
| Primary: organization | Visibility Gini; population low-extraction share | Answer RQ2 with training trajectories and a fixed final training window |
| Secondary: consequences | Mean resource fraction; mean reserve welfare; final wealth Gini | Measure outcomes during frozen evaluation |

Population-perception error compares an agent's observed low-extraction fraction
with its prevalence among all other agents. Majority mismatch means opposite
observed and population majorities; ties are reported separately. Visibility
concentration alone is not evidence of a majority illusion, cooperation, or
collective intelligence.

Use the **final 1,000 training steps** for primary perception and organization
summaries. Full training trajectories show development. Summarize within each
replicate first: agents and timesteps are not independent replicates.

Keep three comparison families:

1. **Ecology:** compare the three configurations within each social treatment.
2. **Observation structure:** compare S1 and S2 descriptively, and
   local/mixed/global replacement under both adaptive and random turnover.
3. **Adaptive selection:** compare each adaptive treatment with its matched
   random control.

B0 anchors resource and welfare outcomes; social-perception measures are
undefined there. Report effect sizes and replicate-level uncertainty
consistently. Formal significance claims across multiple contrasts require a
stated multiplicity adjustment; avoid isolated significance claims.

## Four main figures

1. **Mechanism and design:** ecology, observation, learning, and comparisons.
2. **Population perception:** perception error and majority mismatch across
   ecologies and observation rules.
3. **Collective organization:** visibility concentration and extraction behavior
   over training.
4. **Consequences:** resource, welfare, and inequality differences, emphasizing
   matched comparisons.

Additional diagnostics enter supporting outputs only when they resolve a
specific interpretation.

## Key assumptions and interpretation limits

- **Ecological configurations generate distinguishable patterns.** Evaluate
  the planned between-configuration contrasts; allow negligible or uncertain
  differences rather than redesigning the model to obtain separation.
- **Local observation can misrepresent population behavior.** Measure error,
  majority mismatch, and ties directly; allow majority illusions to be uncommon
  or absent.
- **Surprise and population misperception are different.** The rewiring trigger
  measures prediction error about observed peers, not their representativeness,
  competence, or sustainability. Examine their relationship without equating
  them or assuming beneficial replacement.
- **Measured organization is bounded in meaning.** Visibility concentration and
  extraction prevalence describe network and behavioral patterns, not proof of
  consensus, power, or causal mediation.
- **Inference is model-specific.** Results concern the specified artificial
  society and ecological configurations, not cognitive tools in general.

## One bounded implementation pass

1. Align the README and experiment specification with these questions and
   definitions.
2. Restrict the full campaign to this grid and a separate full-experiment seed
   set.
3. Adapt existing analysis to produce the specified summaries and four figures.
4. Run existing checks and one small end-to-end campaign. Add a targeted check
   only for genuinely new analysis logic.
5. Freeze the code revision, configuration, seeds, and analysis specification;
   then execute the full campaign.

Retain functioning diagnostics and supported code unless they obstruct this
workflow. Another broad refactor is outside the agreed pass.

## Not doing

- New cognitive-tool types, agent abilities, institutions, or learning
  algorithms — these change the scientific object and expand development.
- Additional ecological scenarios, parameter optimization, or population-size
  scaling — these widen the question beyond the agreed comparison.
- Causal mediation — the proposed comparisons do not identify the entire
  perception-to-outcome pathway.
- Guaranteed majority illusions or improved sustainability — absence and null
  findings are valid outcomes.
- Network-reset evaluation as another main question — retain it as supplementary.
- Repeated general audits or broad refactoring — verification is limited to the
  existing checks, necessary new analysis checks, and one end-to-end smoke run.

## Freeze details to record

During the bounded implementation pass, record the exact distinct seed set,
summary-window boundary convention, interval procedure, and any multiplicity
procedure used for formal claims. These are operational details within the
agreed design, not invitations to expand its scope.

## Completion rule

The project is complete when the fixed campaign and planned analysis answer the
questions, including small effects, absent majority illusions, or inconclusive
differences. Results alone will not trigger model redesign or additional sweeps.

## Existing project references

- [Experiment specification](../experiment.md)
- [Analysis contract](../analysis.md)
- [Scientific and implementation provenance](../provenance.md)
- [Bibliography](../references.bib)

These describe the current implementation; this confirmed proposal defines the
scope of the pending alignment pass.
