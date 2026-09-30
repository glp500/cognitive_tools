# Active plan: Stage-5 visibility study

The [new primary study specification](../docs/specs/visibility-bounded-search-study.md)
supersedes the Stage-4 design for future campaigns. The [Stage-4 design](../docs/ideas/ecology-perception-organization.md)
and [capped-harvest implementation plan](archive/capped-harvest-implementation-plan-2026-09-30.md)
remain available for historical reproduction.

## Phase 1 — Freeze treatment semantics and calibration

Define five versioned visibility profiles, keep every observer's attention at
k=4, and make fixed/adaptive and ecology-paired runs begin from identical graphs
and propensity realizations. Calibrate N=64 networks without learning. Freeze
the JSON profile specification and its hash before interpreting outcomes.

## Phase 2 — Implement and observe complete runs

Add the Stage-5 protocol to the existing experiment CLI with capped-harvest
reward, fixed or one adaptive bounded-search mechanism and B0. Reject incompatible
settings. Record per-agent propensity and realized visibility, graph and profile
hashes, G0/GT, turnover, rewiring and perception diagnostics, evaluation utility,
condition duration/errors and run-health summary. Preserve the archived Stage-4
runner, schema and outputs.

## Phase 3 — Analyze, visualize and execute bounded checks

Use independent replicates for final-window summaries and planned H1/H2/H3
paired contrasts. Add a separate visibility analysis with five main figures,
a campaign runner for 22-condition smoke, 330-condition pilot and 3,300-condition
full run, and documentation with exact commands. Run all tests, calibration and
an end-to-end smoke from clean committed source. The pilot/full computation is
provided as a command for later execution; it is not part of implementation
verification.

[Checklist](todo.md)

## Verification result

The reviewed implementation passed 225 tests, lint, formatting and shell syntax.
Network-only calibration sampled 1,000 networks per profile. The final clean
smoke (`visibility_smoke_v2`, producer `1e5631c`) completed all 11 treatments,
22 conditions, five main figures and H2/H3 contrast outputs. Paired graph and
propensity identities, utility caps, 10 unique initial graph pairings and resume
were checked. The smoke has one ecology, so H1 contrasts are intentionally absent.
The pilot and full campaign are prospective commands in the README.
