# Social-dilemma revision tasks

Status: all required implementation and validation tasks complete.
Latest verification: 209 tests passed; four-treatment smoke and resume passed; 600 development episodes replayed; original v1 run audited.

[Plan](plan.md) · [Specification](../docs/specs/social-dilemma-revision.md) ·
[Evidence](../docs/reviews/incentive-redesign-review-2026-09-29.md)

## Completed for this request

- [x] Review the ecology, actions, allocation, utility, learning, networks, evaluation, analysis and compatibility paths.
- [x] Diagnose the failed collective-harvest condition from the original results.
- [x] Screen existing policies, selected parameter changes, capped utility and an effort-cost alternative on development data.
- [x] Run a 20-replicate/four-focal capped-utility feasibility check across all three ecologies.
- [x] Write the reward contract, claim boundary, prospective validation protocol, cost comparison and ordered implementation tasks.
- [x] Preserve the previous completed validation plan and task history in `tasks/archive/`.

## Validation-only implementation: estimated 4–8 focused hours

- [x] 1. Add opt-in capped reward to EcoEnv and prove physical trajectory equivalence.
- [x] 2. Extend paired rollouts with separate gross-harvest and utility accounts, explicit reward identity and v2 records.
- [x] 3. Support original v1 and new v2 audits, correct plot labels and numerical-zero ties.
- [x] Checkpoint: replay development results; run focused and full tests and Ruff checks.
- [x] 4a. Run the development composition pilot on indices 2100–2119.
- [x] 4b. Freeze the candidate/protocol, 100 held-out replicates and four focal agents before using indices 3000–3099.
- [x] 4c. Run the separate held-out endpoint experiment and publish condition-specific findings, including failures.
- [x] Checkpoint: require the stated population criterion in all three ecologies for each of the two returns before adoption.
- [ ] Optional (not executed): extend to the prespecified held-out interior compositions; do not call reused endpoints independent confirmation.

## Learning integration, conditional on validation: another 8–16 hours

- [x] 5. Forward reward identity through every learning/evaluation environment and record actual evaluation utility.
- [x] 6. Reject mismatched reward modes in analysis, resume and adaptive/R0 schedules; preserve legacy harvest behavior.
- [x] Checkpoint: demonstrate B0, S1 and adaptive/matched-R0 smoke lifecycles, with lint/format and full tests passing.
- [x] 7. Write a separate capped-study recipe and bounded learning pilot protocol; retain the original frozen campaign.

Scientific success is not an implementation checkbox: if the held-out gate
fails, stop adoption and use the plan's diagnostic branch. The opt-in reward and validator are implemented. The held-out gate passed and learning integration is implemented.

Previous completed work:
[validation plan](archive/payoff-validation-plan-2026-09-29.md) ·
[validation checklist](archive/payoff-validation-todo-2026-09-29.md) ·
[original findings](../docs/reviews/payoff-validation-results-2026-09-29.md).
