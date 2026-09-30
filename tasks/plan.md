# Social-dilemma revision implementation plan

Status: stages 1–6 implemented; held-out gate passed; stage 7 smoke verification pending. Date: 2026-09-30.

The requested review and specification are complete. Validation implementation
and the pilot are complete; the held-out gate passed all six ecology/return cells. The completed earlier validation plan is preserved in
[the archive](archive/payoff-validation-plan-2026-09-29.md).

[Evidence and alternatives](../docs/reviews/incentive-redesign-review-2026-09-29.md) ·
[Specification](../docs/specs/social-dilemma-revision.md) · [Checklist](todo.md)

## Decision and cost

Retain the codebase and introduce an opt-in `capped_harvest` utility:
`min(actual per-step harvest, metabolism_rate)`, with cap 0.002 for this study.
The default remains gross harvest. Development results support a fear-driven
population dilemma in all three ecologies under both gamma=0.95 and raw H=1000
returns. The subsequent held-out gate supports this criterion; see the
[results](../docs/reviews/capped-harvest-validation-results-2026-09-30.md).

Budget 4–8 focused engineering hours through the independent validation slice,
then another 8–16 hours to integrate learning and compatibility if it passes.
These estimates exclude a new full learning campaign and scientific revisions.
The minimal confirmation gate is 3,000 episodes. The development probe took
121.6 seconds for 600 episodes with 14 workers and lighter output; allow roughly
10–20 minutes for the gate including full recording and analysis, then measure
actual throughput. The optional seven-composition run is 15,000 episodes;
do not purchase that compute before the gate warrants it.

## Dependency order

```text
Reward contract (1) -> Paired runner/accounting (2) -> Audit/analysis (3)
                                                     |
                                         Pilot and held-out gate (4)
                                                     |
                                       only after scientific support
                                                     |
Learning/evaluation utility (5) -> Compatibility and R0 pairing (6)
                                                     |
                                    New-study smoke/pilot recipe (7)
```

Stages 1–4 deliver the separate validation experiment. Stages 5–7 adopt it into
the research workflow. No framework migration, learner rewrite, new action or
ecological refactor is required.

## 1. Reward mode with physical equivalence

Files: `cognitive_tools/env.py`, new `tests/test_reward_modes.py`.
Scope: small, two files. Dependencies: none.

- Add validated `reward_mode`, default `harvest`; capped mode uses positive finite metabolism.
- Keep `infos.harvested` physical; expose utility and uncredited surplus separately.
- Verify fixed actions give identical resources, wealth, energy, observations and randomness in both modes; hand-check cap/scarcity examples and full-energy/zero-harvest behavior.

Verification: `python -m pytest -q tests/test_reward_modes.py tests/test_ecology.py`.
Checkpoint: the physical model and legacy reward are reproduced exactly.

## 2. Auditable independent utility rollouts

Files: `cognitive_tools/payoff.py`, `tests/test_payoff.py`.
Scope: small, two files. Dependency: 1.

- Forward reward mode; accumulate actual utility, gross harvest and uncredited surplus separately for both returns and late-window rates.
- Write explicit v2 reward semantics, hash and mode-specific tail bounds; retain paired reset/assignment behavior and overwrite refusal.
- Test per-step capping versus invalid aggregate capping, wealth/gross equality, utility-plus-surplus accounting, discount timing and serial/parallel equality.

Verification: `python -m pytest -q tests/test_payoff.py tests/test_reward_modes.py`.

## 3. Reward-aware audit and Schelling outputs

Files: `cognitive_tools/payoff_analysis.py`, `tests/test_payoff_analysis.py`.
Scope: small, two files. Dependency: 2.

- Read both original v1 harvest and v2 utility runs, checking stored hashes before defaults; audit separate accounts and reject malformed modes/data.
- Label utility and harvest distinctly, reuse population contrasts/block inference, and handle numerical-zero ties at 1e-12.
- Verify synthetic dilemmas, no-conflict cases, legacy results and tampered accounting; inspect figures.

Verification: `python -m pytest -q tests/test_payoff_analysis.py tests/test_payoff.py`.
Checkpoint: replay the development probe and compare its per-replicate results;
run the existing full test suite before new scientific runs.

## 4. Separate prospective validation

Files: new `configs/payoff/capped_pilot.json`, `capped_gate.json`,
`capped_curves.json`; `docs/payoff-validation.md`; new
`docs/reviews/capped-harvest-validation-protocol.md`.
Scope: medium, five files. Dependency: 3.

- Use development indices 2100–2119 for four-focal/seven-composition pilot; freeze candidate and protocol before indices 3000–3099.
- Run 100-replicate/four-focal held-out endpoints. Require G, E and at least one endpoint conflict branch in all ecologies, with separate supported verdicts for both returns.
- Publish every condition and its interval. Preserve original results. Add full held-out interior curves only if needed after gate support; label exploratory curves explicitly.

Verification: execute the commands in the specification, inspect completeness,
source/config hashes, figure labels and simultaneous intervals. Publish outcomes
in the run's generated report; a later tracked result review can be a separate
small documentation task.

Checkpoint: **a functioning runner is not a scientific pass**. If this gate is
inconclusive or contradicted, stop adoption and apply the diagnostic rules below.

## 5. Learning and evaluation use the validated utility

Files: `cognitive_tools/experiment.py`, `tests/test_experiment_lifecycle.py`,
`tests/test_reward_modes.py`, `docs/experiment.md`.
Scope: medium, four files. Dependencies: 4, supported gate.

- Add CLI/config reward identity and validation before early returns; forward through training, fresh and carried environment construction.
- Accumulate actual evaluation rewards rather than discard them; discount from the first evaluation step, excluding training wealth.
- Verify real Q updates, fixed controls and both evaluation starts use the same reward; retain physical outcomes separately.

Verification: targeted lifecycle/reward tests, then a tiny new B0 run with the
proposed `--reward-mode capped_harvest` flag. Do not reuse old learned Q tables.

## 6. Prevent cross-reward pooling, resume and schedule reuse

Files: `cognitive_tools/analysis.py`, `cognitive_tools/experiment.py`,
`tests/test_social_analysis.py`, `tests/test_parallel_experiment.py`,
`tests/test_core_social_controls.py`.
Scope: medium, five files. Dependency: 5.

- Canonicalize old missing reward identity to harvest only; compare new explicit reward semantics in treatment compatibility and resume.
- Record/check identity in adaptive and matched-R0 schedules; reject an old-harvest schedule for capped learning.
- Expose utility summaries under the new study while preserving original focused metrics and gross wealth Gini.

Verification: focused compatibility/schedule/resume tests plus full existing
suite; run a tiny adaptive/R0 pair and demonstrate mismatch rejection.
Checkpoint: lint/format, tests and complete new-study lifecycle pass.

## 7. New-study recipe and limited learning pilot

Files: new `scripts/run_capped_campaign.sh`, new `docs/capped-harvest-study.md`,
`README.md` (link only).
Scope: medium, three files. Dependency: 6.

- Provide a bounded recipe with explicit capped reward, separate run names and newly generated R0 schedules; reuse existing CLI rather than fork the engine.
- Start with B0, S1 and one adaptive/matched-R0 pair; decide the broader treatment grid from the research question rather than automatically repeating 27 conditions.
- Document that policy-level dilemma validation, learning behavior and information benefits are separate results. Record pilot cost and feasibility before a full campaign decision.

Verification: execute a smoke recipe, verify its manifest/pairing and utility
outputs, then document the prospective learning pilot. A full campaign is a
separate experiment decision, not part of this planning deliverable.

## Failure diagnosis and revision rules

| Observation | Next development action |
|---|---|
| G fails | Locate unmet all-C demand by occupancy and region; inspect whether preserved stock delivers current utility. Do not swap in reserve welfare as payoff. |
| E fails | Inspect lone restrained agents' opportunities, starting states and paired assignments; retain the actual contrast. |
| Both conflict branches fail | Check scarce allocation and alternative fixed policies. The tested pair may be common-interest; do not force a social-dilemma label. |
| Wide intervals | Use development variance to propose a new fixed replication plan; do not repeatedly add seeds until significance appears. |
| Only one ecology or return passes | Narrow the supported claim explicitly or redesign on development data. The requested all-three/both-return adoption gate has not passed. |
| Utility interpretation is unacceptable | Stop the capped route; use the bounded gross-harvest policy/ecology fallback in the review. |

Any new candidate after seeing confirmation data needs a new protocol and unused
confirmation seeds. Record failures. Larger extraction, stronger coupling, or
a direct high-action cost is not guaranteed to preserve both collective benefit
and unilateral conflict.
