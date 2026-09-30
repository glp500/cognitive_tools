# Stage-5 visibility study implementation — 30 September 2026

The [new primary study](../specs/visibility-bounded-search-study.md) is
implemented as a separate Stage-5 protocol under the validated capped-harvest
commons reward. The former Stage-4 hypotheses, treatment codes, campaign script
and focused analysis remain readable as historical designs. The current design
has three ecological settings and five initial visibility profiles crossed with
fixed/adaptive bounded observation, plus ecological baseline B0: 33 conditions.

Network-only calibration sampled 1,000 networks per profile at N=64, k=4 before
any new learning results were inspected. Equal visibility had Gini 0 and no
invisible sources. Average Gini was 0.265 for random, 0.310 for normal centered,
0.396 for low-propensity majority and 0.292 for high-propensity majority.
The average invisible fractions were 0.015, 0.033, 0.084 and 0.026. Normal and
high-majority concentration overlap; the treatment interpretation includes
propensity shape as well as realized Gini. The frozen profile bytes have SHA-256
`c791cf4196deb4da5e1605636afa9fec2ebd2a62ce416c2cc5592c591f8d84c9`.
Raw calibration rows and manifest are generated in
`results/visibility/calibration_v1_final/`.

The clean-source smoke `visibility_smoke_v2` used producer
`1e5631c1c16d929aca9f2d5fb697b3d10852ca16`, N=8, one ecology, two
replicates, 60 training steps and six evaluation steps. All 11 treatments and
22 conditions completed. Every social observer had four sources; fixed and
adaptive pairs used the same initial graph and propensity SHA, and evaluation
utility stayed within the capped-harvest bound. The adaptive treatments
produced one successful rewire across this tiny smoke. The five figures and
PNG/PDF exports were generated and visually inspected. The analysis recorded
10 verified profile/replicate starting graph identities, eight H2 replicate
contrasts and 60 H3 replicate contrasts. The one-ecology smoke cannot yield H1
ecology contrasts; `analysis_health.json` records that as zero rows.

Structured condition events include a correlation ID, duration and failure
type, and each run writes `run_health.json` with coverage and rewiring counts.
The analysis writes its input hashes, analysis code revision and source hashes,
plus an `analysis_health.json` coverage report. Resuming the smoke with the same
clean revision skipped all 11 completed treatments. The full suite passed 225
tests; targeted Stage-5 analysis tests passed after the final figure change.
Ruff lint/format and shell syntax checks passed.
Independent review found and resolved figure, paired-effect, label and
provenance issues.

The smoke proves the implementation path, not H1/H2/H3 scientific findings.
Its two replicates are far too few for substantive inference, and nearly all
fixed/adaptive effects are zero because adaptive rewiring was rare in 60 steps.
The 330 condition-replicate pilot and 3,300 condition-replicate full experiment
have not run. Use
the exact [README commands](../../README.md#run-the-new-experiment) with fresh
tags after reviewing pilot variance, manipulation strength and runtime. Do not
redefine profiles based on pilot scientific outcomes.
