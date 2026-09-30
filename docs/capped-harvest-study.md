# Capped-harvest study

The opt-in variant rewards `min(actual harvest, metabolism_rate)` each step.
With the frozen metabolism of 0.002, additional harvest has no current utility.
Physical harvest, wealth, energy, movement and resource dynamics are unchanged.
The default `harvest` objective and original campaign remain available.

The [held-out validation](reviews/capped-harvest-validation-results-2026-09-30.md)
supports the prespecified population social-dilemma criterion in all three
ecologies under both returns. This is evidence about always-low/always-high
policies at N=64 and H=1000; it does not establish learned cooperation or a
benefit from social information.

## Run a separate learning experiment

From a clean committed checkout with the project dependencies installed:

```bash
# Four small lifecycle checks: B0, S1, adaptive R1 and its own matched R0.
PYTHON_BIN=python WORKERS=2 bash scripts/run_capped_campaign.sh smoke capped_smoke_v1

# Prospective exploratory learning pilot; this is a separate, longer run.
PYTHON_BIN=python WORKERS=2 bash scripts/run_capped_campaign.sh pilot capped_learning_pilot_v1
```

The recipe uses seed 20261001, frozen evaluation networks, and newly trained
Q-tables. Smoke uses N=8, two replicates, 12 training steps and six evaluation
steps. Pilot uses N=64, all three ecologies, ten replicates, 5,000 training steps,
1,000 evaluation steps, search-scope theta=0, prediction-error threshold=0.25 and mu=0.10. It is exploratory, not a full
confirmatory campaign. The matched R0 takes event counts from this pilot's own
adaptive schedule. `RESUME=1` reuses completed conditions only when scientific
configuration and producer revision agree. Use a new tag for a new study.

Individual runs can use `python -m cognitive_tools.experiment --reward-mode
capped_harvest` with the ordinary experiment arguments. Every training and
evaluation environment receives that mode. Reward metadata is included in
configurations and rewiring schedules. Cross-objective analysis, schedule reuse,
and checkpoint resume are rejected. Legacy absent reward metadata means harvest;
legacy checkpoints lacking a verifiable signature require a fresh run.

## Read the outputs

Runs live in `results/q_learning_baseline/experiments/<tag>_<treatment>` and
analysis in `results/q_learning_baseline/social_analysis/<tag>_analysis`.
Each run records its configuration, source revision and matched schedule hash.
The analysis manifest records the shared reward definition and input hashes.

`data/evaluation_summary.csv` contains per-agent population means of actual
evaluation utility and gross harvest, both summed and discounted. Discounting
starts at evaluation step zero, including continuation evaluation; training
wealth is never counted as evaluation utility. Existing physical wealth,
reserve welfare, Gini and resource metrics retain their meaning.

The focused analysis adds `data/utility_evaluation.csv` and
`data/utility_evaluation_summary.csv` for fresh-ecology, carried-terminal-network
Q-learning evaluations. Intervals are pointwise replicate bootstrap intervals;
they are descriptive and are not the simultaneous payoff-validation gate.
Legacy harvest runs without these accounts produce no utility summary. Partial
accounts and missing capped evaluation coverage fail explicitly.

## Pilot decision

Inspect utility alongside resource stocks, gross harvest, action frequencies,
and network event counts. Confirm all treatments share the capped objective and
R0 matches its adaptive source. Estimate replicate variability and paired
contrasts before freezing a larger learning study. Do not adjust incentives to
make a desired learning result appear. Failure to learn cooperation despite a
passing payoff gate calls for a learning/observation/exploration diagnosis, not
an automatic reward redesign.

If a new ecology, horizon, policy pair or parameter change fails the payoff gate,
follow the diagnostic branches in the [implementation plan](../tasks/plan.md).
Treat a revised cap or ecology as a new candidate and validate on new held-out
seeds after development. Preserve failed results and their interpretation.

## Implementation verification — 30 September 2026

The clean producer `51c6035` completed `capped_smoke_v3`: B0, S1, R1 (theta=0)
and matched R0, followed by focused analysis. Evaluation utility stayed within
both the gross-harvest and theoretical cap bounds. All 95 matched R0 rewiring
events matched the adaptive source's event counts. The analysis produced 16
utility/gross-harvest summaries, each using two independent replicates.
Resuming recognized all four completed treatments without rerunning simulation.

The full suite passed 209 tests; Ruff lint/format, shell syntax and Git whitespace
checks passed. Independent review issues in provenance and missing-account
handling were resolved with regression tests. Earlier smoke attempts are retained:
v1 completed simulations but had an analysis CLI invocation error; v2 completed
but used the parser's R2 search scope before the recipe explicitly fixed R1.
The final successful reference is v3.

The longer learning pilot and full campaign have **not** been executed. Smoke
results verify operation, not scientific learning effects.
