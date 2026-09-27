# Cross-Treatment Social Analysis

`social_experiment_analysis.py` is the Stage 5 analysis pipeline for the
social-information and decentralized-rewiring experiment.

It reads completed experiment directories produced by the Stage 4+ version of
`baseline_validation_experiment.py`. It does not rerun the model or modify
experiment outputs.

## Input contract

Every supplied run must contain:

```text
config.json

data/
    evaluation_summary.csv
    training_timeseries.csv
    policy_summary.csv
    agent_social_summary.csv
```

Social runs additionally normally contain:

```text
network_timeseries.csv
network_edges_checkpoints.csv
rewiring_schedule.csv       # active rewiring only
```

The analysis requires:

```text
social_measurement_schema = stage4_v1
```

This prevents silent mixing of pre-Stage-4 and current measurement semantics.

## Primary evaluation row

Cross-treatment outcome comparisons use:

```text
strategy        = q_learning
evaluation_mode = fresh_reset
network_start   = terminal
```

This is the fresh-ecology, carried-terminal-network, frozen-network evaluation.
It is the existing primary fresh-ecology condition and keeps the learned Q
policy and learned terminal topology together.

The Stage-3 topology-memory decomposition is analyzed separately as:

```text
fresh_reset - fresh_reset_network
```

within the same training run and replicate.

## Run-level distributions

The pipeline retains one row per replicate in:

```text
data/resource_distribution.csv
```

and summarizes it with bootstrap confidence intervals in:

```text
data/resource_distribution_summary.csv
```

By default, final resource regimes use the same thirds as the ecological-state
encoding:

```text
low:     final R/K < 1/3
middle:  1/3 <= final R/K < 2/3
high:    final R/K >= 2/3
```

The thresholds can be changed explicitly with:

```bash
--resource-low-threshold
--resource-high-threshold
```

The analysis reports the probability of low, middle, and high final-resource
regimes instead of relying only on treatment means.

## Bootstrap intervals

Run-level means and paired treatment effects use percentile bootstrap
confidence intervals over replicate-level observations.

Defaults:

```text
bootstrap resamples = 2000
bootstrap seed      = 1729
confidence          = 95%
```

Use `--bootstrap-reps 0` for a mechanical smoke test without resampling.

## Exact adaptive/R0 pairing

For `random_matched` R0 runs, the analysis first matches the R0 input schedule
hash recorded in:

```text
config.json::matched_rewire_schedule_sha256
```

to the SHA-256 digest of the loaded adaptive run's:

```text
data/rewiring_schedule.csv
```

This is preferred over matching only by `theta`, because multiple adaptive
runs can share the same `theta` while using different `mu` values.

If the schedule hash cannot be resolved, a unique adaptive run at the same
`theta` is used only as a fallback. Ambiguous pairs are skipped with an
analysis warning.

Paired replicate-level differences are always defined as:

```text
adaptive - matched R0
```

and are written to:

```text
data/paired_adaptive_minus_r0.csv
data/paired_adaptive_minus_r0_summary.csv
```

## Compatibility checks

The analysis refuses to combine runs that differ in core ecological, learner,
training-length, recording, social-capacity, or forecast settings.

`theta` and `mu` are intentionally allowed to vary because they are analysis
factors.

Different scenario/population/replicate coverage produces a warning rather
than silent imputation. Every output table records its actual sample size.

## Output tables

```text
analysis_manifest.json

data/
    run_catalog.csv
    resource_distribution.csv
    resource_distribution_summary.csv
    policy_heatmap_summary.csv
    trajectory_summary.csv
    terminal_training_summary.csv
    paired_adaptive_minus_r0.csv
    paired_adaptive_minus_r0_summary.csv
    network_memory_effects.csv
    network_memory_effects_summary.csv
    theta_mu_summary.csv
```

`analysis_manifest.json` records input paths, input config SHA-256 hashes,
rewiring-schedule hashes, paired source runs, bootstrap settings, resource
thresholds, and any analysis warnings.

## Figure suite

The Stage 5 script generates:

```text
01_resource_outcome_distributions.png
02_joint_state_behavior_heatmap_<scenario>_N<population>.png
03_joint_state_occupancy_heatmap_<scenario>_N<population>.png
04_visibility_gini_trajectories.png
05_perception_error_trajectories.png
06_degree_action_correlation_trajectories.png
07_wealth_gini_vs_visibility_gini.png
08_turnover_vs_resource_outcome.png
09_paired_adaptive_minus_r0_resource_effects.png
10_network_memory_resource_effects.png
11_theta_mu_resource_heatmap_<scenario>_N<population>.png
11_theta_mu_visibility_gini_heatmap_<scenario>_N<population>.png
12_local_vs_population_low_fraction.png
```

The first eleven correspond directly to the planned core figure suite, with
an additional topology-memory figure from Stage 3. The final scatter directly
visualizes local social sampling relative to population behavior.

## Example core analysis

After running B0, S1, S2, R1-R3 and their matched R0 controls with compatible
settings:

```bash
python social_experiment_analysis.py \
    --analysis-name pilot_core_v1 \
    --run results/q_learning_baseline/experiments/b0_pilot \
    --run results/q_learning_baseline/experiments/s1_pilot \
    --run results/q_learning_baseline/experiments/s2_pilot \
    --run results/q_learning_baseline/experiments/r1_pilot \
    --run results/q_learning_baseline/experiments/r0_matched_r1_pilot \
    --run results/q_learning_baseline/experiments/r2_pilot \
    --run results/q_learning_baseline/experiments/r0_matched_r2_pilot \
    --run results/q_learning_baseline/experiments/r3_pilot \
    --run results/q_learning_baseline/experiments/r0_matched_r3_pilot
```

Outputs are written to:

```text
results/q_learning_baseline/social_analysis/pilot_core_v1/
```

## Theta x mu analysis

The heatmaps include every loaded `prediction_error` run. A core pilot with
only `mu=0.10` therefore produces a one-row phase diagram. A later sweep can
add compatible runs at, for example:

```text
mu = 0.05, 0.10, 0.20
theta = 0, 0.25, 1
```

without changing the analysis code.

## Interpretation boundary

The analysis reports distributions, paired differences, and correlations.
It does not convert a bootstrap interval or scatter plot into a claim of
causal mediation. Causal identification comes from the manipulated treatment
and parameter contrasts; visibility, perception, turnover, welfare, and
resource outcomes are endogenous measurements within those treatments.

No multimodality test is added in Stage 5. If alternative-state claims become
central after the pilot, a dedicated preregistered distribution-shape analysis
can be added without changing the current run-level outputs.
