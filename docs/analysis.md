# Analysis

`python -m cognitive_tools.analysis` reads completed experiment directories and
writes tables, figures, and an input manifest without altering source runs.
Mechanisms and outcome definitions are in [experiment.md](experiment.md).

## Input contract

Each `--run` must contain `config.json` and these files under `data/`:

- `evaluation_summary.csv`
- `training_timeseries.csv`
- `policy_summary.csv`
- `agent_social_summary.csv`

Social runs normally also have `network_timeseries.csv` and
`network_edges_checkpoints.csv`; rewiring runs have `rewiring_schedule.csv`.
The required measurement schema is `social_measurement_schema=stage4_v1`.
This versioned identifier protects measurement semantics across datasets.

Analysis refuses incompatible ecological, learner, training-length, recording,
social-capacity, or forecast settings. Theta and mu may vary as analysis factors.
Different scenario/population/replicate coverage warns rather than imputes;
output tables retain actual sample sizes.

## Primary row and contrasts

Cross-treatment comparisons use `strategy=q_learning`,
`evaluation_mode=fresh_reset`, `network_start=terminal`: fresh ecology with the
trained Q table and terminal graph, frozen during evaluation.

Network-memory effects separately subtract `fresh_reset_network` from
`fresh_reset` within the same training run and replicate. The evaluation
semantics and limitations of this contrast are specified in
[the experiment protocol](experiment.md#evaluation-and-network-memory).

## Exact adaptive/R0 pairing

R0's `matched_rewire_schedule_sha256` is matched to the SHA-256 of the loaded
adaptive run's schedule. This disambiguates adaptive runs sharing theta but
using different mu. Replicate rows pair exactly by scenario, population, and
replicate; effects are always adaptive minus matched R0.

The analysis retains a fallback for current datasets whose source schedule hash
cannot be resolved: it uses a unique adaptive run with the same theta. Missing or ambiguous
candidates are skipped with a warning. This input
compatibility path is retained to avoid silently changing the analysis contract;
the campaign rejects manifest warnings, but a unique theta fallback does not
warn. Verify schedule-hash equality when auditing archived inputs. Archive the
adaptive run and R0 together.

## Distributions and bootstrap

One row per replicate is retained for resource distributions. Default final
resource regimes use `R/K < 1/3`, `1/3 <= R/K < 2/3`, and `R/K >= 2/3`.
Explicit `--resource-low-threshold` and `--resource-high-threshold` override these.
Tables report regime probabilities as well as means.

Confidence intervals are percentile bootstraps over replicate observations or
paired replicate differences. Defaults: 2,000 resamples, seed 1729, confidence
95%. Stable derived seeds make tables deterministic for fixed inputs and
settings. `--bootstrap-reps 0` disables resampling for mechanical checks.
Small smoke campaigns do not support scientific inference.

## Outputs

Outputs go to `results/q_learning_baseline/social_analysis/<analysis-name>/`.
`analysis_manifest.json` records input paths, config and schedule hashes,
pairings, bootstrap parameters, resource thresholds, and warnings.

| Table under `data/` | Purpose |
|---|---|
| `run_catalog.csv` | Run settings and provenance |
| `resource_distribution.csv` | Replicate outcomes and resource regimes |
| `resource_distribution_summary.csv` | Distribution summaries and intervals |
| `policy_heatmap_summary.csv` | State-conditioned behavior and occupancy |
| `trajectory_summary.csv` | Ecological and mechanism trajectories |
| `terminal_training_summary.csv` | Training-end diagnostics |
| `paired_adaptive_minus_r0.csv` | Exact paired differences |
| `paired_adaptive_minus_r0_summary.csv` | Paired means and intervals |
| `network_memory_effects.csv` | Within-run topology-memory differences |
| `network_memory_effects_summary.csv` | Network-memory means and intervals |
| `theta_mu_summary.csv` | Parameter-grid summaries |

Figures show resource distributions, state behavior/occupancy heatmaps,
visibility and perception trajectories, degree/action correlations,
welfare/visibility inequality, turnover/resource associations, paired effects,
network memory, theta×mu heatmaps, and local-versus-population observation.
Empty/undefined measurements remain explicit rather than becoming zero effects.

## Invocation and interpretation

```bash
python -m cognitive_tools.analysis --analysis-name paired_example \
    --run results/q_learning_baseline/experiments/adaptive_example \
    --run results/q_learning_baseline/experiments/r0_example
```

The campaign script supplies all treatment paths and verifies its manifest.
Additional compatible parameter runs can extend theta×mu summaries; a single
mu value produces a one-row heatmap. Do not interpret that as a resolved phase
boundary. Distributions and intervals do not establish multimodality or causal
mediation. Causal claims depend on manipulated treatment contrasts; the analysis
adds no distribution-shape hypothesis test.
