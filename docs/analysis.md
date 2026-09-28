# Focused study analysis

The CLI defaults to `--profile focused`. It generates six CSV tables and four
figures (PNG and PDF), using the existing measurements without changing the model.
The older diagnostics below remain available with `--profile diagnostics`.

## Frozen summaries and inference

Primary summaries average recorded training checkpoints satisfying
`max(0, T-1000) < time <= T`: for the full campaign these are steps 4050, 4100,
…, 5000 (20 checkpoints). These are sampled-checkpoint averages, not averages
over every simulated step. Smoke uses all its positive recorded steps.
Each replicate contributes one value per metric, with finite checkpoint counts
reported. Majority mismatch averages the existing non-tie rate over checkpoints
where it is defined; an all-tie window remains undefined, never zero. Report
majority tie rate alongside it. B0 has no social perception or visibility rows.

Secondary resource and reserve welfare average the 1,000-step frozen evaluation;
wealth Gini is its final value. Every summary reports valid and total replicate
counts. Paired contrasts subtract within the same replicate before bootstrapping;
inconsistent replicate sets are rejected.

Intervals are pointwise 95% percentile-bootstrap intervals over independent
replicates: full campaign 5,000 resamples, deterministic analysis seed 1729.
No p-values, significance stars, or equivalence claims are produced. Intervals
are not simultaneous or multiplicity-adjusted; isolated exclusion of zero is
not a family-wide confirmatory finding.

## Outputs

- `primary_window_replicates.csv`, `primary_window_summary.csv`: five metrics,
  including the companion tie rate and checkpoint availability.
- `outcome_replicates.csv`, `outcome_summary.csv`: all reference and rewiring
  conditions, three secondary outcomes.
- `planned_contrasts.csv`, `planned_contrast_summary.csv`: ecology within social
  treatment; S2 minus S1; scope within adaptive/random turnover; and adaptive
  minus exactly paired R0. Explicit left/right columns define subtraction.
- `run_catalog.csv` and `analysis_manifest.json`: input provenance and settings.

The four figures show the mechanism, perception with ties, organization
trajectories, and matched resource/welfare differences. Trajectories are
unsmoothed replicate means, not uncertainty bands. Difference axes share ranges
within each metric across ecologies. Outcome differences use fraction/Gini units,
not percentage points. Reference outcomes remain in the tables. If no matched
pairs are supplied, the outcome figure shows treatment means instead.

Perception/organization associations with consequences are explanatory, not
causal mediation. Network-memory evaluation remains supplementary.

---

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
