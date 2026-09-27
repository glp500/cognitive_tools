# Stage 5 apply and validation

## Repository changes

Apply:

```text
A  social_experiment_analysis.py
A  tests/test_social_analysis.py
A  SOCIAL_ANALYSIS.md
M  .gitignore
M  README.md
M  PROVENANCE.md
M  SOCIAL_EXPERIMENT.md
D  STAGE3_APPLY.md
```

Do not modify the model, ecology, Q-learning, social mechanism, or canonical
experiment runner for Stage 5. This stage is analysis-only.

`STAGE3_APPLY.md` is an obsolete application guide that was accidentally left
tracked during the Stage 4 changeset. Remove it rather than carrying stage
application notes in the repository.

## Copy files

Copy these from the Stage 5 bundle into the repository:

```text
social_experiment_analysis.py
SOCIAL_ANALYSIS.md
tests/test_social_analysis.py
.gitignore
```

Apply the documentation changes with:

```bash
git apply stage5_docs.patch
```

If you copied the supplied `.gitignore` replacement first, skip the
`.gitignore` hunk in the patch or apply the patch first and then copy the file.

Remove the obsolete guide:

```bash
git rm STAGE3_APPLY.md
```

The helper `run_stage5_analysis_smoke.sh` is supplied for validation. It does
not need to be committed unless you want to keep it as a repository smoke-test
launcher.

## Syntax and tests

```bash
python -m py_compile \
    social_experiment_analysis.py \
    tests/test_social_analysis.py

pytest -q
```

The new analysis tests do not import Mesa or the experiment runner; they build
small synthetic Stage-4-compatible result directories and test analysis
semantics directly.

## Full cross-treatment smoke

From the repository root:

```bash
bash run_stage5_analysis_smoke.sh
```

The helper creates B0, S1, S2, R1-R3 and one matched R0 for each adaptive
`theta`, then runs the cross-treatment analysis.

Expected analysis directory:

```text
results/q_learning_baseline/social_analysis/stage5_core_smoke/
```

At minimum inspect:

```text
analysis_manifest.json

data/run_catalog.csv
data/resource_distribution.csv
data/resource_distribution_summary.csv
data/paired_adaptive_minus_r0.csv
data/paired_adaptive_minus_r0_summary.csv
data/network_memory_effects_summary.csv
data/theta_mu_summary.csv

figures/01_resource_outcome_distributions.png
figures/04_visibility_gini_trajectories.png
figures/05_perception_error_trajectories.png
figures/06_degree_action_correlation_trajectories.png
figures/09_paired_adaptive_minus_r0_resource_effects.png
```

The smoke run uses only two replicates. It validates mechanics and figure
production; it is not scientific evidence.

## Lifecycle checks

For S1 and S2, `network_memory_effects.csv` should report exactly zero
`carried_minus_reset` differences because those treatments do not rewire and
therefore use the same initial and terminal network.

For each R0 row in `run_catalog.csv`, `paired_adaptive_run_id` should identify
the adaptive run whose rewiring schedule was used. Pairing is resolved by
schedule SHA-256 before any theta-only fallback.

## Pilot analysis

After the scientific pilot runs exist, call the same script with every pilot
run directory. For example:

```bash
python social_experiment_analysis.py \
    --analysis-name pilot_core_v1 \
    --bootstrap-reps 2000 \
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

The current pilot grid has one `mu` value, so the theta x mu figures will have
one row. When compatible `mu=0.05` and `mu=0.20` adaptive runs are added, the
same script expands the phase diagram automatically.

## Intended commit

After the complete suite and smoke analysis pass:

```bash
git add \
    .gitignore \
    README.md \
    PROVENANCE.md \
    SOCIAL_EXPERIMENT.md \
    SOCIAL_ANALYSIS.md \
    social_experiment_analysis.py \
    tests/test_social_analysis.py

git commit -m "feat: add cross-treatment social analysis"
```
