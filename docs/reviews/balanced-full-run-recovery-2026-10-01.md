# Balanced full-run analysis recovery — 1 October 2026

The `balanced_full_v1` simulation campaign **completed all 3,300 condition-replicates**: 11 treatment runs each have 300 condition files and a `complete.json` marker. The orchestration stopped **after simulation**, before combined analysis, when the standard compatibility guard saw input run metadata from two Git commits: `6602c8e92ebdf779d373fa36ef4c8302099622e4` and `953f3622d031104a283875f62555e142aa49f680`. No learning simulations need to be rerun to recover the analysis.

The campaign freeze records `6602c8e` as the starting revision. Git's committed diff between the two recorded revisions contains only `.gitignore`, `README.md`, figure rendering/palette code, and documentation/presentation files. No environment, reward, agent, network, visibility, experiment, configuration, or inferential analysis source changed between those commits. All input configs retain the same visibility-profile specification SHA-256 (`c791cf4196deb4da5e1605636afa9fec2ebd2a62ce416c2cc5592c591f8d84c9`). The standard cross-treatment scientific configuration checks remain active.

Three input runs independently record `git_worktree_dirty=true`: `balanced_full_v1_normal_centered_adaptive_bounded`, `balanced_full_v1_low_propensity_majority_fixed`, and `balanced_full_v1_high_propensity_majority_adaptive_bounded`. Their metadata does **not** hash the uncommitted worktree contents at run start. The committed-tree check therefore proves only that the *committed* changes were presentation-only; it cannot retrospectively prove the exact dirty contents. The analysis must retain this warning and the original per-run commit IDs. It must never rewrite input metadata to make the revisions appear identical.

The new `--allow-presentation-only-commit-drift` analysis option is explicit and fails closed: Git must resolve each recorded full commit ID, and every changed path must be on the presentation/documentation allowlist. A changed simulation or inferential source file still causes a hard error. The default behavior remains to reject mixed commits. A recovery helper checks all 11 `complete.json` files, uses the frozen 5,000 bootstrap-resample count, runs the existing visibility analysis, and renders the five story figures:

```bash
PYTHON_BIN=python bash scripts/analyze_balanced_campaign.sh balanced_full_v1 \
    --allow-presentation-only-commit-drift
```

For future campaigns, `run_balanced_campaign.sh` checks the frozen Git revision and clean worktree before and after each treatment. A mid-campaign edit now stops before another treatment starts, rather than surfacing only at combined analysis.

The recovery completed successfully. [`analysis_health.json`](../../results/q_learning_baseline/social_analysis/balanced_full_v1_analysis/analysis_health.json) reports all 11 treatments, 500 verified matched initial graph pairs, 15,300 primary replicate rows, no undefined primary windows, and five main figures. The analysis used 5,000 bootstrap resamples; its [`analysis_manifest.json`](../../results/q_learning_baseline/social_analysis/balanced_full_v1_analysis/analysis_manifest.json) retains both original input commits, all per-run configuration hashes, the three dirty-worktree flags, the explicit recovery option, and the exact changed-path warning. A second recovery pass from committed revision `dfc7625e760496e872d080bf7e5dd6f70f4bbcea` recorded a **clean analysis worktree** and SHA-256 hashes of analysis and figure source files. Five story candidates and their input-hash manifest were rendered. The ten-replicate pilot remains exploratory; these 100 new replicates provide the independent full-study estimates.

The pooled local-view-error results are:

| Test | Full-run estimate | Replicate-bootstrap 95% interval | Reading |
| --- | ---: | ---: | --- |
| H1, primary within-run checkpoint association | Pearson r = −0.088 | −0.111 to −0.066 | Opposite to the positive directional prediction. |
| H2, segregated minus dispersed, averaged across 10 social treatments | +0.100 percentage points | −0.013 to +0.219 pp | The interval crosses zero; directional H2 is not resolved. |
| H3, adaptive minus fixed, averaged across 15 landscape/profile cells | −0.118 percentage points | −0.199 to −0.040 pp | Supports a small reduction in local-view error. |

H1 depends on the comparison scale: the primary raw within-run checkpoint association is negative, whereas a linear time-adjusted checkpoint association is positive (`r = +0.035`, interval `+0.026` to `+0.046`). A secondary association among run-average observations, centered within treatment, is also positive (`r = +0.114`, interval `+0.044` to `+0.184`). Thus the primary H1 prediction is **not supported**, even though its adjusted and between-run sensitivity estimates have the predicted sign. Neither correlation identifies a causal effect. H2 and H3's directional wording was informed by the earlier pilot, so the full run is an independent replication of that wording rather than a strictly preregistered test.

For RQ2, descriptive means across 11 treatment cells per landscape show fresh-evaluation resource fraction `0.192` dispersed versus `0.221` segregated, reserve welfare `0.857` versus `0.860`, and final wealth Gini `0.420` versus `0.443`. These are **descriptive treatment means**, not pooled hypothesis tests. Adaptive observation increased terminal observer-count Gini relative to fixed observation in all 15 matched landscape/profile cells, each with a pointwise interval above zero. Its low-extraction-rate contrasts changed sign across cells, and none of their 15 pointwise intervals excluded zero. Resource persistence is measured over the 1,000-step fresh evaluation, not indefinitely.

The retained dirty-run flags limit the strength of the provenance claim. The results are analyzable and internally pass the recorded health checks, but the original uncommitted source contents cannot be reconstructed from run metadata. A clean-worktree replication is the way to remove that limitation if stronger provenance is required for submission.
