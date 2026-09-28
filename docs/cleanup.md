# Scientific cleanup verification

## Frozen baseline and archive

Baseline revision: `b0340b10dd33499a714af5c22ab124b05f680a3a`.
Tag: `pre-cleanup-scientific-snapshot`.
Refactor branch: `refactor/scientific-cleanup`.
No history was rewritten.

Before editing, Python compilation passed and the complete suite passed:
**129 tests, zero failures**, using the existing `eco-marl` Python 3.12 environment.
The shell's default Python did not contain the project dependencies.

| Measure | Before | After |
|---|---:|---:|
| Tracked files | 1,860 | 30 |
| Tracked result files | 1,824 | 0 |
| Tracked source and data size | 135,943,801 bytes | Less than 360 KB |
| Active and test Python lines | 20,378 | 9,049 |
| README lines | 643 | 184 |

Before cleanup, the working tree occupied about 348 MB including approximately
102 MB of Git history and 245 MB of generated results. History is deliberately
retained; deleting results from HEAD does not shrink old Git objects. Smoke
outputs generated during verification are ignored and are not included in the
tracked-size comparison.

The complete results tree, including untracked research files, was archived
outside the source tree before removal:

```text
/home/gavinl/Projects/cognitive-tools-archive/pre-cleanup-b0340b10/
    results.tar.gz
    SHA256SUMS
    source.bundle
    baseline.json
    experiment-help.txt
```

The compressed archive was compared against the original files with `tar -df`
before deletion. Its SHA-256 is:

```text
f75223d8dacf983b93565a3dc09c671382a9ce7aa11f44e00007d369ba90d2d9
```

The source bundle passed `git bundle verify` and contains the snapshot's full
history. This is a verified local archive, not a public research deposit.
The frozen tag also retains all original tracked datasets and historical scripts.

## Behavioral evidence

- All eight scenarios match pre-extraction reference values for capacity,
  regeneration, equilibrium, and region labels on three grid/seed combinations.
  References hash canonical JSON array values, including string region labels.
- Small B0, adaptive, and matched-R0 runs produced **24 byte-identical scientific
  CSV tables** before and after structural migration and formatting.
- AST comparison found no executable changes to definitions in ecological,
  agent-model, or Q-learning modules. Environment changes remove only the unused
  harvest alias; social changes remove the unsupported unmatched random mode.
- The final suite passes **153 tests** in a freshly installed Python 3.12
  environment without inherited packages. Removed tests covered obsolete
  imports and unmatched turnover; the zero-mu invariant still covers adaptive
  rewiring. New checks cover scenarios, package imports, and canonical CLIs.
- Ruff formatting, Ruff lint, compilation, shell syntax, `pip check`, tracked
  artifact scans, and old-name/compatibility audits pass.
- Both editable package installation and console entry points were verified.
  A clean install demonstrated that Mesa requires NetworkX at import time,
  so NetworkX remains pinned; visualization extras and direct GIF dependencies
  were removed.

## End-to-end campaign

The final campaign used the fresh environment and a clean source commit:
`28d9ae0e2a6a2b0f82607aeaa49ebaf1f7e6dce2`.

```bash
bash scripts/run_social_campaign.sh smoke cleanup_final
```

All nine runs completed: B0, S1, S2, R1–R3, and their three matched R0 controls.
The script's tests and manifest checks passed. Independent checks verified
all three schedule hashes, exact per-checkpoint event counts, nonzero rewiring,
clean worktree metadata, and the same Git SHA for all inputs. The analysis
reported no warnings and generated its tables, figures, and manifest at:

```text
results/q_learning_baseline/social_analysis/cleanup_final_analysis/
```

The preceding source commit contains all executable changes; this verification
record is a documentation-only addition. GitHub Actions is configured for the
same installation, format, lint, and test commands. Hosted CI has not run because
this branch has not been pushed.

## Deliberate retained contracts

Scientific behavior, seeds, treatment names, evaluation modes, and measurement
schemas remain unchanged for supported treatments. Historical baseline-only
plotting and campaign scripts are frozen in the snapshot; B0 validation remains
part of the unified runner. Retained output naming and input compatibility are
explained in [provenance](provenance.md) and [analysis](analysis.md).
