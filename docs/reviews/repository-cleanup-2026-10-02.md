# Repository cleanup — 2 October 2026

The checkout now contains only the current capacity-matched study's generated
results: the balanced commons-payoff gate, the balanced exploratory pilot, and
the balanced full run. The current study specification, recovery audit, figure
audit, Draft-4 slide guide, and the code needed to run or analyze these remain.

| Retained evidence | Location |
| --- | --- |
| Commons-payoff validation | `results/payoff_validation/balanced_gate_v1/` |
| Pilot freeze, landscape audit, and 11 completed treatment runs | `results/q_learning_baseline/campaigns/balanced_pilot_v1/` and `experiments/balanced_pilot_v1_*` |
| Revised pilot analysis and figures | `results/q_learning_baseline/social_analysis/balanced_pilot_v1_revised_analysis/` |
| Full freeze, landscape audit, and 11 completed treatment runs | `results/q_learning_baseline/campaigns/balanced_full_v1/` and `experiments/balanced_full_v1_*` |
| Full analysis and figures | `results/q_learning_baseline/social_analysis/balanced_full_v1_analysis/` |

An explicit allowlist deleted 161 older result directories containing 8,209
files and 3,354,389,869 bytes (about 3.12 GiB). These included the earlier
ecology full campaign, Stage-5 visibility pilot, first balanced-pilot analysis,
all smoke and diagnostic outputs, earlier payoff gates/pilots, and legacy
presentation exports. The retained results tree is about 4.0 GiB. Every
obsolete tracked result file was removed from the current Git tree, and
`/results/` is now ignored for future campaigns. Git history was not rewritten;
previously tracked artifacts remain accessible from earlier commits. Older
ignored local outputs that were deleted were not in Git.

The older Stage-4/Stage-5 campaign runners, historical figure/review scripts,
completed task lists, superseded design/review documents, and unused payoff
configuration files were retired. Reusable simulation, payoff, visibility,
analysis, and testing modules remain because the current study still calls
them or they form its supported scientific interface. The obsolete test that
read an earlier campaign CSV was removed; tests for active simulation and
payoff behavior remain.

Verification after cleanup:

- The balanced gate validator reports six supported scenario/return cells.
- All 11 pilot and 11 full experiment directories have `complete.json`.
- Both retained analyses report 11 treatments and no undefined primary windows.
- The complete remaining test suite passes: 238 tests.

The [README](../../README.md) contains only the current commands. The
[full-run recovery audit](balanced-full-run-recovery-2026-10-01.md) still records
the three input runs with dirty-worktree metadata; cleanup did not rewrite
scientific results or run provenance.
